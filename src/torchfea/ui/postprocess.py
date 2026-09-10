"""In-window solve runner and VTK finite-element postprocessor."""

from __future__ import annotations

import traceback
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from .i18n import T
from .logging_utils import LOGGER, log_path
from .model import CAEProject, build_controller


class _SolveWorker(QObject):
    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self, project: CAEProject):
        super().__init__()
        self.project = project

    def run(self) -> None:
        try:
            LOGGER.info("solve worker start")
            controller = build_controller(self.project)
            result = controller.solve()
            LOGGER.info("solve worker complete: converged=%s total_time=%s", result.converged, result.total_time)
            self.finished.emit(controller, result)
        except Exception:
            LOGGER.exception("solve worker failed")
            self.failed.emit(traceback.format_exc())


class VTKMeshViewer(QFrame):
    """PyVista-backed BREP-like view with VTK picking embedded in Qt.

    PyVista owns the polygonal data model; the stock Qt VTK widget is used for
    embedding and prop/point picking, avoiding an additional pyvistaqt
    dependency.  In selection mode each named torchfea surface is an actor,
    which makes a CAE face pick correspond to a real surface definition.
    """

    entityPicked = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._available = False
        self._renderer = None
        self._interactor = None
        self._selection_mode = "navigate"  # navigate | node | face | body
        self._actor_metadata: dict[int, dict] = {}
        self._actors_by_id: dict[int, object] = {}
        self._brep_vertex_actors: list[object] = []
        self._brep_face_actors: list[object] = []
        self._brep_edge_actors: list[object] = []
        self._wireframe = False
        self._edge_visibility = True
        self._highlight = None
        self._scene_bounds: tuple[float, float, float, float, float, float] | None = None
        self._cube_axes = None
        self._orientation_widget = None
        layout = QVBoxLayout(self)
        try:
            from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
            from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
            from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
            from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkCubeAxesActor
            from vtkmodules.vtkRenderingCore import vtkRenderer
            self._interactor = QVTKRenderWindowInteractor(self)
            self._renderer = vtkRenderer()
            self._renderer.SetBackground(.055, .075, .11)
            self._interactor.GetRenderWindow().AddRenderer(self._renderer)
            # Trackball is position based: the camera follows the mouse drag.
            # The joystick style instead interprets the click position as a
            # velocity, which feels like clicking determines the speed.
            self._interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
            self._interactor.AddObserver("LeftButtonPressEvent", self._pick)
            layout.addWidget(self._interactor)
            self._interactor.Initialize()
            self._cube_axes = vtkCubeAxesActor()
            self._cube_axes.SetCamera(self._renderer.GetActiveCamera())
            self._cube_axes.SetXTitle("X")
            self._cube_axes.SetYTitle("Y")
            self._cube_axes.SetZTitle("Z")
            self._cube_axes.SetXLabelFormat("%.3g")
            self._cube_axes.SetYLabelFormat("%.3g")
            self._cube_axes.SetZLabelFormat("%.3g")
            # ClosestTriad/OuterEdges can move the Z labels to a back edge
            # and make them disappear behind the model. StaticEdges pins all
            # three labelled axes to the viewport-facing cube edges.
            self._cube_axes.SetFlyModeToStaticEdges()
            # In an embedded QVTK view depth-tested 3-D text can be hidden by
            # a BREP face (Z is usually the first axis to disappear).  Keep
            # the axes and ticks in world space, but render their labels in
            # the overlay pass so they stay in front of the model.
            self._cube_axes.SetUse2DMode(True)
            self._cube_axes.SetTickLocationToOutside()
            self._cube_axes.SetLabelOffset(10)
            self._cube_axes.SetTitleOffset((15.0, 15.0))
            self._cube_axes.XAxisVisibilityOn()
            self._cube_axes.YAxisVisibilityOn()
            self._cube_axes.ZAxisVisibilityOn()
            self._cube_axes.XAxisLabelVisibilityOn()
            self._cube_axes.YAxisLabelVisibilityOn()
            self._cube_axes.ZAxisLabelVisibilityOn()
            self._cube_axes.XAxisTickVisibilityOn()
            self._cube_axes.YAxisTickVisibilityOn()
            self._cube_axes.ZAxisTickVisibilityOn()
            self._cube_axes.XAxisMinorTickVisibilityOff()
            self._cube_axes.YAxisMinorTickVisibilityOff()
            self._cube_axes.ZAxisMinorTickVisibilityOff()
            self._cube_axes.DrawXGridlinesOff()
            self._cube_axes.DrawYGridlinesOff()
            self._cube_axes.DrawZGridlinesOff()
            self._cube_axes.GetXAxesTitleProperty().SetColor(.85, .90, .98)
            self._cube_axes.GetYAxesTitleProperty().SetColor(.85, .90, .98)
            self._cube_axes.GetZAxesTitleProperty().SetColor(.85, .90, .98)
            self._cube_axes.GetXAxesLabelProperty().SetColor(.75, .82, .92)
            self._cube_axes.GetYAxesLabelProperty().SetColor(.75, .82, .92)
            self._cube_axes.GetZAxesLabelProperty().SetColor(.75, .82, .92)
            for property_ in (
                self._cube_axes.GetXAxesLabelProperty(),
                self._cube_axes.GetYAxesLabelProperty(),
                self._cube_axes.GetZAxesLabelProperty(),
            ):
                property_.SetFontSize(14)
                property_.BoldOn()
            self._cube_axes.SetPickable(False)
            self._cube_axes.SetVisibility(False)
            self._renderer.AddActor(self._cube_axes)

            # A small fixed triad supplements the world-space cube axes and
            # remains readable while the model is rotated.  It is optional:
            # some software-rendered VTK builds cannot initialize overlay
            # widgets even though the main renderer is usable.
            try:
                marker = vtkAxesActor()
                marker.AxisLabelsOn()
                self._orientation_widget = vtkOrientationMarkerWidget()
                self._orientation_widget.SetOrientationMarker(marker)
                self._orientation_widget.SetInteractor(self._interactor)
                self._orientation_widget.SetViewport(0.0, 0.0, 0.18, 0.18)
                self._orientation_widget.SetEnabled(1)
                self._orientation_widget.InteractiveOff()
            except Exception as marker_exc:
                LOGGER.warning("orientation marker unavailable: %s", marker_exc)
                self._orientation_widget = None
            self._available = True
        except Exception as exc:  # optional rendering stack
            text = QLabel(T("VTK 视图不可用：", "VTK view unavailable: ") + str(exc))
            text.setWordWrap(True)
            text.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(text)

    def set_selection_mode(self, mode: str) -> None:
        """Set ``navigate``, ``node``, ``face``, or ``body`` pick mode."""
        valid_modes = {"navigate", "node", "face", "body"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown viewport selection mode: {mode!r}")
        self._selection_mode = mode
        if self._available:
            # CAD vertices are shown as small white spheres.  They are only
            # pickable in node mode, so a face/body selection cannot be
            # accidentally intercepted by an overlapping vertex glyph.
            for actor in self._brep_vertex_actors:
                actor.SetPickable(self._selection_mode == "node")
            for actor in self._brep_face_actors:
                actor.SetPickable(self._selection_mode != "node")
            for actor in self._brep_edge_actors:
                actor.SetPickable(False)
            cursor = Qt.CursorShape.CrossCursor if self._selection_mode != "navigate" else Qt.CursorShape.OpenHandCursor
            self._interactor.setCursor(cursor)

    def reset_camera(self) -> None:
        """Fit all visible geometry in the current VTK renderer."""
        if self._available and self._renderer is not None:
            self._renderer.ResetCamera()
            self._update_axes()
            self._interactor.GetRenderWindow().Render()

    def clear_scene(self) -> None:
        """Remove all displayed geometry after a failed or invalidated build."""
        if not self._available or self._renderer is None:
            return
        self._renderer.RemoveAllViewProps()
        if self._cube_axes is not None:
            self._renderer.AddActor(self._cube_axes)
            self._cube_axes.SetVisibility(False)
        self._reset_display_state()
        self._scene_bounds = None
        self.set_selection_mode("navigate")
        self._interactor.GetRenderWindow().Render()

    def _reset_display_state(self) -> None:
        """Clear actor bookkeeping before drawing a new viewport product."""
        self._actor_metadata = {}
        self._actors_by_id = {}
        self._brep_vertex_actors = []
        self._brep_face_actors = []
        self._brep_edge_actors = []
        self._highlight = None

    def set_standard_view(self, view: str) -> None:
        """Switch to a named orthographic-style CAD view.

        The camera remains perspective-capable for normal trackball
        navigation, but the standard positions use a stable world-axis up
        vector so Front/Top/Left and the isometric view are repeatable.
        """
        if not self._available or self._renderer is None:
            return
        if self._scene_bounds is None:
            self.reset_camera()
            return
        # The tuple is xmin, xmax, ymin, ymax, zmin, zmax; construct the
        # vectors explicitly rather than relying on non-contiguous slicing.
        minimum = np.array([self._scene_bounds[0], self._scene_bounds[2], self._scene_bounds[4]])
        maximum = np.array([self._scene_bounds[1], self._scene_bounds[3], self._scene_bounds[5]])
        center = (minimum + maximum) * 0.5
        span = max(float(np.max(maximum - minimum)), 1e-6)
        directions = {
            "front": (np.array([0., -1., 0.]), np.array([0., 0., 1.])),
            "back": (np.array([0., 1., 0.]), np.array([0., 0., 1.])),
            "left": (np.array([-1., 0., 0.]), np.array([0., 0., 1.])),
            "right": (np.array([1., 0., 0.]), np.array([0., 0., 1.])),
            "top": (np.array([0., 0., 1.]), np.array([0., 1., 0.])),
            "bottom": (np.array([0., 0., -1.]), np.array([0., -1., 0.])),
            "iso": (np.array([1., -1., 1.]) / np.sqrt(3.), np.array([0., 0., 1.])),
        }
        if view not in directions:
            raise ValueError(f"Unknown standard view: {view!r}")
        direction, up = directions[view]
        camera = self._renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + direction * span * 2.6))
        camera.SetViewUp(*up)
        self._renderer.ResetCameraClippingRange()
        self._update_axes()
        self._interactor.GetRenderWindow().Render()

    def _set_scene_bounds(self, points) -> None:
        values = np.asarray(points, dtype=float).reshape(-1, 3) if len(points) else np.empty((0, 3))
        if not len(values):
            self._scene_bounds = None
            return
        minimum = np.min(values, axis=0)
        maximum = np.max(values, axis=0)
        extent = np.maximum(maximum - minimum, 1e-6)
        # Give the axes a little breathing room without changing the model
        # coordinates represented by their tick labels.
        pad = float(np.max(extent)) * 0.04
        self._scene_bounds = (
            float(minimum[0] - pad), float(maximum[0] + pad),
            float(minimum[1] - pad), float(maximum[1] + pad),
            float(minimum[2] - pad), float(maximum[2] + pad),
        )

    def _update_axes(self) -> None:
        if self._cube_axes is None or self._renderer is None:
            return
        if self._scene_bounds is None:
            self._cube_axes.SetVisibility(False)
            return
        self._cube_axes.SetBounds(*self._scene_bounds)
        self._cube_axes.SetCamera(self._renderer.GetActiveCamera())
        axis_ranges = (
            (self._scene_bounds[0], self._scene_bounds[1]),
            (self._scene_bounds[2], self._scene_bounds[3]),
            (self._scene_bounds[4], self._scene_bounds[5]),
        )
        self._cube_axes.SetXAxisRange(*axis_ranges[0])
        self._cube_axes.SetYAxisRange(*axis_ranges[1])
        self._cube_axes.SetZAxisRange(*axis_ranges[2])
        # Do not rely on VTK's automatic label generation.  Depending on the
        # fly mode it may leave one axis (most often Z) with no label array.
        # Supplying all three arrays makes the numeric ticks deterministic.
        from vtkmodules.vtkCommonCore import vtkStringArray
        for axis, (lower, upper) in enumerate(axis_ranges):
            labels = vtkStringArray()
            for value in np.linspace(lower, upper, 5):
                labels.InsertNextValue(f"{float(value):.3g}")
            self._cube_axes.SetAxisLabels(axis, labels)
        self._cube_axes.SetVisibility(True)

    @staticmethod
    def _normalize_reference_points(reference_points) -> list[dict]:
        """Normalize reference-point records used by the VTK overlay."""
        normalized_points = []
        for reference_point in reference_points or []:
            if isinstance(reference_point, dict):
                point_name = str(reference_point.get("name", "RP"))
                point = reference_point.get("point", reference_point.get("node"))
            else:
                point_name = str(getattr(reference_point, "name", "RP"))
                point = getattr(reference_point, "point", getattr(reference_point, "node", None))
            if point is None:
                continue
            if hasattr(point, "detach"):
                point = point.detach().cpu().numpy()
            values = np.asarray(point, dtype=float).reshape(-1)
            if values.size != 3 or not np.all(np.isfinite(values)):
                LOGGER.warning("ignoring invalid reference point %s: %r", point_name, point)
                continue
            normalized_points.append({"name": point_name, "point": values})
        return normalized_points

    def _add_reference_points(self, reference_points) -> None:
        """Add assembly reference points as visible VTK glyphs."""
        if not self._available or self._renderer is None:
            return
        normalized_points = self._normalize_reference_points(reference_points)
        if not normalized_points:
            return
        from vtkmodules.vtkFiltersSources import vtkSphereSource
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        if self._scene_bounds is None:
            radius = 1e-2
        else:
            span = max(
                self._scene_bounds[1] - self._scene_bounds[0],
                self._scene_bounds[3] - self._scene_bounds[2],
                self._scene_bounds[5] - self._scene_bounds[4],
                1e-6,
            )
            radius = max(float(span) * .018, 1e-4)
        for point_record in normalized_points:
            point = point_record["point"]
            source = vtkSphereSource()
            source.SetCenter(*point)
            source.SetRadius(radius)
            source.SetThetaResolution(16)
            source.SetPhiResolution(12)
            source.Update()
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, .72, .08)
            actor.GetProperty().SetEdgeColor(1.0, .95, .40)
            actor.GetProperty().SetEdgeVisibility(True)
            actor.GetProperty().SetLineWidth(1.5)
            actor.SetPickable(False)
            actor_metadata = {
                "is_reference_point": True,
                "reference_point_name": point_record["name"],
                "point": point.copy(),
            }
            self._renderer.AddActor(actor)
            self._actor_metadata[id(actor)] = actor_metadata
            self._actors_by_id[id(actor)] = actor

    def set_wireframe(self, enabled: bool) -> None:
        """Toggle surface/wireframe representation for CAD or FE actors."""
        self._wireframe = bool(enabled)
        if not self._available:
            return
        for actor, actor_metadata in self._iter_actors():
            if (actor_metadata.get("is_cad_vertex")
                    or actor_metadata.get("is_cad_edge")
                    or actor_metadata.get("is_brep_face")
                    or actor_metadata.get("is_reference_point")):
                continue
            prop = actor.GetProperty()
            if self._wireframe:
                prop.SetRepresentationToWireframe()
            else:
                prop.SetRepresentationToSurface()
        self._interactor.GetRenderWindow().Render()

    def set_edge_visibility(self, enabled: bool) -> None:
        """Toggle edge outlines without affecting CAD vertex markers."""
        self._edge_visibility = bool(enabled)
        if not self._available:
            return
        for actor, actor_metadata in self._iter_actors():
            if actor_metadata.get("is_cad_edge"):
                actor.SetVisibility(self._edge_visibility)
            elif (not actor_metadata.get("is_cad_vertex")
                  and not actor_metadata.get("is_brep_face")
                  and not actor_metadata.get("is_reference_point")):
                actor.GetProperty().SetEdgeVisibility(self._edge_visibility)
        self._interactor.GetRenderWindow().Render()

    def _iter_actors(self):
        for actor_id, actor_metadata in self._actor_metadata.items():
            actor = self._actors_by_id.get(actor_id)
            if actor is not None:
                yield actor, actor_metadata

    def _pick(self, _obj, _event) -> None:
        """Translate one VTK actor/point pick into a CAE entity selection."""
        if not self._available or self._selection_mode == "navigate":
            return
        x, y = self._interactor.GetEventPosition()
        if self._selection_mode == "node":
            from vtkmodules.vtkRenderingCore import vtkPointPicker
            picker = vtkPointPicker()
            picker.Pick(x, y, 0, self._renderer)
            actor = picker.GetActor() if hasattr(picker, "GetActor") else picker.GetViewProp()
            point_id = picker.GetPointId()
            actor_metadata = self._actor_metadata.get(id(actor)) if actor else None
            if actor_metadata is not None and point_id >= 0:
                if "vertex_tag" not in actor_metadata:
                    LOGGER.warning("node pick actor has no OCC vertex tag; ignoring ambiguous pick")
                    return
                self._select_actor(actor)
                self.entityPicked.emit({
                    **actor_metadata, "pick_type": "node", "node_id": int(point_id)})
            return
        # CellPicker tests the actual polygon under the cursor.  PropPicker
        # can select a neighbouring actor from its bounding box, which is
        # especially noticeable on adjacent OCC faces sharing an edge.
        from vtkmodules.vtkRenderingCore import vtkCellPicker
        picker = vtkCellPicker()
        picker.SetTolerance(0.002)
        picker.Pick(x, y, 0, self._renderer)
        actor = picker.GetActor() if hasattr(picker, "GetActor") else picker.GetViewProp()
        actor_metadata = self._actor_metadata.get(id(actor)) if actor else None
        if actor_metadata is None or picker.GetCellId() < 0:
            return
        self._select_actor(actor)
        if (self._selection_mode == "face"
                and (actor_metadata.get("surface") is not None
                     or "surface_tag" in actor_metadata)):
            self.entityPicked.emit({**actor_metadata, "pick_type": "face"})
        elif (self._selection_mode == "body"
              and (actor_metadata.get("body_element_ids") is not None
                   or actor_metadata.get("volume_tags"))):
            self.entityPicked.emit({**actor_metadata, "pick_type": "body"})

    def _select_actor(self, actor) -> None:
        if self._highlight is not None and self._highlight is not actor:
            old_metadata = self._actor_metadata.get(id(self._highlight), {})
            old_property = self._highlight.GetProperty()
            old_property.SetEdgeColor(.1, .1, .1)
            old_property.SetLineWidth(1.0)
            if old_metadata.get("is_cad_vertex"):
                old_property.SetColor(.96, .96, .96)
            elif old_metadata.get("is_brep_face"):
                old_property.SetColor(*old_metadata.get("base_color", (.26, .65, .94)))
        self._highlight = actor
        prop = actor.GetProperty()
        prop.SetEdgeColor(1.0, .88, .15)
        prop.SetLineWidth(3.0)
        actor_metadata = self._actor_metadata.get(id(actor), {})
        if actor_metadata.get("is_cad_vertex"):
            prop.SetColor(1.0, .60, .05)
        elif actor_metadata.get("is_brep_face"):
            # CAD faces do not expose the temporary tessellation edges.  Use
            # a solid red highlight instead of relying on edge visibility.
            prop.SetColor(.92, .10, .08)
            prop.SetEdgeColor(1.0, .35, .20)
        self._interactor.GetRenderWindow().Render()

    def highlight_entity(self, entity: dict | None) -> bool:
        """Highlight a BREP face after the scene has been rebuilt.

        Picking briefly highlights the live VTK actor, but a model-tree edit
        rebuilds the scene.  This metadata-based lookup restores the red face
        without coupling the workbench to VTK actor identities.
        """
        if not self._available or not entity:
            return False
        part_name = entity.get("part_name")
        instance_name = entity.get("instance_name")
        exterior = bool(entity.get("surface_all"))
        if not exterior:
            if entity.get("surface_tag") is None:
                return False
            try:
                surface_tag = int(entity["surface_tag"])
            except (TypeError, ValueError):
                return False
        else:
            surface_tag = None
        matches = []
        for actor, actor_metadata in self._iter_actors():
            if not actor_metadata.get("is_brep_face"):
                continue
            if (not exterior
                    and int(actor_metadata.get("surface_tag", -1)) != surface_tag):
                continue
            if (part_name is not None
                    and actor_metadata.get("part_name") != part_name):
                continue
            # Part Geometry records have no instance name.  They are valid
            # matches for a selection made before switching view products.
            if (instance_name is not None
                    and actor_metadata.get("instance_name") not in (None, instance_name)):
                continue
            matches.append(actor)
        if matches:
            if exterior:
                # The Part exterior is the union of all BREP faces.  Paint
                # every matching face red instead of selecting only the
                # first child actor.
                self._highlight = None
                for actor in matches:
                    prop = actor.GetProperty()
                    prop.SetColor(.92, .10, .08)
                    prop.SetEdgeColor(1.0, .35, .20)
                    prop.SetLineWidth(3.0)
                self._highlight = matches[-1]
                self._interactor.GetRenderWindow().Render()
            else:
                self._select_actor(matches[0])
            return True
        LOGGER.debug("BREP highlight target not found: %s", entity)
        return False

    def show_controller(self, controller, result=None, instance_name: str | None = None,
                        brep: bool = False, reference_points=None) -> None:
        if not self._available:
            return
        assembly = controller.assembly
        if reference_points is None:
            reference_points = []
            for reference_name, reference_point in getattr(assembly, "_reference_points", {}).items():
                reference_points.append({
                    "name": reference_name,
                    "point": getattr(reference_point, "node", None),
                })
        normalized_reference_points = self._normalize_reference_points(reference_points)
        instance_names = list(assembly._instances)
        if not instance_names:
            LOGGER.warning("cannot render FE result: assembly has no instances")
            self.clear_scene()
            return
        if instance_name:
            if instance_name not in assembly._instances:
                raise ValueError(f"Display instance does not exist: {instance_name}")
            selected_instance_names = [instance_name]
        else:
            selected_instance_names = instance_names
        reference_displacement = controller.GC2RGC(result.GC) if result is not None else None
        self._renderer.RemoveAllViewProps()
        if self._cube_axes is not None:
            self._renderer.AddActor(self._cube_axes)
        self._reset_display_state()
        scene_points = []
        if normalized_reference_points:
            scene_points.append(np.asarray(
                [point_record["point"] for point_record in normalized_reference_points],
                dtype=float))
        surface_actor_count = 0
        for instance_index, instance_name in enumerate(selected_instance_names):
            instance = assembly._instances[instance_name]
            part = instance.part
            if part is None:
                try:
                    part = assembly._parts[instance.part_name]
                except KeyError as exc:
                    raise ValueError(
                        f"Instance '{instance_name}' references missing Part '{instance.part_name}'.") from exc
            # A workbench preview builds an Assembly but deliberately does not
            # run Assembly.initialize() (which allocates solver DoFs).
            if instance.part is None:
                instance.part = part
            if not part.surfaces._initialized:
                part.initialize()
            points = instance.nodes.detach().cpu().numpy().copy()
            displacement = np.zeros_like(points)
            if reference_displacement is not None:
                displacement = reference_displacement[instance._RGC_index].detach().cpu().numpy()
                points += displacement
            scene_points.append(points)

            displacement_magnitude = np.linalg.norm(displacement, axis=1)
            exterior_surface = getattr(part, "exterior_surface", "extern")
            if brep:
                # The Part exterior surface is a merged display surface; its
                # child surfaces are the selectable BREP faces.  Do not render
                # both or faces become double layered.
                surface_names = [key for key in part.surfaces.keys() if key != exterior_surface]
                if not surface_names:
                    surface_names = [exterior_surface]
            else:
                surface_names = [exterior_surface]
            for surface_index, surface_name in enumerate(surface_names):
                if surface_name not in part.surfaces:
                    LOGGER.warning("display surface missing: instance=%s part=%s surface=%s",
                                   instance_name, instance.part_name, surface_name)
                    continue
                triangles = part.surfaces.get_trimesh(surface_name).detach().cpu().numpy()
                actor = self._create_surface_actor(
                    points, triangles, displacement_magnitude,
                    undeformed=result is None,
                    color_index=instance_index + surface_index,
                )
                self._renderer.AddActor(actor)
                surface_actor_count += 1
                self._actor_metadata[id(actor)] = {
                    "instance_name": instance_name,
                    "part_name": instance.part_name,
                    "surface": part.surfaces[surface_name],
                    "body_element_ids": np.unique(np.concatenate([
                        element._elems_index.detach().cpu().numpy().ravel()
                        for element in part.elems.values()
                    ])),
                }
                self._actors_by_id[id(actor)] = actor
        if not surface_actor_count:
            raise RuntimeError("No displayable surface is available for the selected FE instance(s).")
        self._set_scene_bounds(np.concatenate(scene_points, axis=0) if scene_points else [])
        self._add_reference_points(normalized_reference_points)
        self._update_axes()
        self._renderer.ResetCamera()
        self._update_axes()
        self._interactor.GetRenderWindow().Render()
        self.set_wireframe(self._wireframe)
        self.set_edge_visibility(self._edge_visibility)

    def show_part(self, part, part_name: str = "Part-1", instance_name: str = "Part-1-1",
                  translation=None, rotation=None, reference_points=None) -> None:
        """Render one already-meshed Part through the normal FE actor path.

        A lightweight Assembly wrapper keeps the rendering semantics identical
        to assembly display without rebuilding the project or invoking a
        solver.  ``show_part`` is deliberately only called after the
        workbench has generated this Part's mesh.
        """
        from torchfea.model import Assembly, Instance

        assembly = Assembly()
        assembly.add_part(part, part_name)
        assembly.add_instance(Instance(
            part_name,
            list([0.0, 0.0, 0.0] if translation is None else translation),
            list([0.0, 0.0, 0.0] if rotation is None else rotation)), instance_name)
        self.show_controller(SimpleNamespace(assembly=assembly), result=None,
                             instance_name=instance_name, brep=False,
                             reference_points=reference_points)

    def show_brep(self, records: list[dict], reference_points=None) -> None:
        """Render unmeshed OCC faces, one actor per selectable BREP face.

        ``records`` are generated by :func:`ui.gmsh_cad.preview_brep`; their
        triangles are only display tessellation, never the analysis mesh.
        """
        if not self._available:
            return
        from vtkmodules.vtkFiltersSources import vtkSphereSource
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        self._renderer.RemoveAllViewProps()
        if self._cube_axes is not None:
            self._renderer.AddActor(self._cube_axes)
        self._reset_display_state()
        scene_points = []
        normalized_reference_points = self._normalize_reference_points(reference_points)
        if normalized_reference_points:
            scene_points.append(np.asarray(
                [point_record["point"] for point_record in normalized_reference_points],
                dtype=float))
        vertex_records: dict[tuple[str, str, int], dict] = {}
        for face_index, face_record in enumerate(records):
            points = np.asarray(face_record["points"], dtype=float)
            triangles = np.asarray(face_record["triangles"], dtype=int)
            if not len(points) or not len(triangles):
                raise ValueError(
                    f"BREP display face {face_record.get('surface_tag', '<unknown>')} is empty.")
            scene_points.append(points)
            actor = self._create_surface_actor(
                points, triangles, np.zeros(len(points)),
                undeformed=True,
                color_index=face_index,
            )
            # BREP faces are opaque in the CAD view; selection is communicated
            # by a solid red highlight.  Keep every unselected face the
            # same neutral CAD color: topology, rather than a repeating
            # palette, is what distinguishes adjacent BREP faces.
            actor.GetProperty().SetColor(.40, .50, .66)
            actor.GetProperty().SetOpacity(1.0)
            actor.GetProperty().SetEdgeVisibility(False)
            actor.SetPickable(self._selection_mode != "node")
            self._renderer.AddActor(actor)
            face_info = dict(face_record)
            face_info["is_brep_face"] = True
            face_info["base_color"] = tuple(actor.GetProperty().GetColor())
            self._actor_metadata[id(actor)] = face_info
            self._actors_by_id[id(actor)] = actor
            self._brep_face_actors.append(actor)
            # OCC vertices are semantically different from display-triangle
            # nodes.  Collect a single glyph per instance/tag, so the point
            # used for a node set is unambiguous even where several faces
            # meet at a corner.
            for vertex_tag, vertex_point in zip(
                    face_record.get("vertex_tags", []), face_record.get("vertex_points", [])):
                vertex_key = (
                    face_record.get("instance_name", ""),
                    face_record.get("part_name", ""),
                    int(vertex_tag),
                )
                vertex_records.setdefault(vertex_key, {
                    "instance_name": face_record.get("instance_name"),
                    "part_name": face_record.get("part_name"),
                    "vertex_tag": int(vertex_tag),
                    "vertex_point": np.asarray(vertex_point, dtype=float),
                    "surface_tag": face_record.get("surface_tag"),
                    "volume_tags": face_record.get("volume_tags", []),
                    "is_cad_vertex": True,
                })
        seen_edges: set[tuple[str, str, int]] = set()
        for face_record in records:
            points = np.asarray(face_record["points"], dtype=float)
            for edge_record in face_record.get("edge_lines", []):
                edge_tag = int(edge_record.get("edge_tag", -1))
                edge_key = (
                    face_record.get("instance_name", ""),
                    face_record.get("part_name", ""),
                    edge_tag,
                )
                lines = np.asarray(edge_record.get("lines", []), dtype=int)
                if edge_tag < 0 or edge_key in seen_edges or not len(points) or not len(lines):
                    continue
                seen_edges.add(edge_key)
                actor = self._create_edge_actor(points, lines)
                actor.GetProperty().SetColor(.96, .96, .98)
                actor.GetProperty().SetLineWidth(2.5)
                actor.SetPickable(False)
                actor.SetVisibility(self._edge_visibility)
                self._renderer.AddActor(actor)
                edge_info = {
                    "instance_name": face_record.get("instance_name"),
                    "part_name": face_record.get("part_name"),
                    "edge_tag": edge_tag,
                    "is_cad_edge": True,
                }
                self._actor_metadata[id(actor)] = edge_info
                self._actors_by_id[id(actor)] = actor
                self._brep_edge_actors.append(actor)
        if vertex_records:
            vertex_points = np.asarray(
                [vertex_record["vertex_point"]
                 for vertex_record in vertex_records.values()], dtype=float)
            vertex_extent = float(np.max(np.ptp(vertex_points, axis=0))) if len(vertex_points) else 1.0
            vertex_radius = max(vertex_extent * .012, 1e-4)
            for vertex_record in vertex_records.values():
                source = vtkSphereSource()
                source.SetCenter(*vertex_record["vertex_point"])
                source.SetRadius(vertex_radius)
                source.SetThetaResolution(12)
                source.SetPhiResolution(12)
                source.Update()
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(source.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(.96, .96, .96)
                actor.GetProperty().SetSpecular(.75)
                actor.GetProperty().SetSpecularPower(18)
                actor.SetPickable(self._selection_mode == "node")
                self._renderer.AddActor(actor)
                self._actor_metadata[id(actor)] = vertex_record
                self._actors_by_id[id(actor)] = actor
                self._brep_vertex_actors.append(actor)
        self._set_scene_bounds(np.concatenate(scene_points, axis=0) if scene_points else [])
        self._add_reference_points(normalized_reference_points)
        self._update_axes()
        self._renderer.ResetCamera()
        self._update_axes()
        self._interactor.GetRenderWindow().Render()
        self.set_wireframe(self._wireframe)
        self.set_edge_visibility(self._edge_visibility)
        self.set_selection_mode(self._selection_mode)

    @staticmethod
    def _create_surface_actor(points, triangles, displacement_magnitude,
                               undeformed: bool, color_index: int):
        """Create a VTK actor from one PyVista surface."""
        import pyvista

        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        faces = np.column_stack(
            (np.full(len(triangles), 3, dtype=np.int64), triangles)).ravel()
        mesh = pyvista.PolyData(points, faces)
        mesh.point_data["Displacement magnitude"] = displacement_magnitude
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.SetScalarRange(
            float(np.min(displacement_magnitude)),
            float(np.max(displacement_magnitude) or 1.0),
        )
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetEdgeVisibility(True)
        actor.GetProperty().SetLineWidth(1.0)
        if undeformed:
            palette = ((.26, .65, .94), (.96, .57, .20), (.34, .78, .49), (.82, .42, .78))
            actor.GetProperty().SetColor(*palette[color_index % len(palette)])
            mapper.ScalarVisibilityOff()
        return actor

    @staticmethod
    def _create_edge_actor(points, lines):
        """Create a lightweight VTK polyline actor for an OCC curve.

        BREP faces are tessellated only to obtain a drawable polygon.  Their
        actual CAD edges are represented separately by this actor, so the
        geometry view never exposes the temporary tessellation as a wireframe.
        ``lines`` contains pairs of point indices into ``points``.
        """
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        vtk_points = vtkPoints()
        for point in points:
            vtk_points.InsertNextPoint(*point)
        cells = vtkCellArray()
        for line in lines:
            if len(line) < 2:
                continue
            cells.InsertNextCell(2)
            cells.InsertCellPoint(int(line[0]))
            cells.InsertCellPoint(int(line[1]))
        poly_data = vtkPolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.SetLines(cells)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtkActor()
        actor.SetMapper(mapper)
        return actor


class PostprocessPanel(QWidget):
    """Solve the submitted project and show its deformed mesh / convergence."""

    backRequested = Signal()
    notify = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: CAEProject | None = None
        self.controller = None
        self.result = None
        self._thread: QThread | None = None
        self._worker: _SolveWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        self.back_button = QPushButton("◀ " + T("返回前处理", "Back to preprocessor"))
        self.back_button.clicked.connect(self.backRequested)
        self.solve_button = QPushButton("▶ " + T("生成模型并计算", "Build model and solve"))
        self.solve_button.setStyleSheet(
            "background:#00796b; font-weight:600; padding:7px 18px")
        self.solve_button.clicked.connect(self.start_solve)
        self.instance_selector = QComboBox()
        self.instance_selector.currentTextChanged.connect(self.refresh_view)
        toolbar.addWidget(self.back_button)
        toolbar.addWidget(self.solve_button)
        toolbar.addSpacing(16)
        toolbar.addWidget(QLabel(T("显示实例", "Display instance")))
        toolbar.addWidget(self.instance_selector)
        toolbar.addStretch(1)
        outer_layout.addLayout(toolbar)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.viewer = VTKMeshViewer()
        content_splitter.addWidget(self.viewer)

        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        title_label = QLabel(T("有限元后处理", "Finite element postprocess"))
        title_label.setStyleSheet("font-size:18px; font-weight:600; color:#86b8ee")
        info_layout.addWidget(title_label)
        self.status_label = QLabel(T("等待计算", "Ready"))
        self.status_label.setWordWrap(True)
        info_layout.addWidget(self.status_label)
        self.statistics_label = QLabel("")
        self.statistics_label.setWordWrap(True)
        self.statistics_label.setStyleSheet("color:#aab5c5")
        info_layout.addWidget(self.statistics_label)
        info_layout.addStretch(1)

        content_splitter.addWidget(info_panel)
        content_splitter.setSizes([900, 320])
        content_splitter.setStretchFactor(0, 1)
        outer_layout.addWidget(content_splitter, 1)

    def set_project(self, project: CAEProject) -> None:
        self.project = project
        self.controller = self.result = None
        self.viewer.clear_scene()
        self.instance_selector.blockSignals(True)
        self.instance_selector.clear()
        self.instance_selector.addItems([instance.name for instance in project.instances])
        self.instance_selector.blockSignals(False)
        self.status_label.setText(T("模型已提交", "Model submitted"))
        self.statistics_label.setText("")

    def start_solve(self) -> None:
        if self.project is None or self._thread is not None:
            return
        self.solve_button.setEnabled(False)
        self.back_button.setEnabled(False)
        self.status_label.setText(T("计算中…", "Solving…"))
        self._thread = QThread(self)
        self._worker = _SolveWorker(self.project)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._solved)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._clean_thread)
        self._thread.start()

    def _solved(self, controller, result) -> None:
        self.controller, self.result = controller, result
        display_ok = self.refresh_view()
        node_count = sum(part.nodes.shape[0] for part in controller.assembly._parts.values())
        element_count = sum(
            element._elems.shape[0]
            for part in controller.assembly._parts.values()
            for element in part.elems.values())
        self.status_label.setText(
            T("计算完成" if display_ok else "计算完成，但结果显示失败",
              "Solve completed" if display_ok else "Solve completed, but result display failed"))
        self.statistics_label.setText(T(
            f"收敛：{result.converged}\n节点：{node_count}\n单元：{element_count}\n耗时：{result.total_time:.3f} s",
            f"Converged: {result.converged}\nNodes: {node_count}\n"
            f"Elements: {element_count}\nTime: {result.total_time:.3f} s"))
        self.notify.emit(T("有限元计算完成", "Finite element solve completed"))

    def _failed(self, detail: str) -> None:
        self.controller = self.result = None
        self.viewer.clear_scene()
        summary = detail.strip().splitlines()[-1] if detail else "Unknown error"
        self.status_label.setText(T("计算失败：", "Solve failed: ") + summary)
        QMessageBox.critical(self, T("计算失败", "Solve failed"),
                             detail + T(f"\n\n详细日志：{log_path()}", f"\n\nDetailed log: {log_path()}"))
        self.notify.emit(T("有限元计算失败", "Finite element solve failed"))

    def _clean_thread(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
        self._worker = None
        self.solve_button.setEnabled(True)
        self.back_button.setEnabled(True)

    def refresh_view(self) -> bool:
        if self.controller is not None:
            try:
                self.viewer.show_controller(
                    self.controller, self.result, self.instance_selector.currentText())
            except Exception as exc:
                self.viewer.clear_scene()
                LOGGER.exception("postprocess view failed")
                self.status_label.setText(
                    T("后处理显示失败：", "Postprocess display failed: ") + str(exc))
                return False
            return True
        return False

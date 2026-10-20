"""In-window solve runner and VTK finite-element postprocessor."""

from __future__ import annotations

import csv
import traceback
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from .i18n import T
from .model import CAEProject


class _SolveWorker(QObject):
    finished = Signal(object, object, object)
    failed = Signal(str)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def run(self) -> None:
        try:
            result = self.controller.solve()
            result_fields = _compute_result_fields(self.controller, result)
            self.finished.emit(self.controller, result, result_fields)
        except Exception:
            self.failed.emit(traceback.format_exc())


RESULT_FIELD_LABELS = {
    "U": ("位移幅值", "Displacement magnitude"),
    "U1": ("U1 位移", "U1 displacement"),
    "U2": ("U2 位移", "U2 displacement"),
    "U3": ("U3 位移", "U3 displacement"),
    "F11": ("F11 变形梯度", "F11 deformation gradient"),
    "F22": ("F22 变形梯度", "F22 deformation gradient"),
    "F33": ("F33 变形梯度", "F33 deformation gradient"),
    "F12": ("F12 变形梯度", "F12 deformation gradient"),
    "F21": ("F21 变形梯度", "F21 deformation gradient"),
    "F23": ("F23 变形梯度", "F23 deformation gradient"),
    "F31": ("F31 变形梯度", "F31 deformation gradient"),
    "F32": ("F32 变形梯度", "F32 deformation gradient"),
    "F13": ("F13 变形梯度", "F13 deformation gradient"),
    "S_Mises": ("Mises 应力", "Mises stress"),
    "S11": ("S11 应力", "S11 stress"),
    "S22": ("S22 应力", "S22 stress"),
    "S33": ("S33 应力", "S33 stress"),
    "S12": ("S12 应力", "S12 stress"),
    "S23": ("S23 应力", "S23 stress"),
    "S13": ("S13 应力", "S13 stress"),
    "E_Mises": ("等效 Green 应变", "Equivalent Green strain"),
    "E11": ("E11 应变", "E11 strain"),
    "E22": ("E22 应变", "E22 strain"),
    "E33": ("E33 应变", "E33 strain"),
    "E12": ("E12 应变", "E12 strain"),
    "E23": ("E23 应变", "E23 strain"),
    "E13": ("E13 应变", "E13 strain"),
    "J": ("体积比 J", "Volume ratio J"),
    "Energy": ("应变能密度", "Strain energy density"),
}


def _finite_numpy(value, size: int | None = None) -> np.ndarray:
    """Convert a tensor/array to finite float data for VTK."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    values = np.asarray(value, dtype=float).reshape(-1)
    if size is not None and values.size != size:
        raise ValueError(f"Expected {size} values, received {values.size}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Result field contains non-finite values.")
    return values


def _compute_result_fields(controller, result) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate nodal result fields from the solved generalized coordinates.

    ``Element_3D`` evaluates constitutive quantities at Gaussian points.  The
    viewer is nodal, so each element value is averaged onto its connectivity;
    this is the same nodal smoothing convention used by most CAE contour
    views and does not change any solver data.
    """
    fields_by_instance: dict[str, dict[str, np.ndarray]] = {}
    rgc = controller.GC2RGC(result.GC)

    for instance_name, instance in controller.assembly._instances.items():
        part = instance.part
        node_count = int(part.nodes.shape[0])
        displacement = rgc[instance._RGC_index]
        if instance.rotation_matrix is not None:
            rotation = instance.rotation_matrix.to(displacement.device, displacement.dtype)
            local_displacement = torch.einsum("ij,aj->ai", rotation.T, displacement)
        else:
            local_displacement = displacement
        local_displacement = local_displacement.detach()
        displacement_np = _finite_numpy(displacement)
        fields: dict[str, np.ndarray] = {
            "U": np.linalg.norm(displacement_np.reshape(-1, 3), axis=1),
            "U1": displacement_np.reshape(-1, 3)[:, 0],
            "U2": displacement_np.reshape(-1, 3)[:, 1],
            "U3": displacement_np.reshape(-1, 3)[:, 2],
        }
        node_sum = {key: np.zeros(node_count, dtype=float) for key in RESULT_FIELD_LABELS
                    if key not in fields}
        node_count_map = np.zeros(node_count, dtype=float)

        for element in part.elems.values():
            connectivity = element._elems[:, :element.num_nodes_per_elem]
            connectivity_np = connectivity.detach().cpu().numpy().astype(int)
            if not len(connectivity_np):
                continue
            element_displacement = local_displacement.to(element.shape_function[0].device)
            deformation_gradient = element.get_deformation_gradient(element_displacement)
            identity = torch.eye(3, device=deformation_gradient.device,
                                 dtype=deformation_gradient.dtype)
            right_cauchy_green = deformation_gradient.transpose(-1, -2) @ deformation_gradient
            green_strain = 0.5 * (right_cauchy_green - identity)
            jacobian = deformation_gradient.det()
            solid_stress, _ = element.components_Solid(element_displacement)
            # The element convention returns P.T; recover P and then the
            # spatial Cauchy stress for a frame-independent contour.
            first_piola = solid_stress.transpose(-1, -2)
            cauchy = (first_piola @ deformation_gradient.transpose(-1, -2)) \
                / jacobian[..., None, None]
            cauchy = 0.5 * (cauchy + cauchy.transpose(-1, -2))
            deviatoric = cauchy - torch.eye(3, device=cauchy.device,
                                             dtype=cauchy.dtype) * cauchy.diagonal(
                                                 dim1=-2, dim2=-1).sum(-1)[..., None, None] / 3.0
            stress_mises = torch.sqrt((1.5 * deviatoric.square().sum((-1, -2))).clamp_min(0.0))
            strain_trace = green_strain.diagonal(dim1=-2, dim2=-1).sum(-1)
            strain_dev = green_strain - torch.eye(3, device=green_strain.device,
                                                   dtype=green_strain.dtype) * strain_trace[..., None, None] / 3.0
            strain_mises = torch.sqrt((2.0 / 3.0 * strain_dev.square().sum((-1, -2))).clamp_min(0.0))
            energy = element.get_potential_energy_density(element_displacement)
            values = {
                "S_Mises": stress_mises.mean(0),
                "S11": cauchy[..., 0, 0].mean(0),
                "S22": cauchy[..., 1, 1].mean(0),
                "S33": cauchy[..., 2, 2].mean(0),
                "S12": cauchy[..., 0, 1].mean(0),
                "S23": cauchy[..., 1, 2].mean(0),
                "S13": cauchy[..., 0, 2].mean(0),
                "E_Mises": strain_mises.mean(0),
                "E11": green_strain[..., 0, 0].mean(0),
                "E22": green_strain[..., 1, 1].mean(0),
                "E33": green_strain[..., 2, 2].mean(0),
                "E12": green_strain[..., 0, 1].mean(0),
                "E23": green_strain[..., 1, 2].mean(0),
                "E13": green_strain[..., 0, 2].mean(0),
                "F11": deformation_gradient[..., 0, 0].mean(0),
                "F22": deformation_gradient[..., 1, 1].mean(0),
                "F33": deformation_gradient[..., 2, 2].mean(0),
                "F12": deformation_gradient[..., 0, 1].mean(0),
                "F21": deformation_gradient[..., 1, 0].mean(0),
                "F23": deformation_gradient[..., 1, 2].mean(0),
                "F31": deformation_gradient[..., 2, 0].mean(0),
                "F32": deformation_gradient[..., 2, 1].mean(0),
                "F13": deformation_gradient[..., 0, 2].mean(0),
                "J": jacobian.mean(0),
                "Energy": energy.mean(0),
            }
            element_values = {
                key: _finite_numpy(value, len(connectivity_np))
                for key, value in values.items()
            }
            flat_nodes = connectivity_np.reshape(-1)
            np.add.at(node_count_map, flat_nodes, 1.0)
            for key, element_value in element_values.items():
                np.add.at(node_sum[key], flat_nodes,
                          np.repeat(element_value, connectivity_np.shape[1]))

        valid_nodes = node_count_map > 0
        for key, values in node_sum.items():
            values[valid_nodes] /= node_count_map[valid_nodes]
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Result field '{key}' contains non-finite values.")
            fields[key] = values
        fields_by_instance[instance_name] = fields
    return fields_by_instance


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
        self._shut_down = False
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
        self._section_plane = None
        self._section_enabled = False
        self._section_axis = "z"
        self._section_fraction = 0.5
        self._scalar_bar = None
        layout = QVBoxLayout(self)
        try:
            from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
            from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
            from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
            from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkCubeAxesActor
            from vtkmodules.vtkCommonDataModel import vtkPlane
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
            self._section_plane = vtkPlane()

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
            except Exception:
                self._orientation_widget = None
            self._available = True
        except Exception as exc:  # optional rendering stack
            text = QLabel(T("VTK 视图不可用：", "VTK view unavailable: ") + str(exc))
            text.setWordWrap(True)
            text.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(text)

    def shutdown(self) -> None:
        """Release VTK/Qt resources before the QApplication is destroyed.

        QVTKRenderWindowInteractor normally finalizes itself from its Qt
        ``closeEvent``.  A parent window can, however, destroy the widget
        after the OpenGL context and Python import machinery have already
        started shutting down.  Explicitly disabling the VTK overlays and
        finalizing the render window while Qt is still alive avoids that
        native teardown race.  The method is intentionally idempotent because
        Qt may also deliver the child widget's close event afterwards.
        """
        if self._shut_down:
            return
        self._shut_down = True
        self._available = False
        orientation_widget = self._orientation_widget
        interactor = self._interactor
        renderer = self._renderer
        self._orientation_widget = None
        self._renderer = None
        self._interactor = None
        self._cube_axes = None
        self._section_plane = None
        self._scalar_bar = None
        self._reset_display_state()
        if orientation_widget is not None:
            try:
                orientation_widget.SetEnabled(0)
            except Exception:
                pass
            try:
                orientation_widget.SetInteractor(None)
            except Exception:
                pass
        if interactor is None:
            return
        # QVTK keeps a QTimer alive for VTK interaction events.  Stop it
        # before removing observers so no callback can enter a half-closed
        # VTK object during the remaining Qt event processing.
        timer = getattr(interactor, "_Timer", None)
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                pass
        try:
            interactor.RemoveAllObservers()
        except Exception:
            pass
        try:
            render_window = interactor.GetRenderWindow()
            if renderer is not None:
                renderer.RemoveAllViewProps()
                render_window.RemoveRenderer(renderer)
            render_window.Finalize()
        except Exception:
            pass
        try:
            interactor.Finalize()
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

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
        self._scalar_bar = None

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

    def set_section(self, enabled: bool, axis: str = "z", fraction: float = 0.5) -> None:
        """Show one side of the current scene through a clipping plane."""
        self._section_enabled = bool(enabled)
        self._section_axis = axis if axis in {"x", "y", "z"} else "z"
        self._section_fraction = float(np.clip(fraction, 0.0, 1.0))
        self._update_section_clip()
        if self._available and self._interactor is not None:
            self._interactor.GetRenderWindow().Render()

    def _update_section_clip(self) -> None:
        if self._section_plane is None or self._scene_bounds is None:
            return
        lower = np.array([self._scene_bounds[0], self._scene_bounds[2], self._scene_bounds[4]])
        upper = np.array([self._scene_bounds[1], self._scene_bounds[3], self._scene_bounds[5]])
        axis_index = {"x": 0, "y": 1, "z": 2}[self._section_axis]
        origin = lower + (upper - lower) * self._section_fraction
        normal = np.zeros(3, dtype=float)
        normal[axis_index] = 1.0
        self._section_plane.SetOrigin(*origin)
        self._section_plane.SetNormal(*normal)
        for actor, metadata in self._iter_actors():
            if (metadata.get("is_reference_point")
                    or metadata.get("is_load_glyph")
                    or metadata.get("is_boundary_glyph")
                    or metadata.get("is_couple_link")):
                continue
            mapper = actor.GetMapper()
            if mapper is None:
                continue
            try:
                mapper.RemoveAllClippingPlanes()
                if self._section_enabled:
                    mapper.AddClippingPlane(self._section_plane)
            except AttributeError:
                pass

    def _add_scalar_bar(self, actor, title: str) -> None:
        if self._renderer is None or actor is None:
            return
        try:
            from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
            scalar_bar = vtkScalarBarActor()
            scalar_bar.SetLookupTable(actor.GetMapper().GetLookupTable())
            # Keep the legend readable in both UI languages.  The mesh data
            # uses stable English keys so result selection remains serializable;
            # only the visible legend is translated here.
            scalar_bar.SetTitle(T(*RESULT_FIELD_LABELS.get(title, (title, title))))
            scalar_bar.SetNumberOfLabels(5)
            scalar_bar.SetWidth(0.12)
            scalar_bar.SetHeight(0.50)
            scalar_bar.GetTitleTextProperty().SetColor(.85, .90, .98)
            scalar_bar.GetLabelTextProperty().SetColor(.75, .82, .92)
            self._renderer.AddActor2D(scalar_bar)
            self._scalar_bar = scalar_bar
        except Exception:
            pass

    @staticmethod
    def _normalize_reference_points(reference_points) -> list[dict]:
        """Normalize reference-point records used by the VTK overlay."""
        normalized_points = []
        for reference_point in reference_points:
            point_name = str(reference_point["name"])
            point = reference_point["point"]
            if hasattr(point, "detach"):
                point = point.detach().cpu().numpy()
            values = np.asarray(point, dtype=float).reshape(-1)
            if values.size != 3 or not np.all(np.isfinite(values)):
                raise ValueError(f"Reference point '{point_name}' must contain three finite coordinates.")
            normalized_points.append({"name": point_name, "point": values})
        return normalized_points

    def _add_reference_points(self, reference_points) -> None:
        """Add assembly reference points as visible VTK glyphs."""
        if not self._available or self._renderer is None:
            return
        normalized_points = self._normalize_reference_points(reference_points or [])
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

    def _add_load_glyphs(self, load_records) -> None:
        """Draw lightweight arrows/links for loads in the active viewport.

        Load glyphs are display-only actors.  They are intentionally not
        pickable and do not participate in wireframe/edge toggles, so a CAD
        face remains the only selectable object in the viewport.
        """
        if not self._available or self._renderer is None:
            return
        records = list(load_records or [])
        if not records:
            return
        span = 1.0
        if self._scene_bounds is not None:
            span = max(
                self._scene_bounds[1] - self._scene_bounds[0],
                self._scene_bounds[3] - self._scene_bounds[2],
                self._scene_bounds[5] - self._scene_bounds[4],
                1e-6,
            )
        default_length = span * .22
        for record in records:
            color = tuple(float(value) for value in record["color"])
            name = str(record["name"])
            if "vector" in record:
                vector = record["vector"]
                origin = record["origin"]
                origin = np.asarray(origin, dtype=float).reshape(-1)
                vector = np.asarray(vector, dtype=float).reshape(-1)
                if origin.size != 3 or vector.size != 3 or not np.all(np.isfinite(origin)):
                    raise ValueError(f"Load glyph '{name}' has invalid vector coordinates.")
                magnitude = float(np.linalg.norm(vector))
                if magnitude <= 1e-12:
                    continue
                direction = vector / magnitude
                length = float(record.get("length", default_length))
                length = max(length, span * .035)
                end = origin + direction * length
                line = self._create_line_actor(origin, end, color, 2.6)
                self._add_load_actor(line, name, color)
                # A cone at the tip makes direction visible even with a
                # dense mesh or a nearly edge-on camera angle.
                from vtkmodules.vtkFiltersSources import vtkConeSource
                from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
                cone_height = max(length * .24, span * .018)
                cone = vtkConeSource()
                cone.SetDirection(*direction)
                cone.SetHeight(cone_height)
                cone.SetRadius(cone_height * .34)
                cone.SetCenter(*(end - direction * cone_height * .5))
                cone.SetResolution(12)
                cone.Update()
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(cone.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(mapper)
                self._add_load_actor(actor, name, color)
                continue
            start = record["start"]
            end = record["end"]
            start = np.asarray(start, dtype=float).reshape(-1)
            end = np.asarray(end, dtype=float).reshape(-1)
            if start.size != 3 or end.size != 3 or not np.all(np.isfinite(np.r_[start, end])):
                raise ValueError(f"Load glyph '{name}' has invalid endpoint coordinates.")
            line = self._create_line_actor(start, end, color, 3.0)
            self._add_load_actor(line, name, color)

    def _add_load_actor(self, actor, name: str, color, line_width: float = 2.5) -> None:
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(float(line_width))
        actor.GetProperty().SetLighting(False)
        actor.SetPickable(False)
        self._renderer.AddActor(actor)
        self._actor_metadata[id(actor)] = {
            "is_load_glyph": True,
            "load_name": name,
            "base_color": color,
        }
        self._actors_by_id[id(actor)] = actor

    def _add_boundary_glyphs(self, boundary_records) -> None:
        """Draw one small three-sided marker at every constrained node."""
        if not self._available or self._renderer is None:
            return
        records = list(boundary_records or [])
        if not records:
            return
        from vtkmodules.vtkFiltersSources import vtkConeSource
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        span = 1.0
        if self._scene_bounds is not None:
            span = max(
                self._scene_bounds[1] - self._scene_bounds[0],
                self._scene_bounds[3] - self._scene_bounds[2],
                self._scene_bounds[5] - self._scene_bounds[4],
                1e-6,
            )
        radius = max(span * .014, 1e-4)
        for record in records:
            point = np.asarray(record.get("point"), dtype=float).reshape(-1)
            if point.size != 3 or not np.all(np.isfinite(point)):
                continue
            marker = vtkConeSource()
            marker.SetCenter(*point)
            marker.SetDirection(0.0, 0.0, 1.0)
            marker.SetRadius(radius)
            marker.SetHeight(radius * 1.8)
            marker.SetResolution(3)
            marker.Update()
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(marker.GetOutputPort())
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, .22, .12)
            actor.GetProperty().SetEdgeColor(1.0, .82, .25)
            actor.GetProperty().SetEdgeVisibility(True)
            actor.GetProperty().SetLineWidth(1.2)
            actor.GetProperty().SetLighting(False)
            actor.SetPickable(False)
            self._renderer.AddActor(actor)
            self._actor_metadata[id(actor)] = {
                "is_boundary_glyph": True,
                "boundary_name": str(record.get("name", "")),
                "dofs": tuple(record.get("dofs", ())),
            }
            self._actors_by_id[id(actor)] = actor

    def _add_couple_links(self, couple_records) -> None:
        """Connect coupled nodes to their reference point with thin lines."""
        if not self._available or self._renderer is None:
            return
        for record in list(couple_records or []):
            target = np.asarray(record.get("reference_point"), dtype=float).reshape(-1)
            points = np.asarray(record.get("points", []), dtype=float).reshape(-1, 3)
            if target.size != 3 or not np.all(np.isfinite(target)):
                continue
            color = tuple(record.get("color", (.95, .75, .15)))
            for point in points:
                if not np.all(np.isfinite(point)):
                    continue
                actor = self._create_line_actor(point, target, color, 1.4)
                actor.SetPickable(False)
                self._renderer.AddActor(actor)
                self._actor_metadata[id(actor)] = {
                    "is_couple_link": True,
                    "constraint_name": str(record.get("name", "")),
                }
                self._actors_by_id[id(actor)] = actor

    @staticmethod
    def _create_line_actor(start, end, color, line_width: float = 2.0):
        from vtkmodules.vtkFiltersSources import vtkLineSource
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        source = vtkLineSource()
        source.SetPoint1(*np.asarray(start, dtype=float))
        source.SetPoint2(*np.asarray(end, dtype=float))
        source.Update()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(float(line_width))
        actor.GetProperty().SetLighting(False)
        return actor

    def set_wireframe(self, enabled: bool) -> None:
        """Toggle surface/wireframe representation for CAD or FE actors."""
        self._wireframe = bool(enabled)
        if not self._available:
            return
        for actor, actor_metadata in self._iter_actors():
            if (actor_metadata.get("is_cad_vertex")
                    or actor_metadata.get("is_cad_edge")
                    or actor_metadata.get("is_brep_face")
                    or actor_metadata.get("is_reference_point")
                    or actor_metadata.get("is_load_glyph")
                    or actor_metadata.get("is_boundary_glyph")
                    or actor_metadata.get("is_couple_link")):
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
                  and not actor_metadata.get("is_reference_point")
                  and not actor_metadata.get("is_load_glyph")
                  and not actor_metadata.get("is_boundary_glyph")
                  and not actor_metadata.get("is_couple_link")):
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
        self._clear_highlight()
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

    def _clear_highlight(self) -> None:
        """Restore every selectable actor before applying a new highlight."""
        for actor, metadata in self._iter_actors():
            prop = actor.GetProperty()
            if metadata.get("is_brep_face"):
                prop.SetColor(*metadata.get("base_color", (.40, .50, .66)))
                prop.SetOpacity(1.0)
                prop.SetEdgeColor(.1, .1, .1)
                prop.SetLineWidth(1.0)
            elif metadata.get("is_cad_vertex"):
                prop.SetColor(*metadata.get("base_color", (.96, .96, .96)))
                prop.SetEdgeColor(.1, .1, .1)
                prop.SetLineWidth(1.0)
        self._highlight = None

    def highlight_entity(self, entity: dict | None) -> bool:
        """Highlight a BREP face after the scene has been rebuilt.

        Picking briefly highlights the live VTK actor, but a model-tree edit
        rebuilds the scene.  This metadata-based lookup restores the red face
        without coupling the workbench to VTK actor identities.
        """
        if not self._available:
            return False
        self._clear_highlight()
        if not entity:
            self._interactor.GetRenderWindow().Render()
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
        self._interactor.GetRenderWindow().Render()
        return False

    def show_controller(self, controller, result=None, instance_name: str | None = None,
                        brep: bool = False, reference_points=None,
                        load_records=None, boundary_records=None, couple_records=None,
                        result_fields=None, field_name: str | None = None,
                        deformed: bool = True, deformation_scale: float = 1.0) -> None:
        if not self._available:
            return
        assembly = controller.assembly
        if reference_points is None:
            reference_points = []
            for reference_name, reference_point in assembly._reference_points.items():
                reference_points.append({
                    "name": reference_name,
                    "point": reference_point.node,
                })
        normalized_reference_points = self._normalize_reference_points(reference_points or [])
        instance_names = list(assembly._instances)
        if not instance_names:
            self.clear_scene()
            return
        if instance_name:
            if instance_name not in assembly._instances:
                raise ValueError(f"Display instance does not exist: {instance_name}")
            selected_instance_names = [instance_name]
        else:
            selected_instance_names = instance_names
        scalar_range = None
        if result is not None and result_fields and field_name:
            selected_values = [
                _finite_numpy(result_fields[name][field_name])
                for name in selected_instance_names
            ]
            combined = np.concatenate(selected_values)
            scalar_range = (float(np.min(combined)), float(np.max(combined)))
        reference_displacement = controller.GC2RGC(result.GC) if result is not None else None
        if reference_displacement is not None and deformed:
            # Reference points are allocated after instances in Assembly's
            # RGC list.  Add the translational part of each RP's solved
            # displacement to its world coordinate for the result overlay.
            for point_record in normalized_reference_points:
                native_point = assembly._reference_points[point_record["name"]]
                rgc_index = native_point._RGC_index
                if not 0 <= int(rgc_index) < len(reference_displacement):
                    raise ValueError(
                        f"Reference point '{point_record['name']}' has an invalid RGC index.")
                displacement = reference_displacement[int(rgc_index)].detach().cpu().numpy().reshape(-1)
                if displacement.size < 3 or not np.all(np.isfinite(displacement[:3])):
                    continue
                displacement = displacement[:3] * float(deformation_scale)
                point_record["point"] = point_record["point"] + displacement
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
                if deformed:
                    points += displacement * float(deformation_scale)
            scene_points.append(points)

            displacement_magnitude = np.linalg.norm(displacement, axis=1)
            exterior_surface = part.exterior_surface
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
                    continue
                triangles = part.surfaces.get_trimesh(surface_name).detach().cpu().numpy()
                scalar_values = None
                if result is not None:
                    scalar_values = displacement_magnitude
                    if result_fields and field_name:
                        scalar_values = result_fields[instance_name][field_name]
                    scalar_values = _finite_numpy(scalar_values, len(points))
                actor = self._create_surface_actor(
                    points, triangles, displacement_magnitude,
                    undeformed=result is None,
                    color_index=instance_index + surface_index,
                    scalar_values=scalar_values,
                    scalar_name=field_name or "Displacement magnitude",
                    scalar_range=scalar_range,
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
        self._add_load_glyphs(load_records)
        self._add_boundary_glyphs(boundary_records)
        self._add_couple_links(couple_records)
        if result is not None and field_name and surface_actor_count:
            first_actor = next(
                (actor for actor, metadata in self._iter_actors()
                 if not metadata.get("is_reference_point")
                 and not metadata.get("is_load_glyph")), None)
            self._add_scalar_bar(first_actor, field_name)
        self._update_axes()
        self._renderer.ResetCamera()
        self._update_axes()
        self._update_section_clip()
        self._interactor.GetRenderWindow().Render()
        self.set_wireframe(self._wireframe)
        self.set_edge_visibility(self._edge_visibility)

    def show_part(self, part, part_name: str = "Part-1", instance_name: str = "Part-1-1",
                  translation=None, rotation=None, reference_points=None,
                  load_records=None, boundary_records=None, couple_records=None) -> None:
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
                             reference_points=reference_points,
                             load_records=load_records,
                             boundary_records=boundary_records,
                             couple_records=couple_records)

    def show_brep(self, records: list[dict], reference_points=None, load_records=None,
                  boundary_records=None, couple_records=None) -> None:
        """Render unmeshed OCC faces, one actor per selectable BREP face.

        ``records`` are generated by :func:`torchfea.cad.preview_brep`; their
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
        normalized_reference_points = self._normalize_reference_points(reference_points or [])
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
                smooth=True,
            )
            # BREP faces are opaque in the CAD view; selection is communicated
            # by a solid red highlight.  Keep every unselected face the
            # same neutral CAD color: topology, rather than a repeating
            # palette, is what distinguishes adjacent BREP faces.
            actor.GetProperty().SetColor(.40, .50, .66)
            actor.GetProperty().SetOpacity(1.0)
            # The temporary triangles only tessellate the OCC face for
            # display.  Smooth interpolation keeps the assembly/Part
            # geometry view visibly BREP-like instead of resembling an FE
            # surface mesh; the actual FE mesh path below remains faceted.
            actor.GetProperty().SetInterpolationToPhong()
            actor.GetProperty().SetSpecular(.12)
            actor.GetProperty().SetSpecularPower(12)
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
        self._add_load_glyphs(load_records)
        self._add_boundary_glyphs(boundary_records)
        self._add_couple_links(couple_records)
        self._update_axes()
        self._renderer.ResetCamera()
        self._update_axes()
        self._update_section_clip()
        self._interactor.GetRenderWindow().Render()
        self.set_wireframe(self._wireframe)
        self.set_edge_visibility(self._edge_visibility)
        self.set_selection_mode(self._selection_mode)

    @staticmethod
    def _create_surface_actor(points, triangles, displacement_magnitude,
                               undeformed: bool, color_index: int,
                               scalar_values=None, scalar_name: str = "Result",
                               scalar_range=None, smooth: bool = False):
        """Create a VTK actor from one PyVista surface."""
        import pyvista

        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        faces = np.column_stack(
            (np.full(len(triangles), 3, dtype=np.int64), triangles)).ravel()
        mesh = pyvista.PolyData(points, faces)
        if smooth:
            # BREP tessellation is only a display representation.  Generate
            # shared point normals so curved OCC faces remain smooth while
            # their separate boundary actors continue to show topology.
            mesh.compute_normals(
                cell_normals=False, point_normals=True,
                split_vertices=False, inplace=True)
        mesh.point_data["Displacement magnitude"] = displacement_magnitude
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetEdgeVisibility(True)
        actor.GetProperty().SetLineWidth(1.0)
        if undeformed:
            palette = ((.26, .65, .94), (.96, .57, .20), (.34, .78, .49), (.82, .42, .78))
            actor.GetProperty().SetColor(*palette[color_index % len(palette)])
            mapper.ScalarVisibilityOff()
        else:
            scalar_values = _finite_numpy(scalar_values if scalar_values is not None
                                          else displacement_magnitude, len(points))
            mesh.point_data[scalar_name] = scalar_values
            mesh.point_data.SetActiveScalars(scalar_name)
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray(scalar_name)
            lower = float(np.min(scalar_values)) if len(scalar_values) else 0.0
            upper = float(np.max(scalar_values)) if len(scalar_values) else 1.0
            if scalar_range is not None:
                lower, upper = scalar_range
            if abs(upper - lower) < 1e-14:
                upper = lower + 1.0
            mapper.SetScalarRange(lower, upper)
            mapper.ScalarVisibilityOn()
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
    importResultRequested = Signal()
    exportResultRequested = Signal()
    notify = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: CAEProject | None = None
        self._submitted_controller = None
        self.controller = None
        self.result = None
        self._result_fields: dict[str, dict[str, np.ndarray]] = {}
        self._solve_summary = ""
        self._instance_summary = ""
        self._solve_metrics = None
        self._thread: QThread | None = None
        self._worker: _SolveWorker | None = None
        self._build_ui()

    def close_cleanup(self) -> None:
        """Release the embedded result viewer before Qt is torn down."""
        self.viewer.shutdown()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        self.back_button = QPushButton("◀ " + T("返回前处理", "Back to preprocessor"))
        self.back_button.clicked.connect(self.backRequested)
        self.solve_button = QPushButton("1 · " + T("开始计算", "Start solve"))
        self.solve_button.setToolTip(T(
            "使用已生成的 FE 网格开始计算",
            "Solve the generated FE mesh"))
        self.solve_button.setStyleSheet(
            "background:#00796b; font-weight:600; padding:7px 18px")
        self.solve_button.clicked.connect(self.start_solve)
        self.solve_button.setEnabled(False)
        self.export_result_button = QPushButton(T("导出结果", "Export result"))
        self.export_result_button.setToolTip(T(
            "导出当前计算结果和结果场",
            "Export the solved result and computed result fields"))
        self.export_result_button.setEnabled(False)
        self.export_result_button.clicked.connect(self.exportResultRequested)
        self.import_result_button = QPushButton(T("导入结果", "Import result"))
        self.import_result_button.setToolTip(T(
            "导入与当前模型匹配的计算结果",
            "Import a result that matches the current model"))
        self.import_result_button.setEnabled(False)
        self.import_result_button.clicked.connect(self.importResultRequested)
        self.instance_label = QLabel(T("显示实例", "Display instance"))
        self.instance_selector = QComboBox()
        self.instance_selector.currentTextChanged.connect(self.refresh_view)
        toolbar.addWidget(self.solve_button)
        toolbar.addWidget(self.import_result_button)
        toolbar.addWidget(self.export_result_button)
        toolbar.addSpacing(16)
        toolbar.addWidget(self.instance_label)
        toolbar.addWidget(self.instance_selector)
        self.field_label = QLabel(T("结果变量", "Result field"))
        self.field_selector = QComboBox()
        self.field_selector.setMinimumWidth(150)
        self.field_selector.setEnabled(False)
        self.field_selector.currentIndexChanged.connect(self.refresh_view)
        toolbar.addWidget(self.field_label)
        toolbar.addWidget(self.field_selector)
        toolbar.addStretch(1)
        outer_layout.addLayout(toolbar)

        display_controls = QHBoxLayout()
        self.deformed_check = QCheckBox(T("显示变形", "Deformed"))
        self.deformed_check.setChecked(True)
        self.deformed_check.setEnabled(False)
        self.deformed_check.toggled.connect(self.refresh_view)
        self.scale_label = QLabel(T("变形倍率", "Scale"))
        self.deformation_scale = QDoubleSpinBox()
        self.deformation_scale.setRange(0.0, 1.0e6)
        self.deformation_scale.setDecimals(3)
        self.deformation_scale.setSingleStep(0.1)
        self.deformation_scale.setValue(1.0)
        self.deformation_scale.setFixedWidth(86)
        self.deformation_scale.setEnabled(False)
        self.deformation_scale.valueChanged.connect(self.refresh_view)
        display_controls.addWidget(self.deformed_check)
        display_controls.addWidget(self.scale_label)
        display_controls.addWidget(self.deformation_scale)
        display_controls.addSpacing(18)
        self.section_label = QLabel(T("剖视", "Section"))
        display_controls.addWidget(self.section_label)
        self.section_enabled = QCheckBox(T("启用", "On"))
        self.section_axis = QComboBox()
        self.section_axis.addItem("X", "x")
        self.section_axis.addItem("Y", "y")
        self.section_axis.addItem("Z", "z")
        self.section_axis.setCurrentIndex(2)
        self.section_position = QSlider(Qt.Orientation.Horizontal)
        self.section_position.setRange(0, 100)
        self.section_position.setValue(50)
        self.section_position.setFixedWidth(130)
        self.section_position.setToolTip(T("拖动剖切面位置", "Move the section plane"))
        display_controls.addWidget(self.section_enabled)
        display_controls.addWidget(self.section_axis)
        display_controls.addWidget(self.section_position)
        display_controls.addStretch(1)
        outer_layout.addLayout(display_controls)

        view_controls = QHBoxLayout()
        self.view_label = QLabel(T("视图", "View"))
        view_controls.addWidget(self.view_label)
        self.fit_view_button = QPushButton(T("充满视图", "Fit view"))
        self.fit_view_button.setToolTip(T(
            "将可见结果居中并缩放到视图内",
            "Center and scale the visible result to the viewport"))
        view_controls.addWidget(self.fit_view_button)
        self._view_buttons = []
        for chinese, english, view_name in (
                ("前", "Front", "front"), ("后", "Back", "back"),
                ("左", "Left", "left"), ("右", "Right", "right"),
                ("顶", "Top", "top"), ("底", "Bottom", "bottom"),
                ("等轴", "Iso", "iso")):
            button = QPushButton(T(chinese, english))
            button.setFixedWidth(48)
            button.clicked.connect(
                lambda _checked=False, mode=view_name: self.viewer.set_standard_view(mode))
            view_controls.addWidget(button)
            self._view_buttons.append((button, chinese, english))
        view_controls.addStretch(1)
        outer_layout.addLayout(view_controls)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.viewer = VTKMeshViewer()
        self.fit_view_button.clicked.connect(self.viewer.reset_camera)
        self.section_enabled.toggled.connect(self._update_section_view)
        self.section_axis.currentIndexChanged.connect(self._update_section_view)
        self.section_position.valueChanged.connect(self._update_section_view)
        content_splitter.addWidget(self.viewer)

        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        self.title_label = QLabel(T("有限元后处理", "Finite element postprocess"))
        self.title_label.setStyleSheet("font-size:18px; font-weight:600; color:#86b8ee")
        info_layout.addWidget(self.title_label)
        self.status_label = QLabel(T("等待提交", "Waiting for submission"))
        self.status_label.setWordWrap(True)
        info_layout.addWidget(self.status_label)
        self.statistics_label = QLabel("")
        self.statistics_label.setWordWrap(True)
        self.statistics_label.setStyleSheet("color:#aab5c5")
        info_layout.addWidget(self.statistics_label)
        self.instance_summary_title = QLabel(T("实例摘要", "Instance summary"))
        self.instance_summary_title.setStyleSheet("font-weight:600; color:#86b8ee")
        info_layout.addWidget(self.instance_summary_title)
        self.instance_summary_label = QLabel("")
        self.instance_summary_label.setWordWrap(True)
        self.instance_summary_label.setStyleSheet("color:#aab5c5")
        info_layout.addWidget(self.instance_summary_label)
        info_layout.addStretch(1)

        content_splitter.addWidget(info_panel)
        content_splitter.setSizes([1100, 320])
        content_splitter.setStretchFactor(0, 1)
        outer_layout.addWidget(content_splitter, 1)

        # Keep page navigation in the footer, matching the MorphOpt observer
        # page: the top bar is reserved for task controls and display scope.
        foot = QHBoxLayout()
        foot.addStretch(1)
        self.back_button.setStyleSheet(
            "background:#37474f; font-weight:600; padding:6px 18px")
        foot.addWidget(self.back_button)
        outer_layout.addLayout(foot)

    def apply_language(self) -> None:
        """Update static page labels without rebuilding the VTK widget."""
        self.back_button.setText("◀ " + T("返回前处理", "Back to preprocessor"))
        self.solve_button.setText("1 · " + T(
            "重新计算" if self.result is not None else "开始计算",
            "Re-solve" if self.result is not None else "Start solve"))
        self.solve_button.setToolTip(T(
            "使用当前已提交模型生成有限元对象并计算",
            "Build the finite-element model from the submitted definition and solve it"))
        self.export_result_button.setText(T("导出结果", "Export result"))
        self.export_result_button.setToolTip(T(
            "导出当前计算结果和结果场",
            "Export the solved result and computed result fields"))
        self.import_result_button.setText(T("导入结果", "Import result"))
        self.import_result_button.setToolTip(T(
            "导入与当前模型匹配的计算结果",
            "Import a result that matches the current model"))
        self.instance_label.setText(T("显示实例", "Display instance"))
        if self.instance_selector.count() and not self.instance_selector.itemData(0):
            self.instance_selector.setItemText(0, T("全部实例", "All instances"))
        self.field_label.setText(T("结果变量", "Result field"))
        self.deformed_check.setText(T("显示变形", "Deformed"))
        self.scale_label.setText(T("变形倍率", "Scale"))
        self.section_label.setText(T("剖视", "Section"))
        self.view_label.setText(T("视图", "View"))
        self.fit_view_button.setText(T("充满视图", "Fit view"))
        self.fit_view_button.setToolTip(T(
            "将可见结果居中并缩放到视图内",
            "Center and scale the visible result to the viewport"))
        for button, chinese, english in self._view_buttons:
            button.setText(T(chinese, english))
            button.setToolTip(T(f"切换到{chinese}视角", f"Switch to {english} view"))
        self.instance_summary_title.setText(T("实例摘要", "Instance summary"))
        self.title_label.setText(T("有限元后处理", "Finite element postprocess"))
        if self._thread is not None:
            self.status_label.setText(T("计算中…", "Solving…"))
        elif self.result is not None:
            self.status_label.setText(T("计算完成", "Solve completed"))
        elif self.project is not None:
            self.status_label.setText(T(
                "模型已提交：点击 1 开始计算",
                "Model submitted: click 1 to start solve"))
        else:
            self.status_label.setText(T("等待提交", "Waiting for submission"))
        self._refresh_field_labels()
        if self._solve_metrics is not None:
            self._solve_summary = self._format_solve_summary()
            self._update_result_statistics()
            self._update_instance_summary()

    def _format_solve_summary(self) -> str:
        if self._solve_metrics is None:
            return ""
        converged, node_count, element_count, total_time, iterations = self._solve_metrics
        return T(
            f"收敛：{converged}\n迭代：{iterations}\n节点：{node_count}\n"
            f"单元：{element_count}\n耗时：{total_time:.3f} s",
            f"Converged: {converged}\nIterations: {iterations}\n"
            f"Nodes: {node_count}\nElements: {element_count}\nTime: {total_time:.3f} s")

    def _refresh_field_labels(self) -> None:
        for index in range(self.field_selector.count()):
            key = self.field_selector.itemData(index)
            if key in RESULT_FIELD_LABELS:
                self.field_selector.setItemText(index, T(*RESULT_FIELD_LABELS[key]))

    def _update_section_view(self, *_args) -> None:
        self.viewer.set_section(
            self.section_enabled.isChecked(),
            str(self.section_axis.currentData() or "z"),
            self.section_position.value() / 100.0,
        )

    def _populate_result_fields(self) -> None:
        previous_field = self.field_selector.currentData()
        self.field_selector.blockSignals(True)
        self.field_selector.clear()
        for key in RESULT_FIELD_LABELS:
            self.field_selector.addItem(T(*RESULT_FIELD_LABELS[key]), key)
        if previous_field in RESULT_FIELD_LABELS:
            self.field_selector.setCurrentIndex(self.field_selector.findData(previous_field))
        self.field_selector.blockSignals(False)
        self.field_selector.setEnabled(bool(self._result_fields))
        self.deformed_check.setEnabled(bool(self._result_fields))
        self.deformation_scale.setEnabled(bool(self._result_fields))

    def _selected_instance_name(self) -> str | None:
        value = self.instance_selector.currentData()
        return str(value) if value else None

    def _update_result_statistics(self) -> None:
        if not self._result_fields:
            self.statistics_label.setText(self._solve_summary)
            return
        field_name = self.field_selector.currentData() or "U"
        instance_name = self._selected_instance_name()
        names = [instance_name] if instance_name else list(self._result_fields)
        arrays = [self._result_fields[name].get(field_name, np.empty(0))
                  for name in names if name in self._result_fields]
        arrays = [np.asarray(values, dtype=float).reshape(-1) for values in arrays if len(values)]
        if not arrays:
            self.statistics_label.setText(self._solve_summary)
            return
        values = np.concatenate(arrays)
        field_label = T(*RESULT_FIELD_LABELS.get(field_name, (field_name, field_name)))
        summary = (
            f"{self._solve_summary}\n"
            f"{field_label}: min={float(np.min(values)):.5g}, "
            f"max={float(np.max(values)):.5g}, avg={float(np.mean(values)):.5g}")
        self.statistics_label.setText(summary)

    def _update_instance_summary(self) -> None:
        """Show compact per-instance extrema beside the active cloud."""
        if self.controller is None or not self._result_fields:
            self._instance_summary = ""
            self.instance_summary_label.setText("")
            return
        rows = []
        for instance_name, instance in self.controller.assembly._instances.items():
            fields = self._result_fields.get(instance_name, {})
            displacement = np.asarray(fields.get("U", np.empty(0)), dtype=float).reshape(-1)
            stress = np.asarray(fields.get("S_Mises", np.empty(0)), dtype=float).reshape(-1)
            displacement_max = float(np.max(displacement)) if displacement.size else 0.0
            stress_max = float(np.max(stress)) if stress.size else 0.0
            element_count = sum(int(element._elems.shape[0]) for element in instance.part.elems.values())
            rows.append(T(
                f"{instance_name}：节点 {instance.part.nodes.shape[0]} · 单元 {element_count}\n"
                f"  |U|max={displacement_max:.5g} · Mises|max={stress_max:.5g}",
                f"{instance_name}: nodes {instance.part.nodes.shape[0]} · elements {element_count}\n"
                f"  |U|max={displacement_max:.5g} · Mises|max={stress_max:.5g}"))
        self._instance_summary = "\n".join(rows)
        self.instance_summary_label.setText(self._instance_summary)

    def set_project(self, project: CAEProject, controller=None) -> None:
        self.project = project
        self._submitted_controller = controller
        self.controller = self.result = None
        self._result_fields = {}
        self._solve_summary = ""
        self._instance_summary = ""
        self._solve_metrics = None
        self.viewer.clear_scene()
        self.instance_selector.blockSignals(True)
        self.instance_selector.clear()
        self.instance_selector.addItem(T("全部实例", "All instances"), "")
        for instance in project.instances:
            self.instance_selector.addItem(instance.name, instance.name)
        self.instance_selector.blockSignals(False)
        self._populate_result_fields()
        self.solve_button.setEnabled(controller is not None)
        self.import_result_button.setEnabled(True)
        self.export_result_button.setEnabled(False)
        self.back_button.setEnabled(True)
        self.solve_button.setText("1 · " + T("开始计算", "Start solve"))
        if controller is None:
            self.status_label.setText(T(
                "请先生成 FE 网格",
                "Generate the FE mesh before solving"))
        else:
            self.status_label.setText(T(
                "模型已提交：点击 1 开始计算",
                "Model submitted: click 1 to start solve"))
        self.statistics_label.setText("")
        self.instance_summary_label.setText("")

    def start_solve(self) -> None:
        if self.project is None or self._submitted_controller is None or self._thread is not None:
            return
        self.solve_button.setEnabled(False)
        self.back_button.setEnabled(False)
        self.status_label.setText(T("计算中…", "Solving…"))
        self._thread = QThread(self)
        self._worker = _SolveWorker(self._submitted_controller)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._solved)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._clean_thread)
        self._thread.start()

    def _solved(self, controller, result, result_fields=None) -> None:
        self.controller, self.result = controller, result
        if not result.model_hash:
            raise ValueError("Solver returned a result without a model hash.")
        self._result_fields = (
            result_fields if result_fields is not None
            else _compute_result_fields(controller, result)
        )
        self._populate_result_fields()
        display_ok = self.refresh_view()
        node_count = sum(part.nodes.shape[0] for part in controller.assembly._parts.values())
        element_count = sum(
            element._elems.shape[0]
            for part in controller.assembly._parts.values()
            for element in part.elems.values())
        self.status_label.setText(
            T("计算完成；可点击 1 重新计算" if display_ok else "计算完成，但结果显示失败",
              "Solve completed; click 1 to re-solve" if display_ok
              else "Solve completed, but result display failed"))
        time_items = getattr(result, "time_items", {}) or {}
        iterations = max((len(values) for values in time_items.values()), default=0)
        self._solve_metrics = (
            result.converged, node_count, element_count, result.total_time, iterations)
        self._solve_summary = self._format_solve_summary()
        self._update_result_statistics()
        self._update_instance_summary()
        self.export_result_button.setEnabled(True)
        self.notify.emit(T("有限元计算完成", "Finite element solve completed"))

    def _failed(self, detail: str) -> None:
        self.controller = self.result = None
        self._result_fields = {}
        self._solve_summary = ""
        self._instance_summary = ""
        self._solve_metrics = None
        self.export_result_button.setEnabled(False)
        self._populate_result_fields()
        self.viewer.clear_scene()
        self.instance_summary_label.setText("")
        summary = detail.strip().splitlines()[-1] if detail else "Unknown error"
        self.status_label.setText(T("计算失败：", "Solve failed: ") + summary)
        QMessageBox.critical(self, T("计算失败", "Solve failed"), detail)
        self.notify.emit(T("有限元计算失败", "Finite element solve failed"))

    def _clean_thread(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
        self._worker = None
        self.solve_button.setEnabled(self.project is not None)
        self.back_button.setEnabled(True)
        self.solve_button.setText("1 · " + T(
            "重新计算" if self.result is not None else "开始计算",
            "Re-solve" if self.result is not None else "Start solve"))

    def refresh_view(self) -> bool:
        if self.controller is not None:
            try:
                self.viewer.show_controller(
                    self.controller, self.result, self._selected_instance_name(),
                    field_name=self.field_selector.currentData() or "U",
                    result_fields=self._result_fields,
                    deformed=self.deformed_check.isChecked(),
                    deformation_scale=self.deformation_scale.value())
                self._update_result_statistics()
            except Exception as exc:
                self.viewer.clear_scene()
                self.status_label.setText(
                    T("后处理显示失败：", "Postprocess display failed: ") + str(exc))
                return False
            return True
        return False

    def export_result(self, path: str) -> None:
        """Export the result through the native result serializer.

        ``.npz`` uses ``StaticResult.save`` and remains readable by the
        solver kernel.  ``.csv`` is a presentation-only table generated by
        the UI for inspection in spreadsheets.
        """
        if self.result is None or self.controller is None:
            raise RuntimeError("No solved result is available for export.")
        current_hash = self.controller.get_model_hash()
        if self.result.model_hash != current_hash:
            current_compatibility_hash = (
                self.controller.get_result_compatibility_hash())
            if self.result.compatibility_hash != current_compatibility_hash:
                raise RuntimeError("The model changed after this result was computed; solve it again.")
        else:
            self.result.model_hash = current_hash
            if not self.result.compatibility_hash:
                self.result.compatibility_hash = (
                    self.controller.get_result_compatibility_hash())
        output = Path(path)
        suffix = output.suffix.casefold()
        if suffix not in {".npz", ".csv"}:
            raise ValueError("Result export must use .npz or .csv.")
        if suffix == ".csv":
            self._export_result_csv(output)
            return
        self.result.save(str(output))

    def _export_result_csv(self, output: Path) -> None:
        field_names = list(RESULT_FIELD_LABELS)
        header = ["model_hash", "instance", "node", "X", "Y", "Z", "Xd", "Yd", "Zd"] + field_names
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for instance_name, instance in self.controller.assembly._instances.items():
                coordinates = instance.nodes.detach().cpu().numpy().reshape(-1, 3)
                fields = self._result_fields[instance_name]
                displacement = np.column_stack([
                    _finite_numpy(fields[key], len(coordinates))
                    for key in ("U1", "U2", "U3")])
                deformed_coordinates = coordinates + displacement
                arrays = {
                    key: _finite_numpy(fields[key], len(coordinates))
                    for key in field_names
                }
                for node_index, (coordinate, deformed) in enumerate(
                        zip(coordinates, deformed_coordinates)):
                    row = [self.result.model_hash, instance_name, node_index,
                           *coordinate, *deformed]
                    row.extend(float(arrays[key][node_index]) for key in field_names)
                    writer.writerow(row)

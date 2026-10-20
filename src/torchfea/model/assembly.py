import numpy as np
import pyvista as pv
import torch

from ..interfaces import Serializable
from . import Instance, ReferencePoint, boundarys, constraints, loads
from .part import Part, _Surfaces


class WorkCondition(Serializable):
    def __init__(self):
        self.load_info: dict[str, torch.Tensor] = {}
        self.ins_enabled: dict[str, bool] = {}
        self.load_enabled: dict[str, bool] = {}
        self.constraint_enabled: dict[str, bool] = {}
        self.boundary_enabled: dict[str, bool] = {}


class Assembly(Serializable):
    _serialized_attributes: list[str] = [
        "_parts",
        "_instances",
        "_surfaces",
        "_reference_points",
        "_loads",
        "_constraints",
        "_boundarys",
    ]

    def __init__(self):

        self.device = torch.zeros(1).device
        """The default device where the tensors are stored (CPU or GPU)."""

        self._parts: dict[str, Part] = {}
        """Dictionary to store parts with part names as keys and Part objects as values."""

        self._instances: dict[str, Instance] = {}
        """Dictionary to store instances with instance names as keys and Instance objects as values."""

        self._surfaces: dict[(str, str), _Surfaces] = {}
        """Dictionary to store surface sets with keys as (instance_name, set_name) and values as Surface objects."""

        self._reference_points: dict[str, ReferencePoint] = {}
        """Dictionary to store reference points with reference point names as keys and ReferencePoint objects as values."""
        self._loads: dict[str, loads.BaseLoad] = {}
        """Dictionary to store loads with load names as keys and Load objects as values."""
        self._constraints: dict[str, constraints.BaseConstraint] = {}
        """Dictionary to store constraints with constraint names as keys and Constraint objects as values."""
        self._boundarys: dict[str, boundarys.BaseBoundary] = {}
        """Dictionary to store boundary conditions with names as keys and Boundary objects as values."""

        self._instances_enabled: list[Instance] = []
        """List to keep track of enabled instances for analysis."""
        self._loads_enabled: list[loads.BaseLoad] = []
        """List to keep track of enabled loads for analysis."""
        self._constraints_forward: list[constraints.BaseConstraint] = []
        """Enabled kinematic constraints in coordinate-reconstruction order."""

        self._constraints_backward: list[constraints.BaseConstraint] = []
        """Enabled kinematic constraints in residual-reduction order."""

        self._boundarys_enabled: list[boundarys.BaseBoundary] = []
        """List to keep track of enabled boundary conditions for analysis."""

        self._RGC: list[torch.Tensor]
        """
        record the redundant generalized coordinates
        """

        self._RGC_size: list[tuple[int]]
        """Record the size of each RGC component
        """

        self._RGC_remain_index: list[np.ndarray]
        """
        record the remaining index of the RGC\n
        """

        self._RGC_remain_index_flatten: torch.Tensor
        """
        record the remaining index of the RGC (flattened)\n
        """

        # initialize the GC (generalized coordinates)
        self._GC: torch.Tensor
        """
        record the generalized coordinates\n
        """

        self._GC_list_indexStart: list[int] = []
        """
        record the start index of the GC\n
        """
        self._RGC_list_indexStart: list[int] = []
        """Record the start index of the RGC\n
        """

        self._mass_matrix_indices: torch.Tensor
        """The indices of the mass matrix"""

        self._mass_matrix_values: torch.Tensor
        """The values of the mass matrix"""

    # region visualization
    def get_meshes(self, GC: torch.Tensor = None) -> dict[str, pv.PolyData]:
        """
        Get the meshes of all instances in the assembly.

        Args:
            GC (torch.Tensor, optional): The generalized coordinates to use for visualization. Defaults to None.

        Returns:
            dict[str, pv.PolyData]: A dictionary with instance names as keys and their corresponding meshes as values.
        """
        meshes = {}
        for ins_name, ins in self._instances.items():
            if GC is not None:
                RGC = self._GC2RGC(GC)
            else:
                RGC = None
            mesh = ins.get_mesh(RGC=RGC)
            meshes[ins_name] = mesh
        return meshes

    def _setup_point_picker(self, plotter: pv.Plotter, meshes: dict[str, pv.PolyData]):
        """Enable point picking that prints the picked point coordinates and index."""

        def callback(picked_point, picker):
            point_id = picker.GetPointId()
            if point_id < 0:
                return
            point = list(meshes.values())[0].points[point_id]
            print(f"Node Index: {point_id}, Coordinates: {point}")
            plotter.add_point_labels(
                [point],
                [f"ID: {point_id}"],
                point_size=20,
                font_size=18,
                name="picked_label",
                always_visible=True,
            )

        plotter.enable_point_picking(
            callback=callback,
            show_message=True,
            use_picker=True,
            show_point=True,
            color="red",
            picker="point",
        )

    def show_ins(self, ins_name: str, GC: torch.Tensor = None, surf_name: str = None):
        """
        Visualize the specified instance.

        Args:
            ins_name (str): The name of the instance to visualize.
            GC (torch.Tensor, optional): The generalized coordinates to use for visualization. Defaults to None.
            surf_name (str, optional): The name of the surface to visualize. Defaults to None.
        """
        if GC is not None:
            RGC = self._GC2RGC(GC)
        else:
            RGC = None
        mesh = self._instances[ins_name].get_mesh(RGC=RGC, surf_name=surf_name)
        pv.global_theme.allow_empty_mesh = True
        plotter = pv.Plotter()
        try:
            plotter.add_mesh(mesh, show_edges=True, opacity=1.0, label=ins_name)
            self._setup_point_picker(plotter, {ins_name: mesh})
            plotter.add_legend()
            plotter.show()
        finally:
            plotter.close()

    def show_all(self, GC: torch.Tensor = None):
        """
        Visualize all instances in the assembly.

        before calling this function, make sure each Part has an exterior
        surface set (``part.exterior_surface``).

        Args:
            GC (torch.Tensor, optional): The generalized coordinates to use for visualization. Defaults to None.
        """
        meshes = self.get_meshes(GC=GC)
        pv.global_theme.allow_empty_mesh = True
        plotter = pv.Plotter()
        try:
            for ins_name, mesh in meshes.items():
                plotter.add_mesh(mesh, show_edges=True, opacity=1.0, label=ins_name)
            self._setup_point_picker(plotter, meshes)
            plotter.add_legend()
            plotter.show()
        finally:
            plotter.close()

    # endregion

    # region Initialization

    class _Initializer:
        @staticmethod
        def _sort_objects(assembly: "Assembly"):
            assembly._parts = dict(sorted(assembly._parts.items()))
            assembly._instances = dict(sorted(assembly._instances.items()))
            assembly._loads = dict(sorted(assembly._loads.items()))
            assembly._constraints = dict(sorted(assembly._constraints.items()))
            assembly._boundarys = dict(sorted(assembly._boundarys.items()))
            assembly._reference_points = dict(
                sorted(assembly._reference_points.items())
            )

        @staticmethod
        def _initialize_instance_with_part(assembly: "Assembly"):
            for ins in assembly._instances.values():
                part_name = ins.part_name
                if part_name not in assembly._parts:
                    raise ValueError(
                        f"Part '{part_name}' not found for instance '{ins}'."
                    )
                ins.part = assembly._parts[part_name]

        @staticmethod
        def _initialize_RGC(assembly: "Assembly"):
            assembly._RGC = []
            assembly._RGC_remain_index = []
            assembly._RGC_list_indexStart = [0]
            assembly._RGC_size = []

            for ins in assembly._instances:
                RGC_index = assembly._allocate_RGC(
                    size=assembly._instances[ins]._RGC_requirements
                )
                assembly._instances[ins].set_RGC_index(RGC_index)

            for rp in assembly._reference_points:
                RGC_index = assembly._allocate_RGC(
                    size=assembly._reference_points[rp]._RGC_requirements
                )
                assembly._reference_points[rp].set_RGC_index(RGC_index)
                assembly._RGC[RGC_index][-1] = 1e-5

            for f in assembly._loads:
                RGC_index = assembly._allocate_RGC(
                    size=assembly._loads[f]._RGC_requirements
                )
                assembly._loads[f].set_RGC_index(RGC_index)

            for c in assembly._constraints:
                RGC_index = assembly._allocate_RGC(
                    size=assembly._constraints[c]._RGC_requirements
                )
                assembly._constraints[c].set_RGC_index(RGC_index)

            for b in assembly._boundarys:
                RGC_index = assembly._allocate_RGC(
                    size=assembly._boundarys[b]._RGC_requirements
                )
                assembly._boundarys[b].set_RGC_index(RGC_index)

        @staticmethod
        def _initialize_objects(assembly: "Assembly"):
            for part in assembly._parts.values():
                part.initialize()

            for ins in assembly._instances.values():
                ins.initialize(assembly)

            for f in assembly._loads.values():
                f.initialize(assembly)

            for c in assembly._constraints.values():
                c.initialize(assembly)

            for b in assembly._boundarys.values():
                b.initialize(assembly)

        @staticmethod
        def _initialize_enabled_objects(assembly: "Assembly"):

            assembly._instances_enabled = []
            for ins in assembly._instances.values():
                if ins.enabled or (ins.enabled is None):
                    assembly._instances_enabled.append(ins)
                    ins.enabled = True  # Set default to True if enabled is None

            assembly._loads_enabled = []
            for f in assembly._loads.values():
                if f.enabled or (f.enabled is None):
                    assembly._loads_enabled.append(f)
                    f.enabled = True  # Set default to True if enabled is None

            # Constraints are represented by the derived forward/backward
            # execution plans rather than a third enabled list.  Normalize
            # the default state here so the plan builder preserves the
            # historical ``enabled=None -> True`` behavior.
            for c in assembly._constraints.values():
                if c.enabled is None:
                    c.enabled = True

            assembly._boundarys_enabled = []
            for b in assembly._boundarys.values():
                if b.enabled or (b.enabled is None):
                    assembly._boundarys_enabled.append(b)
                    b.enabled = True  # Set default to True if enabled is None

        @staticmethod
        def initialize(assembly: "Assembly"):
            Assembly._Initializer._sort_objects(assembly)
            Assembly._Initializer._initialize_instance_with_part(assembly)
            Assembly._Initializer._initialize_RGC(assembly)
            Assembly._Initializer._initialize_objects(assembly)
            Assembly._Initializer._initialize_enabled_objects(assembly)

    def initialize(self, *args, **kwargs):
        """
        Initialize the finite element model.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.

        Returns:
            None
        """

        self._Initializer.initialize(self)
        # define_required_DoFs refreshes enabled objects, builds the
        # dependency plan, and creates the reduced-coordinate maps.
        self.define_required_DoFs()

    def _build_constraint_execution_plan(self):
        """Build deterministic forward and reverse constraint plans.

        ``_constraints`` remains the persistent name-to-object registry.  The
        derived lists below are runtime data.  A constraint whose output RGC
        block is consumed by another constraint must execute first during
        coordinate reconstruction.  Residual and tangent reduction use the
        reverse order.
        """
        enabled = {
            name: constraint
            for name, constraint in self._constraints.items()
            if constraint.enabled
        }

        names = list(enabled.keys())
        graph = {name: set() for name in names}
        indegree = {name: 0 for name in names}

        inputs = {
            name: set(constraint.get_input_blocks())
            for name, constraint in enabled.items()
        }
        outputs = {
            name: set(constraint.get_output_blocks())
            for name, constraint in enabled.items()
        }

        for name_a in names:
            for name_b in names:
                if name_a == name_b:
                    continue

                # A writes a block consumed by B: A -> B.
                if outputs[name_a] & inputs[name_b]:
                    if name_b not in graph[name_a]:
                        graph[name_a].add(name_b)
                        indegree[name_b] += 1

        # Kahn's algorithm.  Sorting the ready queue preserves the assembly's
        # deterministic dictionary order whenever multiple branches are
        # independent.
        ready = sorted(name for name in names if indegree[name] == 0)
        order = []

        while ready:
            name = ready.pop(0)
            order.append(name)

            for target in sorted(graph[name]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
            ready.sort()

        if len(order) != len(names):
            cyclic = [name for name in names if indegree[name] > 0]
            raise ValueError(
                "Kinematic constraint dependency cycle detected among: "
                + ", ".join(cyclic)
            )

        self._constraints_forward = [enabled[name] for name in order]
        self._constraints_backward = list(reversed(self._constraints_forward))

    def define_required_DoFs(self):
        # Rebuild the enabled lists as this method is also used after a work
        # condition changes object enablement.
        self._Initializer._initialize_enabled_objects(self)
        self._build_constraint_execution_plan()

        # This method may be called repeatedly.  Start from a clean mask so a
        # previously enabled constraint cannot leave stale active DoFs behind.
        self._RGC_remain_index = [np.zeros(size, dtype=bool) for size in self._RGC_size]

        for ins in self._instances_enabled:
            self._RGC_remain_index = ins.set_required_DoFs(self._RGC_remain_index)

        for f in self._loads_enabled:
            self._RGC_remain_index = f.set_required_DoFs(self._RGC_remain_index)

        for c in self._constraints_backward:
            self._RGC_remain_index = c.set_required_DoFs(self._RGC_remain_index)

        for b in self._boundarys_enabled:
            self._RGC_remain_index = b.set_required_DoFs(self._RGC_remain_index)

        self._RGC_remain_index_flatten = np.concatenate(
            [
                self._RGC_remain_index[i].reshape(-1)
                for i in range(len(self._RGC_remain_index))
            ]
        ).tolist()
        self._RGC_remain_index_flatten = torch.tensor(
            self._RGC_remain_index_flatten, dtype=torch.bool
        )

        # GC core
        self._GC = self._RGC2GC(self._RGC)
        self._GC_list_indexStart = np.cumsum(
            [
                self._RGC_remain_index[j].sum()
                for j in range(len(self._RGC_remain_index))
            ]
        ).tolist()
        self._GC_list_indexStart.insert(0, 0)

    def initialize_dynamic(self):

        for ins in self._instances_enabled:
            ins.initialize_dynamic()

        for l in self._loads_enabled:
            l.initialize_dynamic()

        for c in self._constraints_forward:
            c.initialize_dynamic()

        # assemble the redundant mass matrix
        mass_indices = []
        mass_values = []
        for ins in self._instances_enabled:
            indices_now, values_now = ins.get_mass_matrix()
            mass_indices.append(indices_now)
            mass_values.append(values_now)
        self._mass_matrix_indices = torch.cat(mass_indices, dim=1)
        self._mass_matrix_values = torch.cat(mass_values, dim=0)

    def reinitialize(self, RGC: list[torch.Tensor]):
        """
        Reinitializes the finite element analysis problem.

        Args:
            RGC (list[torch.Tensor]): The redundant generalized coordinates.
        """
        self._RGC = RGC
        self._GC = self._RGC2GC(self._RGC)

        for ins in self._instances_enabled:
            ins.reinitialize(RGC)

        for l in self._loads_enabled:
            l.reinitialize(RGC)

        for c in self._constraints_forward:
            c.reinitialize(RGC)

    # endregion

    # region Stiffness Matrix Assembly

    def assemble_force(
        self, RGC: list[torch.Tensor] = None, GC: torch.Tensor = None
    ) -> torch.Tensor:

        if RGC is None:
            if GC is None:
                raise ValueError("Either RGC or GC must be provided.")
            RGC = self._GC2RGC(GC)

        # region evaluate the structural K and R

        R_values = []
        R_indices = []

        for ins in self._instances_enabled:
            Ra_indice, Ra_values = ins.structural_stiffness(RGC=RGC, if_onlyforce=True)
            R_values.append(Ra_values)
            R_indices.append(Ra_indice)

        ff = []
        for f in self._loads_enabled:
            Rf_indice, Rf_values = f.get_stiffness(RGC=RGC, if_onlyforce=True)
            R_values.append(-Rf_values)
            R_indices.append(Rf_indice)

            ff.append(
                torch.zeros(self._RGC_list_indexStart[-1]).scatter_add_(
                    0, Rf_indice.to(torch.int64), Rf_values
                )
            )

        # endregion

        R_indices = torch.cat(R_indices, dim=0)
        R_values = torch.cat(R_values, dim=0)

        R0 = torch.zeros(self._RGC_list_indexStart[-1])
        # Convert R_indices to int64 explicitly for scatter operation
        R0.scatter_add_(0, R_indices.to(torch.int64), R_values)
        R = R0
        # region consider the constraints
        # Reduce coordinate transformations from leaves back to masters.  The
        # accumulated residual is passed on so a downstream constraint can
        # consume force already transferred by an upstream reduction.
        for c in self._constraints_backward:
            R_new = c.modify_R_K(RGC, R, if_onlyforce=True)
            R = R + R_new
        # endregion

        # get the global stiffness matrix and force vector

        R = R[self._RGC_remain_index_flatten]

        return R

    def assemble_Stiffness_Matrix(
        self, RGC: list[torch.Tensor] = None, GC: torch.Tensor = None
    ):
        """
        Assemble the stiffness matrix.

        Args:
            RGC (list[torch.Tensor]): The redundant generalized coordinates.
            GC (torch.Tensor, optional): The generalized coordinates. If provided, it will be converted to RGC internally. Defaults to None.

        Returns:
            tuple: A tuple containing:
                R (torch.Tensor): The global force vector.
                K_indices (torch.Tensor): The indices of the global stiffness matrix.
                K_values (torch.Tensor): The values of the global stiffness matrix.
        """

        if RGC is None:
            if GC is None:
                raise ValueError("Either RGC or GC must be provided.")
            RGC = self._GC2RGC(GC)

        # region evaluate the structural K and R
        R0, K_indices, K_values = self._assemble_generalized_Matrix(RGC)
        # endregion
        R, K_indices, K_values = self._assemble_reduced_Matrix(
            RGC, R0, K_indices, K_values
        )

        return R, K_indices, K_values

    def _assemble_generalized_Matrix(
        self, RGC: list[torch.Tensor] = None, GC: torch.Tensor = None
    ):
        if RGC is None:
            if GC is None:
                raise ValueError("Either RGC or GC must be provided.")
            RGC = self._GC2RGC(GC)

        # region evaluate the structural K and R
        K_values = []
        K_indices = []
        R_values = []
        R_indices = []

        for ins in self._instances_enabled:
            Ra_indice, Ra_values, Ka_indice, Ka_value = ins.structural_stiffness(
                RGC=RGC
            )
            K_values.append(Ka_value)
            K_indices.append(Ka_indice)
            R_values.append(Ra_values)
            R_indices.append(Ra_indice)

        ff = []
        for f in self._loads_enabled:
            Rf_indice, Rf_values, Kf_indice, Kf_value = f.get_stiffness(RGC=RGC)
            K_values.append(-Kf_value)
            K_indices.append(Kf_indice)
            R_values.append(-Rf_values)
            R_indices.append(Rf_indice)

            ff.append(
                torch.zeros(self._RGC_list_indexStart[-1]).scatter_add_(
                    0, Rf_indice.to(torch.int64), Rf_values
                )
            )
        # endregion

        K_indices = torch.cat(K_indices, dim=1)
        K_values = torch.cat(K_values, dim=0)
        R_indices = torch.cat(R_indices, dim=0)
        R_values = torch.cat(R_values, dim=0)

        R0 = torch.zeros(self._RGC_list_indexStart[-1])
        # Convert R_indices to int64 explicitly for scatter operation
        R0.scatter_add_(0, R_indices.to(torch.int64), R_values)
        return R0, K_indices, K_values

    def _assemble_reduced_Matrix(
        self,
        RGC: list[torch.Tensor],
        R0: torch.Tensor,
        K_indices: torch.Tensor,
        K_values: torch.Tensor,
    ):

        R = R0
        # region consider the constraints
        # Residual/tangent reduction follows the reverse dependency order.
        # K_indices/K_values already contain contributions added by previous
        # constraints in this loop.
        for c in self._constraints_backward:
            R_new, Kc_indices, Kc_values = c.modify_R_K(RGC, R, K_indices, K_values)
            K_indices = torch.cat([K_indices, Kc_indices], dim=1)
            K_values = torch.cat([K_values, Kc_values])
            R = R + R_new

        # endregion

        # get the global stiffness matrix and force vector
        index_remain = (
            self._RGC_remain_index_flatten[K_indices[0].cpu()]
            & self._RGC_remain_index_flatten[K_indices[1].cpu()]
        )
        K_values = K_values[index_remain]
        K_indices = K_indices[:, index_remain]

        K_indices[0] = K_indices[0].unique(return_inverse=True)[1]
        K_indices[1] = K_indices[1].unique(return_inverse=True)[1]

        R = R[self._RGC_remain_index_flatten]

        return R, K_indices, K_values

    def _total_Potential_Energy(
        self, RGC: list[torch.Tensor] = None, GC: torch.Tensor = None
    ) -> float:
        """
        Calculate the total potential energy of the finite element model.

        Args:
            RGC (list[torch.Tensor]): The redundant generalized coordinates.

        Returns:
            float: The total potential energy.
        """

        if RGC is None:
            if GC is None:
                raise ValueError("Either RGC or GC must be provided.")
            RGC = self._GC2RGC(GC)

        # structural energy
        energy = 0
        for ins in self._instances_enabled:
            energy = energy + ins.potential_energy(RGC=RGC)

        # force potential
        for f in self._loads_enabled:
            energy = energy - f.get_potential_energy(RGC=RGC)

        return energy

    # endregion

    # region for Dynamic Mass Matrix

    def assemble_mass_matrix(self, GC_now: torch.Tensor):
        mass_indices = [self._mass_matrix_indices]
        mass_values = [self._mass_matrix_values]
        RGC = self._GC2RGC(GC_now)

        current_indices = self._mass_matrix_indices
        current_values = self._mass_matrix_values

        # Mass transformations use the same reverse dependency direction as
        # stiffness reduction and consume the accumulated matrix.
        for c in self._constraints_backward:
            transformed = c.modify_mass_matrix(
                mass_indices=current_indices,
                mass_values=current_values,
                RGC=RGC,
            )
            if transformed is None:
                raise RuntimeError(
                    f"Constraint '{c.__class__.__name__}' returned None "
                    "from modify_mass_matrix()."
                )

            indices_now, values_now = transformed
            mass_indices.append(indices_now)
            mass_values.append(values_now)

            current_indices = torch.cat(mass_indices, dim=1)
            current_values = torch.cat(mass_values, dim=0)

        mass_indices = torch.cat(mass_indices, dim=1)
        mass_values = torch.cat(mass_values, dim=0)

        # get the global stiffness matrix and force vector
        index_remain = (
            self._RGC_remain_index_flatten[mass_indices[0].cpu()]
            & self._RGC_remain_index_flatten[mass_indices[1].cpu()]
        )
        mass_values = mass_values[index_remain]
        mass_indices = mass_indices[:, index_remain]

        mass_indices[0] = mass_indices[0].unique(return_inverse=True)[1]
        mass_indices[1] = mass_indices[1].unique(return_inverse=True)[1]

        return mass_indices, mass_values

    # endregion

    # region GC
    def _allocate_RGC(self, size: list[int] | tuple[int], *args, **kwargs):
        """
        Allocate memory for the RGC data structure.

        Args:
        - size: A list of integers representing the size of the RGC tensor.
        - name: (optional) A string representing the name of the RGC tensor.

        Returns:
        None
        """

        index_now = len(self._RGC)

        self._RGC.append(torch.randn(size) * 0)
        self._RGC_remain_index.append(np.zeros(size, dtype=bool))
        self._RGC_size.append(size)
        self._RGC_list_indexStart.append(self._RGC_list_indexStart[-1] + np.prod(size))

        return index_now

    def _GC2RGC(self, GC: torch.Tensor):
        """
        Converts the global control vector (GC) to the reduced global control vector (RGC).

        Args:
            GC (torch.Tensor): The global control vector.

        Returns:
            list: The reduced global control vector (RGC).
        """
        RGC = []
        for i in range(len(self._RGC_remain_index)):
            RGC.append(torch.zeros(self._RGC_size[i]))
            RGC[-1][self._RGC_remain_index[i]] = GC[
                self._GC_list_indexStart[i] : self._GC_list_indexStart[i + 1]
            ]

        for c in self._constraints_forward:
            RGC = c.modify_RGC(RGC)

        for b in self._boundarys_enabled:
            RGC = b.modify_RGC(RGC)

        return RGC

    def _RGC2GC(self, RGC: list[torch.Tensor]):
        GC = torch.cat(
            [RGC[i][self._RGC_remain_index[i]].flatten() for i in range(len(RGC))],
            dim=0,
        )
        return GC

    def refine_RGC(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
        RGC_out = [RGC[i].clone().detach() for i in range(len(RGC))]
        for instance in self._instances.values():
            RGC_out = instance.refine_RGC(RGC_out)
        return RGC_out

    # endregion

    # region Instance Management

    def add_part(self, part: Part, name: str = None) -> None:
        if name is None:
            name = part.__class__.__name__
            number = len(self._parts)
            while ("%s-%d" % (name, number)) in self._parts:
                number += 1
            name = "%s-%d" % (name, number)

        if name in self._parts:
            raise ValueError(f"Part with name {name} already exists in the assembly.")
        self._parts[name] = part

    def get_part(self, name: str) -> Part:
        if name not in self._parts:
            raise ValueError(f"Part with name {name} does not exist in the assembly.")
        return self._parts[name]

    def delete_part(self, name: str) -> None:
        if name in self._parts:
            del self._parts[name]
        else:
            raise ValueError(f"Part with name {name} does not exist in the assembly.")

    def add_instance(self, instance: Instance, name: str | None = None) -> None:
        if name is None:
            name = instance.__class__.__name__
            number = len(self._instances)
            while ("%s-%d" % (name, number)) in self._instances:
                number += 1
            name = "%s-%d" % (name, number)

        if name in self._instances:
            raise ValueError(
                f"Instance with name {name} already exists in the assembly."
            )
        if instance.part_name in self._parts:
            # Keep the Instance exterior-surface property as a direct
            # forwarding view of its Part as soon as the relationship exists.
            instance.part = self._parts[instance.part_name]
        self._instances[name] = instance

    def get_instance(self, name: str) -> Instance:
        if name not in self._instances:
            raise ValueError(
                f"Instance with name {name} does not exist in the assembly."
            )
        return self._instances[name]

    def delete_instance(self, name: str) -> None:
        if name in self._instances:
            del self._instances[name]
        else:
            raise ValueError(
                f"Instance with name {name} does not exist in the assembly."
            )

    def add_reference_point(self, rp: ReferencePoint, name: str | None = None):
        """
        Adds a reference point to the FEA object.

        Parameters:
            node (torch.Tensor): The node to be added as a reference point.

        Returns:
            str: The name of the reference point.
        """

        if name is None:
            number = len(self._reference_points)
            while ("rp-%d" % number) in self._reference_points:
                number += 1
            name = "rp-%d" % number

        self._reference_points[name] = rp

        return name

    def get_reference_point(self, name: str) -> ReferencePoint:
        """
        Retrieves a reference point from the FEA object.

        Parameters:
        - name (str): The name of the reference point to be retrieved.

        Returns:
        - ReferencePoint: The requested reference point.
        """
        if name in self._reference_points:
            return self._reference_points[name]
        else:
            raise ValueError(f"Reference point '{name}' not found in the model.")

    def delete_reference_point(self, name: str):
        """
        Deletes a reference point from the FEA object.

        Parameters:
        - name (str): The name of the reference point to be deleted.

        Returns:
        - None
        """
        if name in self._reference_points:
            del self._reference_points[name]
        else:
            raise ValueError(f"Reference point '{name}' not found in the model.")

    def add_load(self, load: loads.BaseLoad, name: str = None):
        """
        Add a load to the FEA model.

        Parameters:
            load (Load.Force_Base): The load to be added.

        Returns:
            str: The name of the load.
        """
        if name is None:
            name = load.__class__.__name__
            number = len(self._loads)
            while ("%s-%d" % (name, number)) in self._loads:
                number += 1
            name = "%s-%d" % (name, number)
        self._loads[name] = load

        return name

    def add_loads(self, loads_dict: dict[str, loads.BaseLoad]):
        """
        Add multiple loads to the FEA model.

        Parameters:
            loads_dict (dict): A dictionary where keys are load names and values are Load.Force_Base objects.

        Returns:
            None
        """
        for name, load in loads_dict.items():
            self.add_load(load, name)

    def get_load(self, name: str) -> loads.BaseLoad:
        """
        Retrieve a load from the FEA model.

        Parameters:
            name (str): The name of the load to be retrieved.

        Returns:
            Load.Force_Base: The requested load.
        """
        if name in self._loads:
            return self._loads[name]
        else:
            raise ValueError(f"Load '{name}' not found in the model.")

    def delete_load(self, name: str):
        """
        Delete a load from the FEA model.

        Parameters:
            name (str): The name of the load to be deleted.

        Returns:
            None
        """
        if name in self._loads:
            del self._loads[name]
        else:
            raise ValueError(f"Load '{name}' not found in the model.")

    def delete_all_loads(self):
        """
        Delete all loads from the FEA model.

        Returns:
            None
        """
        self._loads.clear()

    def get_object(self, name: str, obj_type: str = "auto"):
        """
        Retrieve an object from the assembly by its name.

        Args:
            name (str): The object name.
            obj_type (str): One of 'auto', 'instance', 'rp', 'load', 'constraint', 'boundary'.
                If 'auto', search all supported object containers and return the unique match.

        Returns:
            object: The requested assembly object.
        """
        containers = {
            "instance": self._instances,
            "rp": self._reference_points,
            "load": self._loads,
            "constraint": self._constraints,
            "boundary": self._boundarys,
        }

        if obj_type == "auto":
            found = []
            for kind, data in containers.items():
                if name in data:
                    found.append((kind, data[name]))
            if len(found) == 0:
                raise ValueError(
                    f"Object '{name}' not found in instance/rp/load/constraint/boundary."
                )
            if len(found) > 1:
                kinds = [item[0] for item in found]
                raise ValueError(
                    f"Object name '{name}' is ambiguous in {kinds}. Please set obj_type explicitly."
                )
            return found[0][1]

        if obj_type not in containers:
            raise ValueError(
                "obj_type must be one of {'auto', 'instance', 'rp', 'load', 'constraint', 'boundary'}"
            )

        if name not in containers[obj_type]:
            raise ValueError(f"{obj_type} '{name}' not found in the model.")

        return containers[obj_type][name]

    def get_work_conditions(self) -> WorkCondition:
        """
        Get parameters about all loads in the FEA model.

        Returns:
            WorkCondition: The work condition object containing the parameters.
        """
        workcondition = WorkCondition()

        for name, load in self._loads.items():
            workcondition.load_info[name] = load._parameters

        for name, ins in self._instances.items():
            workcondition.ins_enabled[name] = ins.enabled
        for name, f in self._loads.items():
            workcondition.load_enabled[name] = f.enabled
        for name, c in self._constraints.items():
            workcondition.constraint_enabled[name] = c.enabled
        for name, b in self._boundarys.items():
            workcondition.boundary_enabled[name] = b.enabled

        return workcondition

    def set_work_conditions(self, workcondition: WorkCondition):
        """
        Set parameters for loads in the FEA model.

        Args:
            workcondition (WorkCondition): The work condition object containing the parameters.

        Returns:
            None
        """
        load_info = workcondition.load_info
        for name, info in load_info.items():
            if name in self._loads:
                self._loads[name]._parameters = info
            else:
                print(
                    f"Warning: Load '{name}' not found in the model. Skipping parameter update."
                )

        for name, enabled in workcondition.ins_enabled.items():
            if name in self._instances:
                self._instances[name].enabled = enabled
            else:
                print(
                    f"Warning: Instance '{name}' not found in the model. Skipping parameter update."
                )

        for name, enabled in workcondition.load_enabled.items():
            if name in self._loads:
                self._loads[name].enabled = enabled
            else:
                print(
                    f"Warning: Load '{name}' not found in the model. Skipping parameter update."
                )

        for name, enabled in workcondition.constraint_enabled.items():
            if name in self._constraints:
                self._constraints[name].enabled = enabled
            else:
                print(
                    f"Warning: Constraint '{name}' not found in the model. Skipping parameter update."
                )

        for name, enabled in workcondition.boundary_enabled.items():
            if name in self._boundarys:
                self._boundarys[name].enabled = enabled
            else:
                print(
                    f"Warning: Boundary '{name}' not found in the model. Skipping parameter update."
                )

        self.define_required_DoFs()

    def add_constraint(self, constraint: constraints.BaseConstraint, name: str = None):
        """
        Add a constraint to the FEA model.

        Parameters:
            constraint (Constraints.Constraints_Base): The constraint to be added.

        Returns:
            str: The name of the constraint.
        """
        if name is None:
            number = len(self._constraints)
            name = constraint.__class__.__name__
            while ("%s-%d" % (name, number)) in self._constraints:
                number += 1
            name = "%s-%d" % (name, number)
        self._constraints[name] = constraint
        return name

    def get_constraint(self, name: str) -> constraints.BaseConstraint:
        """
        Retrieve a constraint from the FEA model.

        Parameters:
            name (str): The name of the constraint to be retrieved.

        Returns:
            Constraints.Constraints_Base: The requested constraint.
        """
        if name in self._constraints:
            return self._constraints[name]
        else:
            raise ValueError(f"Constraint '{name}' not found in the model.")

    def delete_constraint(self, name: str):
        """
        Delete a constraint from the FEA model.

        Parameters:
            name (str): The name of the constraint to be deleted.

        Returns:
            None
        """
        if name in self._constraints:
            del self._constraints[name]
        else:
            raise ValueError(f"Constraint '{name}' not found in the model.")

    def add_boundary(self, boundary: object, name: str = None):
        """
        Add a boundary condition object to the model.

        Parameters:
            boundary: The boundary condition object (from assemble.boundarys).

        Returns:
            str: The name of the boundary.
        """
        if name is None:
            name = boundary.__class__.__name__
            number = len(self._boundarys)
            while (f"{name}-{number}") in self._boundarys:
                number += 1
            name = f"{name}-{number}"
        self._boundarys[name] = boundary
        return name

    def get_boundary(self, name: str):
        if name in self._boundarys:
            return self._boundarys[name]
        else:
            raise ValueError(f"Boundary '{name}' not found in the model.")

    def delete_boundary(self, name: str):
        if name in self._boundarys:
            del self._boundarys[name]
        else:
            raise ValueError(f"Boundary '{name}' not found in the model.")

    # endregion

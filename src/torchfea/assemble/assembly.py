from optparse import Values
import time
import numpy as np
import torch
from . import Instance
from . import ReferencePoint
from . import loads, constraints, boundarys
from .part import _Surfaces, Part
import pyvista as pv

class Assembly:
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


        self.RGC: list[torch.Tensor]
        """
        record the redundant generalized coordinates
        """

        self._RGC_size: list[tuple[int]]
        """Record the size of each RGC component
        """

        self.RGC_remain_index: list[np.ndarray]
        """
        record the remaining index of the RGC\n
        """

        self.RGC_remain_index_flatten: torch.Tensor
        """
        record the remaining index of the RGC (flattened)\n
        """

        # initialize the GC (generalized coordinates)
        self.GC: torch.Tensor
        """
        record the generalized coordinates\n
        """

        self._GC_list_indexStart: list[int] = []
        """
        record the start index of the GC\n
        """
        self.RGC_list_indexStart: list[int] = []
        """Record the start index of the RGC\n
        """

        self.mass_matrix_indices: torch.Tensor
        """The indices of the mass matrix"""

        self.mass_matrix_values: torch.Tensor
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

    def show_ins(self, ins_name: str, GC: torch.Tensor = None):
        """
        Visualize the specified instance.

        Args:
            ins_name (str): The name of the instance to visualize.
            GC (torch.Tensor, optional): The generalized coordinates to use for visualization. Defaults to None.
        """
        if GC is not None:
            RGC = self._GC2RGC(GC)
        else:
            RGC = None
        mesh = self._instances[ins_name].get_mesh(RGC=RGC)
        pv.global_theme.allow_empty_mesh = True
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=True, opacity=1.0, label=ins_name)
        plotter.add_legend()
        plotter.show()

    def show_all(self, GC: torch.Tensor = None):
        """
        Visualize all instances in the assembly.

        Args:
            GC (torch.Tensor, optional): The generalized coordinates to use for visualization. Defaults to None.
        """
        meshes = self.get_meshes(GC=GC)
        pv.global_theme.allow_empty_mesh = True
        plotter = pv.Plotter()
        for ins_name, mesh in meshes.items():
            plotter.add_mesh(mesh, show_edges=True, opacity=1.0, label=ins_name)
        plotter.add_legend()
        plotter.show()
    # endregion

    # region Initialization

    def initialize(self, *args, **kwargs):
        """
        Initialize the finite element model.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.

        Returns:
            None
        """

        # region sort the parts, instances, loads, and constraints
        self._parts = dict(sorted(self._parts.items()))
        self._instances = dict(sorted(self._instances.items()))
        self._loads = dict(sorted(self._loads.items()))
        self._constraints = dict(sorted(self._constraints.items()))
        self._reference_points = dict(sorted(self._reference_points.items()))
        self._boundarys = dict(sorted(self._boundarys.items()))
        # endregion

        # region initialize the RGC

        # initialize the RGC (redundant generalized coordinate)
        self.RGC = []
        self.RGC_remain_index = []
        self.RGC_list_indexStart = [0]
        self._RGC_size = []

        for ins in self._instances.keys():
            RGC_index = self._allocate_RGC(
                size=self._instances[ins]._RGC_requirements)
            self._instances[ins].set_RGC_index(RGC_index)

        for rp in self._reference_points.keys():
            RGC_index = self._allocate_RGC(
                size=self._reference_points[rp]._RGC_requirements)
            self._reference_points[rp].set_RGC_index(RGC_index)
        self.RGC[RGC_index][-1] = 1e-5

        for f in self._loads.keys():
            RGC_index = self._allocate_RGC(
                size=self._loads[f]._RGC_requirements)
            self._loads[f].set_RGC_index(RGC_index)

        for c in self._constraints.keys():
            RGC_index = self._allocate_RGC(
                size=self._constraints[c]._RGC_requirements)
            self._constraints[c].set_RGC_index(RGC_index)

        for b in self._boundarys.keys():
            RGC_index = self._allocate_RGC(
                size=self._boundarys[b]._RGC_requirements)
            self._boundarys[b].set_RGC_index(RGC_index)

        # endregion

        # region initialize the elements, loads, and constraints

        # initialize the parts
        for part in self._parts.values():
            part.initialize(self)

        # initialize the instances
        for ins in self._instances.values():
            ins.initialize(self)

        # initialize the loads
        for l in self._loads.values():
            l.initialize(self)

        # initialize the constraints
        for c in self._constraints.values():
            c.initialize(self)

        # initialize the boundary conditions
        for b in self._boundarys.values():
            b.initialize(self)

        # endregion

        # region modify the RGC_remain_index
        for ins in self._instances.values():
            self.RGC_remain_index = ins.set_required_DoFs(self.RGC_remain_index)

        for f in self._loads.values():
            self.RGC_remain_index = f.set_required_DoFs(self.RGC_remain_index)

        for c in self._constraints.values():
            self.RGC_remain_index = c.set_required_DoFs(self.RGC_remain_index)

        # Finally, apply boundary conditions to deactivate Dirichlet DOFs
        for b in self._boundarys.values():
            self.RGC_remain_index = b.set_required_DoFs(self.RGC_remain_index)

        self.RGC_remain_index_flatten = np.concatenate([
            self.RGC_remain_index[i].reshape(-1)
            for i in range(len(self.RGC_remain_index))
        ]).tolist()
        self.RGC_remain_index_flatten = torch.tensor(
            self.RGC_remain_index_flatten, dtype=torch.bool)

        # GC core
        self.GC = self._RGC2GC(self.RGC)
        self._GC_list_indexStart = np.cumsum([
            self.RGC_remain_index[j].sum()
            for j in range(len(self.RGC_remain_index))
        ]).tolist()
        self._GC_list_indexStart.insert(0, 0)

        # endregion

    def initialize_dynamic(self):
            
        for ins in self._instances.values():
            ins.initialize_dynamic()

        for l in self._loads.values():
            l.initialize_dynamic()

        for c in self._constraints.values():
            c.initialize_dynamic()

        # assemble the redundant mass matrix
        mass_indices = []
        mass_values = []
        for ins in self._instances.values():
            indices_now, values_now = ins.get_mass_matrix()
            mass_indices.append(indices_now)
            mass_values.append(values_now)
        self.mass_matrix_indices = torch.cat(mass_indices, dim=1)
        self.mass_matrix_values = torch.cat(mass_values, dim=0)

    def reinitialize(self, RGC: list[torch.Tensor]):
        """
        Reinitializes the finite element analysis problem.

        Args:
            RGC (list[torch.Tensor]): The redundant generalized coordinates.
        """
        self.RGC = RGC
        self.GC = self._RGC2GC(self.RGC)

        for ins in self._instances.values():
            ins.reinitialize(RGC)

        for l in self._loads.values():
            l.reinitialize(RGC)

        for c in self._constraints.values():
            c.reinitialize(RGC)
    # endregion

    # region Stiffness Matrix Assembly

    def assemble_force(self, RGC: list[torch.Tensor] = None, GC: torch.Tensor = None) -> torch.Tensor:
        
        if RGC is None:
            if GC is None:
                raise ValueError("Either RGC or GC must be provided.")
            RGC = self._GC2RGC(GC)

        #region evaluate the structural K and R
        t0 = time.time()
        R_values = []
        R_indices = []

        for ins in self._instances.keys():
            Ra_indice, Ra_values = self._instances[ins].structural_stiffness(
                RGC=RGC, if_onlyforce=True)
            R_values.append(Ra_values)
            R_indices.append(Ra_indice)
        t1 = time.time()

        ff = []
        for f in self._loads.values():
            Rf_indice, Rf_values = f.get_stiffness(
                RGC=RGC, if_onlyforce=True)
            R_values.append(-Rf_values)
            R_indices.append(Rf_indice)

            ff.append(torch.zeros(self.RGC_list_indexStart[-1]).scatter_add_(0, Rf_indice.to(torch.int64), Rf_values))
        t2 = time.time()
        # endregion

        R_indices = torch.cat(R_indices, dim=0)
        R_values = torch.cat(R_values, dim=0)

        R0 = torch.zeros(self.RGC_list_indexStart[-1])
        # Convert R_indices to int64 explicitly for scatter operation
        R0.scatter_add_(0, R_indices.to(torch.int64), R_values)
        t0 = time.time()
        R = R0
        #region consider the constraints
        for c in self._constraints.values():
            R_new = c.modify_R_K(
                RGC, R0, if_onlyforce=True)
            R = R + R_new
        t4 = time.time()
        #endregion

        # get the global stiffness matrix and force vector

        R = R[self.RGC_remain_index_flatten]

        t6 = time.time()
        return R
    
    def assemble_Stiffness_Matrix(self,
                                   RGC: list[torch.Tensor] = None, GC: torch.Tensor = None):
        """
        Assemble the stiffness matrix.

        Args:
            RGC (list[torch.Tensor]): The redundant generalized coordinates.
            GC (torch.Tensor, optional): The generalized coordinates. If provided, it will be converted to RGC internally. Defaults to None.

        Returns:
            tuple: A tuple containing the right-hand side vector, the indices of the stiffness matrix, and the values of the stiffness matrix.
                -
        """

        if RGC is None:
            if GC is None:
                raise ValueError("Either RGC or GC must be provided.")
            RGC = self._GC2RGC(GC)

        #region evaluate the structural K and R
        R0, K_indices, K_values = self._assemble_generalized_Matrix(
            RGC)
        # endregion
        R, K_indices, K_values = self._assemble_reduced_Matrix(
            RGC, R0, K_indices, K_values)

        return R, K_indices, K_values

    def _assemble_generalized_Matrix(self,
                                     RGC: list[torch.Tensor] = None, GC: torch.Tensor = None):
        if RGC is None:
            if GC is None:
                raise ValueError("Either RGC or GC must be provided.")
            RGC = self._GC2RGC(GC)

        #region evaluate the structural K and R
        t0 = time.time()
        K_values = []
        K_indices = []
        R_values = []
        R_indices = []

        for ins in self._instances.keys():
            Ra_indice, Ra_values, Ka_indice, Ka_value = self._instances[ins].structural_stiffness(
                RGC=RGC)
            K_values.append(Ka_value)
            K_indices.append(Ka_indice)
            R_values.append(Ra_values)
            R_indices.append(Ra_indice)
        t1 = time.time()

        ff = []
        for f in self._loads.values():
            Rf_indice, Rf_values, Kf_indice, Kf_value = f.get_stiffness(
                RGC=RGC)
            K_values.append(-Kf_value)
            K_indices.append(Kf_indice)
            R_values.append(-Rf_values)
            R_indices.append(Rf_indice)

            ff.append(torch.zeros(self.RGC_list_indexStart[-1]).scatter_add_(0, Rf_indice.to(torch.int64), Rf_values))
        t2 = time.time()
        # endregion

        K_indices = torch.cat(K_indices, dim=1)
        K_values = torch.cat(K_values, dim=0)
        R_indices = torch.cat(R_indices, dim=0)
        R_values = torch.cat(R_values, dim=0)

        R0 = torch.zeros(self.RGC_list_indexStart[-1])
        # Convert R_indices to int64 explicitly for scatter operation
        R0.scatter_add_(0, R_indices.to(torch.int64), R_values)
        return R0, K_indices, K_values

    def _assemble_reduced_Matrix(self, RGC: list[torch.Tensor],
                                 R0: torch.Tensor, K_indices: torch.Tensor,
                                 K_values: torch.Tensor):
        t0 = time.time()
        R = R0
        #region consider the constraints
        for c in self._constraints.values():
            R_new, Kc_indices, Kc_values = c.modify_R_K(
                RGC, R0, K_indices, K_values)
            K_indices = torch.cat([K_indices, Kc_indices], dim=1)
            K_values = torch.cat([K_values, Kc_values])
            R = R + R_new
        t4 = time.time()
        #endregion

        # get the global stiffness matrix and force vector
        index_remain = self.RGC_remain_index_flatten[K_indices[0].cpu(
        )] & self.RGC_remain_index_flatten[K_indices[1].cpu()]
        K_values = K_values[index_remain]
        K_indices = K_indices[:, index_remain]
        t44 = time.time()

        K_indices[0] = K_indices[0].unique(return_inverse=True)[1]
        K_indices[1] = K_indices[1].unique(return_inverse=True)[1]

        t5 = time.time()

        R = R[self.RGC_remain_index_flatten]

        t6 = time.time()
        return R, K_indices, K_values

    def _total_Potential_Energy(self,
                                RGC: list[torch.Tensor] = None, GC: torch.Tensor = None) -> float:
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
        for ins in self._instances.values():
            energy = energy + ins.potential_energy(RGC=RGC)

        # force potential
        for f in self._loads.values():
            energy = energy - f.get_potential_energy(RGC=RGC)

        return energy
    
    # endregion

    # region for Dynamic Mass Matrix

    def assemble_mass_matrix(self, GC_now: torch.Tensor):
        mass_indices = [self.mass_matrix_indices]
        mass_values = [self.mass_matrix_values]
        RGC = self._GC2RGC(GC_now)
        for c in self._constraints.values():
            indices_now, values_now = c.modify_mass_matrix(mass_indices=self.mass_matrix_indices, mass_values=self.mass_matrix_values, RGC=RGC)
            mass_indices.append(indices_now)
            mass_values.append(values_now)

        mass_indices = torch.cat(mass_indices, dim=1)
        mass_values = torch.cat(mass_values, dim=0)

        # get the global stiffness matrix and force vector
        index_remain = self.RGC_remain_index_flatten[mass_indices[0].cpu(
        )] & self.RGC_remain_index_flatten[mass_indices[1].cpu()]
        mass_values = mass_values[index_remain]
        mass_indices = mass_indices[:, index_remain]
        t44 = time.time()

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

        index_now = len(self.RGC)

        self.RGC.append(torch.randn(size) * 0)
        self.RGC_remain_index.append(np.zeros(size, dtype=bool))
        self._RGC_size.append(size)
        self.RGC_list_indexStart.append(
            self.RGC_list_indexStart[-1] + np.prod(size))

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
        for i in range(len(self.RGC_remain_index)):
            RGC.append(torch.zeros(self._RGC_size[i]))
            RGC[-1][self.RGC_remain_index[i]] = GC[
                self._GC_list_indexStart[i]:self._GC_list_indexStart[i + 1]]

        for c in self._constraints.values():
            RGC = c.modify_RGC(RGC)

        for b in self._boundarys.values():
            RGC = b.modify_RGC(RGC)

        return RGC

    def _RGC2GC(self, RGC: list[torch.Tensor]):
        GC = torch.cat([
            RGC[i][self.RGC_remain_index[i]].flatten() for i in range(len(RGC))
        ],
                       dim=0)
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
            while ('%s-%d' % (name, number)) in self._parts:
                number += 1
            name = '%s-%d' % (name, number)

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

    def add_instance(self, instance: Instance, name: str = None) -> None:
        if name is None:
            name = instance.__class__.__name__
            number = len(self._instances)
            while ('%s-%d' % (name, number)) in self._instances:
                number += 1
            name = '%s-%d' % (name, number)

        if name in self._instances:
            raise ValueError(f"Instance with name {name} already exists in the assembly.")
        self._instances[name] = instance
        
    def get_instance(self, name: str) -> Instance:
        if name not in self._instances:
            raise ValueError(f"Instance with name {name} does not exist in the assembly.")
        return self._instances[name]

    def delete_instance(self, name: str) -> None:
        if name in self._instances:
            del self._instances[name]
        else:
            raise ValueError(f"Instance with name {name} does not exist in the assembly.")

    def add_reference_point(self, rp: ReferencePoint, name: str = None):
        """
        Adds a reference point to the FEA object.

        Parameters:
            node (torch.Tensor): The node to be added as a reference point.

        Returns:
            str: The name of the reference point.
        """

        if name is None:
            number = len(self._reference_points)
            while ('rp-%d' % number) in self._reference_points:
                number += 1
            name = 'rp-%d' % number

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
            raise ValueError(
                f"Reference point '{name}' not found in the model.")

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
            raise ValueError(
                f"Reference point '{name}' not found in the model.")

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
            while ('%s-%d' % (name, number)) in self._loads:
                number += 1
            name = '%s-%d' % (name, number)
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

    def get_load_parameters(self) -> dict[str, torch.Tensor]:
        """
        Get parameters about all loads in the FEA model.

        Returns:
            dict: A dictionary where keys are load names and values are numpy arrays containing load parameters.
        """
        load_info = {}
        for name, load in self._loads.items():
            load_info[name] = load._parameters
        return load_info
    
    def set_load_parameters(self, load_info: dict[str, torch.Tensor]):
        """
        Set parameters for loads in the FEA model.

        Args:
            load_info (dict): A dictionary where keys are load names and values are torch tensors containing load parameters.

        Returns:
            None
        """
        for name, info in load_info.items():
            if name in self._loads:
                self._loads[name]._parameters = info
            else:
                raise ValueError(f"Load '{name}' not found in the model.")

    def add_constraint(self,
                       constraint: constraints.BaseConstraint,
                       name: str = None):
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
            while ('%s-%d' % (name, number)) in self._constraints:
                number += 1
            name = '%s-%d' % (name, number)
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

    # region Boundary Management

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

    # endregion
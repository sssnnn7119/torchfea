from __future__ import annotations
import re
from typing import Optional, TYPE_CHECKING

from .elements import Element_3D

if TYPE_CHECKING:
    from .. import Assembly

import numpy as np
import torch
from . import elements
from .obj_base import BaseObj
from .elements import BaseSurface, BaseElement, surfaces
from ..interfaces import Serializable
import pyvista as pv

class _Surfaces(Serializable):
    """
    Class representing a set of surfaces in the finite element model.
    """

    _serialized_attributes_exclude = ['_surface_elements']

    def __init__(self):
        self._surface_dict: dict[str, list[tuple[np.ndarray, int]]] = {}
        self._surface_elements: dict[str, list[BaseSurface]] = {}
        self._initialized = False

    def initialize(self, part: Part):
        """
        Initialize the surface set before FEA.
        """
        self._surface_elements = {}
        for name, surface_indices in self._surface_dict.items():
            element_now = part.extract_surfaces(name)
            self._surface_elements[name] = element_now

            # initialize the surface elements
            for se in element_now:
                se.initialize(part)
                self._initialized = True

    def get_elements(self, name: str) -> list[BaseSurface]:
        """
        Get the surface elements by their name.

        Args:
            name (str): The name of the surface.

        Returns:
            list[BaseSurface]: The list of surface elements.
        """
        return self._surface_elements.get(name, [])
    
    def get_trimesh(self, name: str) -> torch.Tensor:
        """
        Get the triangles of a surface set by name.

        Args:
            name (str): Name of the surface set.

        Returns:
            torch.Tensor: Triangles of the surface set.
        """
        surface_elements = self.get_elements(name)
        if len(surface_elements) == 0:
            raise ValueError(f"Surface set '{name}' not found in the model.")
        
        triangles = []
        for se in surface_elements:
            triangles.append(se.trimesh)
        return torch.cat(triangles, dim=0)

    def __getitem__(self, key: str):
        """
        Get a surface by its name.

        Args:
            key (str): The name of the surface.

        Returns:
            list[tuple[np.ndarray, int]]: The surface data.
        """
        return self._surface_dict[key]

    def __setitem__(self, key: str, value: list[tuple[np.ndarray, int]]):
        """
        Set a surface by its name.

        Args:
            key (str): The name of the surface.
            value (list[tuple[np.ndarray, int]]): The surface data.
        """
        self._surface_dict[key] = value



    def __contains__(self, key: str):
        """
        Check if a surface exists by its name.

        Args:
            key (str): The name of the surface.

        Returns:
            bool: True if the surface exists, False otherwise.
        """
        return key in self._surface_dict

    def keys(self):
        """
        Get the keys of the surface set.

        Returns:
            list[str]: The list of surface names.
        """
        return list(self._surface_dict.keys())


class Part(Serializable):



    def __init__(self, nodes: torch.Tensor) -> None:

        self.nodes: torch.Tensor = nodes
        """
        Nodes of the part.
        Shape: (num_nodes, 3)
        """
        self.elems: dict[str, Element_3D] = {}
        """
        Elements of the part.
        """

        self.set_nodes: dict[str, np.ndarray] = {}
        """Set of nodes for the part."""

        self.set_elements: dict[str, np.ndarray] = {}
        """Named element sets for selection, export, and CAE preprocessing."""

        self.surfaces = _Surfaces()
        """Surfaces of the part."""

        self.exterior_surface: str = "extern"
        """Name of the immutable all-boundary surface set for this Part."""

        self.mass_matrix_indices: torch.Tensor = None
        """
        Indices of the mass matrix.
        """
        self.mass_matrix_values: torch.Tensor = None
        """
        Values of the mass matrix.
        """

        self.mid_pt_idxmap: dict[tuple[int, int], int] = {}
        """A mapping from edge (node index pair) to midpoint node index, used for quadratic element conversion."""

        self.mid_pt_idxmap_torch: torch.Tensor = None
        """Same mapping as mid_pt_idxmap but in torch.Tensor format for efficient lookup during FEA calculations.
        
        [0]: node index 0 of the edge
        [1]: node index 1 of the edge
        [2]: midpoint node index corresponding to the edge
        """

    def initialize(self):
        for e in self.elems.values():
            e.initialize(self.nodes)
        self.surfaces.initialize(self)

    # region CAD
    def add_element(self, element: BaseElement, name: str = None):
        """
        Add an element to the FEA model.

        Parameters:
            element (elements.Element_Base): The element to be added.

        Returns:
            str: The name of the element.
        """
        if name is None:
            number = len(self.elems)
            while ('element-%d' % number) in self.elems:
                number += 1
            name = 'element-%d' % number
        self.elems[name] = element
        return name

    def delete_element(self, name: str):
        """
        Delete an element from the FEA model.

        Parameters:
            name (str): The name of the element to be deleted.

        Returns:
            None
        """
        if name in self.elems:
            del self.elems[name]
        else:
            raise ValueError(f"Element '{name}' not found in the model.")

    def add_node_set(self, name: str, node_indices: np.ndarray):
        """
        Add a node set to the part.

        Args:
            name (str): Name of the node set.
            node_indices (np.ndarray): Array of node indices to be included in the set.

        Returns:
            str: The name of the added node set.
        """
        if type(node_indices) is torch.Tensor:
            node_indices = node_indices.cpu().numpy()
        self.set_nodes[name] = node_indices
        return name
    
    def delete_node_set(self, name: str):
        """
        Delete a node set from the part.

        Args:
            name (str): Name of the node set to be deleted.
        """
        if name in self.set_nodes:
            del self.set_nodes[name]
        else:
            raise ValueError(f"Node set '{name}' not found in the model.")

    def add_element_set(self, name: str, element_indices: np.ndarray):
        """Register a named set of zero-based element indices.

        Element sets are deliberately lightweight: element connectivity remains
        owned by ``self.elems`` while the set provides a stable CAE selection
        for material assignment workflows and Abaqus-style export.
        """
        if isinstance(element_indices, torch.Tensor):
            element_indices = element_indices.detach().cpu().numpy()
        self.set_elements[name] = np.asarray(element_indices, dtype=int).ravel()
        return name

    def delete_element_set(self, name: str):
        """Delete a named element set."""
        if name not in self.set_elements:
            raise ValueError(f"Element set '{name}' not found in the model.")
        del self.set_elements[name]
        

    def add_surface_set(self, name: str, elements: list[tuple[np.ndarray, int]]):
        """
        Add a surface set to the FEA model.
        
        Args:
            name (str): Name of the surface set.
            elements (list[tuple[np.ndarray, int]]): A list of tuples, where each tuple contains an array of element indices and a surface index.
            
        Returns:
            str: Name of the added surface set.
        """
        if name == getattr(self, "exterior_surface", "extern") and name in self.surfaces:
            raise ValueError(f"Part exterior surface '{name}' cannot be overwritten.")
        self.surfaces[name] = elements
        self.surfaces._initialized = False  # Mark surfaces as not initialized since we added new surface data
        return name
    
    def delete_surface_set(self, name: str):
        """
        Delete a surface set from the FEA model.
        
        Args:
            name (str): Name of the surface set to delete.
            
        Raises:
            KeyError: If the surface set doesn't exist.
        """
        if name == getattr(self, "exterior_surface", "extern"):
            raise ValueError(f"Part exterior surface '{name}' cannot be deleted.")
        if name in self.surfaces:
            del self.surfaces[name]
        else:
            raise KeyError(f"Surface set '{name}' not found in the model.")
        
    def refine_RGC(self, RGC: torch.Tensor) -> torch.Tensor:
        RGC_out = RGC
        for e in self.elems.values():
            RGC_out = e.refine_RGC(RGC_out, self.nodes)
        return RGC_out

    def merge_elements(self, element_name_list: list[str], element_name_new: str) -> None:
        """
        Merge multiple elements into a single new element.
        
        Args:
            element_name_list (list[str]): List of element names to merge.
            element_name_new (str): Name for the new merged element.
            
        Returns:
            str: Name of the merged element.
            
        Raises:
            ValueError: If elements are of different types or if any element name is not found.
        """
        if len(element_name_list) < 2:
            elems0 = self.elems[element_name_list[0]]
            self.add_element(elems0, name=element_name_new)
            self.delete_element(element_name_list[0])
            return
            
        # Check if all elements exist
        for name in element_name_list:
            if name not in self.elems:
                raise ValueError(f"Element '{name}' not found in the model")
            
        # Check if all elements are of the same type
        element_type = self.elems[element_name_list[0]].__class__.__name__
        for name in element_name_list[1:]:
            if self.elems[name].__class__.__name__ != element_type:
                raise ValueError(f"Cannot merge elements of different types: {type(self.elems[name])} and {element_type}")
        
        # Create a new element of the same type
        merged_elems = []
        merged_index = []
        for name in element_name_list:
            merged_elems.append(self.elems[name]._elems)
            merged_index.append(self.elems[name]._elems_index)
        merged_elems = torch.cat(merged_elems, dim=0)
        merged_index = torch.cat(merged_index, dim=0)
        merged_element = elements.initialize_element(element_type=element_type, elems_index=merged_index, elems=merged_elems, nodes=self.nodes)
        
        
        # Add merged element to the model
        self.add_element(merged_element, name=element_name_new)
        
        # Clean up the original elements if needed
        for name in element_name_list:
            self.delete_element(name)
            
        return

    def convert_linear_to_quadratic_elements(self, element_name_list: list[str], new_element_name_list: list[str], if_update_setnodes = True):
        """
        Convert selected linear elements into quadratic elements.

        Args:
            element_name_list (list[str]): Names of the existing linear elements.
            new_element_name_list (list[str]): Names for the new quadratic elements.
            

        This method replaces each old element in-place by a new quadratic element:
        1. Extract all unique edges from the selected linear elements.
        2. Create midpoint nodes for those edges and append them to part nodes.
        3. Rebuild element connectivity according to the quadratic element node order.
        4. Preserve density and material definitions.
        """
        if len(element_name_list) != len(new_element_name_list):
            raise ValueError('element_name_list and new_element_name_list must have the same length.')

        if len(set(new_element_name_list)) != len(new_element_name_list):
            raise ValueError('new_element_name_list contains duplicate names.')

        supported_conversion = {
            'C3D4': 'C3D10',
            'C3D6': 'C3D15',
            'C3D8': 'C3D20',
            'C3D8R': 'C3D20',
        }

        selected: list[tuple[str, str, BaseElement]] = []
        reserved_new_names = set(self.elems.keys())

        for old_name, new_name in zip(element_name_list, new_element_name_list):
            if old_name not in self.elems:
                raise ValueError(f"Element '{old_name}' not found in the model.")
            if new_name in reserved_new_names and new_name != old_name:
                raise ValueError(f"New element name '{new_name}' already exists in the model.")

            element = self.elems[old_name]
            old_type = element.__class__.__name__
            if old_type not in supported_conversion:
                raise ValueError(f"Unsupported linear element type '{old_type}'.")

            expected_new_type = supported_conversion[old_type]
            if new_name != expected_new_type and new_name != old_name:
                # We only verify the class type, not the text of the name,
                # because user may pass an arbitrary new element name.
                pass

            selected.append((old_name, new_name, element))

        edge_to_mid_index: dict[tuple[int, int], int] = {}
        node_count = self.nodes.shape[0]

        # extract all edges
        edges_all = []
        for _, _, element in selected:
            linear_edges_idx = self._get_linear_edges_for_element_type(element.__class__.__name__)
            elems = element._elems
            linear_edges = elems[:, linear_edges_idx].cpu().numpy()
            edges_all.append(linear_edges.reshape(-1, 2))
        edges_all = np.concatenate(edges_all, axis=0)
        edges_all = np.sort(edges_all, axis=1)  # sort node indices in each edge to avoid duplicates like (1,2) and (2,1)
        large_number = edges_all.max() + 1
        edge_hash = edges_all[:, 0] * large_number + edges_all[:, 1]
        _, unique_indices = np.unique(edge_hash, return_index=True)
        unique_edges = edges_all[unique_indices]

        # create the mapping from edge to midpoint node index
        for i, (n1, n2) in enumerate(unique_edges):
            edge_to_mid_index[(n1.item(), n2.item())] = node_count + i
            edge_to_mid_index[(n2.item(), n1.item())] = node_count + i  # add both directions for easy lookup

        # create mid nodes for unique edges
        new_node_tensor = self.nodes[unique_edges].mean(dim=1)
        self.nodes = torch.cat([self.nodes, new_node_tensor], dim=0)

        for old_name, new_name, element in selected:
            old_type = element.__class__.__name__
            new_type = supported_conversion[old_type]
            new_connectivity = self._build_quadratic_connectivity(old_type, element._elems, edge_to_mid_index)

            new_element = elements.initialize_element(
                element_type=new_type,
                elems_index=element._elems_index,
                elems=new_connectivity,
                part=self,
            )
            new_element.density = element.density
            if isinstance(element.materials, dict):
                new_element.materials = dict(element.materials)
            else:
                new_element.materials = element.materials
            if hasattr(element, "material_element_indices"):
                new_element.material_element_indices = {
                    name: np.asarray(indices, dtype=int).copy()
                    for name, indices in element.material_element_indices.items()
                }

            if new_name == old_name:
                self.delete_element(old_name)
                self.add_element(new_element, name=new_name)
            else:
                self.delete_element(old_name)
                self.add_element(new_element, name=new_name)

        self.mid_pt_idxmap = edge_to_mid_index
        self.mid_pt_idxmap_torch = torch.tensor([[n1, n2, mid_idx] for (n1, n2), mid_idx in edge_to_mid_index.items()], dtype=torch.long)

        if if_update_setnodes:
            for key in self.set_nodes.keys():
                set_nodes = self.set_nodes[key]
                new_set_nodes = set(set_nodes.tolist())
                for n1, n2 in unique_edges:
                    if n1 in new_set_nodes and n2 in new_set_nodes:
                        mid_idx = edge_to_mid_index[(n1, n2)]
                        new_set_nodes.add(mid_idx)
                self.set_nodes[key] = np.sort(np.array(list(new_set_nodes)))

    def _get_linear_edges_for_element_type(self, element_type: str) -> list[list[int]]:
        if element_type == 'C3D4':
            return [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]]
        if element_type == 'C3D6':
            return [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5], [0, 3], [1, 4], [2, 5]]
        if element_type in ['C3D8', 'C3D8R']:
            return [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        raise ValueError(f"Unsupported element type '{element_type}' for edge extraction.")

    def _build_quadratic_connectivity(self, element_type: str, elems: torch.Tensor, edge_to_mid_index: dict[tuple[int, int], int]) -> torch.Tensor:
        def mid(i: int, j: int) -> torch.Tensor:
            edge0 = elems[:, i].cpu().numpy()
            edge1 = elems[:, j].cpu().numpy()
            mid_idx = torch.tensor([edge_to_mid_index[(n1, n2)] for n1, n2 in zip(edge0, edge1)], dtype=torch.long, device=elems.device)
            return mid_idx

        if element_type == 'C3D4':
            return torch.stack([
                elems[:, 0], elems[:, 1], elems[:, 2], elems[:, 3],
                mid(0, 1), mid(1, 2), mid(0, 2), mid(0, 3), mid(1, 3), mid(2, 3),
            ], dim=1)

        if element_type == 'C3D6':
            return torch.stack([
                elems[:, 0], elems[:, 1], elems[:, 2], elems[:, 3], elems[:, 4], elems[:, 5],
                mid(0, 1), mid(1, 2), mid(0, 2), mid(3, 4), mid(4, 5), mid(3, 5),
                mid(0, 3), mid(1, 4), mid(2, 5),
            ], dim=1)

        if element_type in ['C3D8', 'C3D8R']:
            return torch.stack([
                elems[:, 0], elems[:, 1], elems[:, 2], elems[:, 3],
                elems[:, 4], elems[:, 5], elems[:, 6], elems[:, 7],
                mid(0, 1), mid(1, 2), mid(2, 3), mid(3, 0),
                mid(4, 5), mid(5, 6), mid(6, 7), mid(7, 4),
                mid(0, 4), mid(1, 5), mid(2, 6), mid(3, 7),
            ], dim=1)

        raise ValueError(f"Unsupported element type '{element_type}' for quadratic connectivity.")

    def extract_surfaces(self, name: str) -> list[BaseSurface]:
        """        Get the triangles of a surface set by name.  
        
        Args:
            name (str): Name of the surface set.
            
        Returns:
            list[BaseSurface]: List of triangles in the surface set.
            
        Raises:
            ValueError: If the surface set is not found.
        """
        surface = []
        for surf_index in self.surfaces[name]:
            elem_ind = surf_index[0]
            surf_ind = surf_index[1]
            for e in self.elems.values():
                s_now = e.extract_surface(surf_ind, elem_ind)
                surface += s_now
        if len(surface) == 0:
            raise ValueError(f"Surface {surf_ind} not found in the model.")
        else:
            return surfaces.merge_surfaces(surface)

    def export_inp(self, file_path: str, part_name: str = 'Part-1') -> None:
        """
        Export the part to an Abaqus INP file.

        The generated INP includes:
        - node coordinates
        - element connectivity for all elements in the part
        - node sets from self.set_nodes
        - element sets from self.set_elements
        - surfaces from self.surfaces
        """
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write('** Generated by torchfea Part.export_inp\n')
            f.write(f'*Part, name={part_name}\n')

            # Write nodes
            f.write('*Node\n')
            for node_id, node in enumerate(self.nodes, start=1):
                coords = node.tolist()
                f.write(f'{node_id}, {coords[0]}, {coords[1]}, {coords[2]}\n')

            # Write elements grouped by Abaqus element type
            elements_by_type: dict[str, list[BaseElement]] = {}
            for element in self.elems.values():
                elem_type = element.__class__.__name__
                elements_by_type.setdefault(elem_type, []).append(element)

            for elem_type, element_list in elements_by_type.items():
                f.write(f'*Element, type={elem_type}\n')
                for element in element_list:
                    connectivity = element._elems
                    element_ids = element._elems_index
                    for elem_row, elem_id in zip(connectivity, element_ids):
                        node_ids = [str(int(node_id.item()) + 1) for node_id in elem_row]
                        f.write(f'{int(elem_id.item()) + 1}, ' + ', '.join(node_ids) + '\n')

            # Write node sets
            for set_name, node_list in self.set_nodes.items():
                sorted_nodes = sorted(np.asarray(node_list, dtype=int).tolist())
                if len(sorted_nodes) == 0:
                    continue
                f.write(f'*Nset, nset={set_name}\n')
                for i in range(0, len(sorted_nodes), 10):
                    line_nodes = sorted_nodes[i:i + 10]
                    f.write(', '.join(str(node_id + 1) for node_id in line_nodes) + '\n')

            # Write surfaces
            for set_name, element_list in self.set_elements.items():
                sorted_elements = sorted(np.asarray(element_list, dtype=int).tolist())
                if not sorted_elements:
                    continue
                f.write(f'*Elset, elset={set_name}\n')
                for i in range(0, len(sorted_elements), 10):
                    f.write(', '.join(str(element_id + 1)
                                      for element_id in sorted_elements[i:i + 10]) + '\n')

            for surface_name in self.surfaces.keys():
                surface_items = self.surfaces[surface_name]
                if len(surface_items) == 0:
                    continue
                f.write(f'*Surface, type=ELEMENT, name={surface_name}\n')
                for elem_ids, surface_index in surface_items:
                    if isinstance(elem_ids, np.ndarray):
                        elem_ids_iter = elem_ids.ravel().tolist()
                    elif isinstance(elem_ids, (list, tuple)):
                        elem_ids_iter = list(elem_ids)
                    else:
                        elem_ids_iter = [int(elem_ids)]
                    for elem_id in elem_ids_iter:
                        f.write(f'{int(elem_id) + 1}, S{int(surface_index) + 1}\n')

            f.write('*End Part\n')

    # endregion

    # region FEA

    def potential_energy(self, RGC: torch.Tensor, rotation_matrix: torch.Tensor = None) -> torch.Tensor:
        p = torch.tensor(0.0)
        for e in self.elems.values():
            p = p + e.potential_Energy(RGC, rotation_matrix=rotation_matrix)
        return p

    def structural_stiffness(self, RGC: torch.Tensor, rotation_matrix: torch.Tensor = None) -> list[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        K_values = []
        K_indices = []
        R_values = []
        R_indices = []
        for e in self.elems.values():
            Ra_indice, Ra_values, Ka_indice, Ka_value = e.structural_Force(
                RGC=RGC, rotation_matrix=rotation_matrix)
            K_values.append(Ka_value)
            K_indices.append(Ka_indice)
            R_values.append(Ra_values)
            R_indices.append(Ra_indice)

        K_indices = torch.cat(K_indices, dim=1)
        K_values = torch.cat(K_values, dim=0)
        R_indices = torch.cat(R_indices, dim=0)
        R_values = torch.cat(R_values, dim=0)
        return R_indices, R_values, K_indices, K_values
    
    def structural_force(self, RGC: torch.Tensor, rotation_matrix: torch.Tensor = None) -> list[torch.Tensor, torch.Tensor]:
        
        R_values = []
        R_indices = []
        for e in self.elems.values():
            Ra_indice, Ra_values = e.structural_Force(
                RGC=RGC, rotation_matrix=rotation_matrix, if_onlyforce=True)
            R_values.append(Ra_values)
            R_indices.append(Ra_indice)
        R_indices = torch.cat(R_indices, dim=0)
        R_values = torch.cat(R_values, dim=0)
        return R_indices, R_values

    # endregion

    # region dynamic

    def get_mass_matrix(self, rotation_matrix: torch.Tensor = None):
        M_indices = []
        M_values = []
        for e in self.elems.values():
            Me_indice, Me_value = e.get_mass_matrix(rotation_matrix=rotation_matrix)
            M_indices.append(Me_indice)
            M_values.append(Me_value)

        M_indices = torch.cat(M_indices, dim=1)
        M_values = torch.cat(M_values, dim=0)

        return M_indices, M_values

    # endregion

    def get_mesh(self, surf_name: str = None) -> pv.PolyData:
        """
        Get the Part mesh for visualization.
        Args:
            surf_name (str, optional): Name of the surface set to visualize. 
                If None, use the Part exterior surface. Defaults to None.
        Returns:
            pv.PolyData: The mesh of the Part.
        """
        import pyvista as pv
        
        if surf_name is None:
            surf_name = getattr(self, "exterior_surface", "extern")
            triangles = self.surfaces.get_trimesh(surf_name).cpu().numpy()
        else:
            triangles = self.surfaces.get_trimesh(surf_name).cpu().numpy()
        nodes_transformed = self.nodes.cpu().numpy()

        mesh = pv.PolyData(nodes_transformed, np.hstack([ np.full((triangles.shape[0], 1), 3), triangles ]))
        return mesh


class Instance(BaseObj):

    _serialized_attributes_exclude = ['part', ]
    def __init__(self, part_name: str, translation: list[float] = None, rotation: list[float] = None) -> None:
        """
        Create an instance of a part.

        Parameters:
            part_name: Name of the Part owned by the Assembly.
            translation: Optional instance translation vector.
            rotation: Optional instance rotation vector.
        """

        super().__init__()

        self.part_name = part_name

        self.part: Part = None

        if translation is not None:
            self._translation = torch.tensor(translation)
        else:
            self._translation: torch.Tensor = torch.zeros(3)
            """the translation vector of the instance"""

        if rotation is not None:
            self._rotation = torch.tensor(rotation)
        else:
            self._rotation: torch.Tensor = torch.randn(3) * 0.0
            """the rotation vector of the instance defined in exponential map"""

        theta = torch.norm(self._rotation)
        if theta == 0:
            self.rotation_matrix = None
        else:
            r = self._rotation / theta
            r = r.view(3, 1)
            R = torch.cos(theta) * torch.eye(3) + (1 - torch.cos(theta)) * (r @ r.t()) + torch.sin(theta) * torch.tensor([[0, -r[2, 0], r[1, 0]], [r[2, 0], 0, -r[0, 0]], [-r[1, 0], r[0, 0], 0]])
            self.rotation_matrix = R

    @property
    def exterior_surface(self) -> str:
        """Forward the Part exterior surface name for instance-level access."""
        if self.part is None:
            raise RuntimeError(
                f"Instance '{self.part_name}' is not linked to a Part yet; "
                "set the exterior surface on the Part before initialization.")
        return self.part.exterior_surface

    @exterior_surface.setter
    def exterior_surface(self, name: str) -> None:
        """Forward an instance assignment to its owning Part."""
        if self.part is None:
            raise RuntimeError(
                f"Instance '{self.part_name}' is not linked to a Part yet; "
                "set the exterior surface on the Part before initialization.")
        self.part.exterior_surface = str(name).strip()


    @property
    def elems(self) -> dict[str, BaseElement]:
        return self.part.elems
    
    @property
    def nodes(self) -> torch.Tensor:
        return self._transform(self.part.nodes, self._rotation) + self._translation.unsqueeze(0)
    
    @property
    def surfaces(self) -> _Surfaces:
        return self.part.surfaces

    @property
    def set_nodes(self) -> dict[str, np.ndarray]:
        return self.part.set_nodes

    @staticmethod
    def _transform(vector0: torch.Tensor, rotation_vector: torch.Tensor = None):
        """
        Rotate a 3D vector by a rotation vector
        :param rotation_vector: rotation vector (3,)
        :param vector0: 3D vector (n, 3)
        :return: 3D vector (n, 3)
        """
        vector0 = vector0.view(-1, 3)
        if rotation_vector is None:
            return vector0
        
        theta = torch.norm(rotation_vector)
        if theta == 0:
            return vector0
        else:
            rotation_vector = rotation_vector / theta
            rotation_vector = rotation_vector.view(1, 3)
            vector1 = vector0 * torch.cos(theta) + torch.cross(
                rotation_vector, vector0, dim=1) * torch.sin(
                    theta) + rotation_vector * (rotation_vector * vector0).sum(
                        dim=1).unsqueeze(-1) * (1 - torch.cos(theta))
        return vector1

    def set_required_DoFs(self, RGC_remain_index):
        for e in self.part.elems.values():
            RGC_remain_index[self._RGC_index] = e.set_required_DoFs(RGC_remain_index[self._RGC_index])
        return RGC_remain_index
    
    def set_RGC_index(self, index):
        super().set_RGC_index(index)
    
    def initialize(self, assembly: Assembly):
        
        super().initialize(assembly=assembly)

    def get_mass_matrix(self):
        mass_indices, mass_values = self.part.get_mass_matrix(self.rotation_matrix)
        return mass_indices + self._index_start, mass_values

    def refine_RGC(self, RGC: list[torch.Tensor]) -> list[torch.Tensor]:
        RGC_out = RGC
        RGC_out[self._RGC_index] = self.part.refine_RGC(RGC[self._RGC_index])
        return RGC_out
    
    def potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        return self.part.potential_energy(RGC=RGC[self._RGC_index], rotation_matrix=self.rotation_matrix)
    
    def structural_stiffness(self, RGC: list[torch.Tensor], if_onlyforce: bool = False, *args, **kwargs) -> list[torch.Tensor]:

        if if_onlyforce:
            R_indices, R_values = self.part.structural_force(RGC[self._RGC_index], rotation_matrix=self.rotation_matrix)
            return R_indices + self._index_start, R_values
        
        R_indices, R_values, K_indices, K_values = self.part.structural_stiffness(RGC[self._RGC_index], rotation_matrix=self.rotation_matrix)
        return R_indices + self._index_start, R_values, K_indices + self._index_start, K_values
    
    def extract_surfaces(self, name: str) -> list[BaseSurface]:
        surfaces = self.part.extract_surfaces(name)
        return surfaces

    def get_mesh(self, RGC: list[torch.Tensor] = None, surf_name: str = None) -> pv.PolyData:
        """
        Get the mesh of the instance for visualization.
        Args:
            surf_name (str, optional): Name of the surface set to visualize. 
                If None, use the Part exterior surface. Defaults to None.
        Returns:
            pv.PolyData: The mesh of the instance.
        """
        import pyvista as pv

        if surf_name is None:
            surf_name = self.exterior_surface
        if surf_name == '':
            return pv.PolyData()
        
        triangles = self.surfaces.get_trimesh(surf_name).cpu().numpy()
        nodes_transformed = self.nodes.cpu().numpy()

        if RGC is not None:
            nodes_transformed += RGC[self._RGC_index].cpu().numpy()

        mesh = pv.PolyData(nodes_transformed, np.hstack([ np.full((triangles.shape[0], 1), 3), triangles ]))
        return mesh
    
    def show(self, RGC: list[torch.Tensor] = None, surf_name: str = None):
        mesh = self.get_mesh(RGC=RGC, surf_name=surf_name)
        if mesh.n_points == 0:
            print("No surface to show.")
            return
        plotter = pv.Plotter()
        try:
            plotter.add_mesh(mesh, color='lightblue', show_edges=True)
            plotter.show()
        finally:
            plotter.close()

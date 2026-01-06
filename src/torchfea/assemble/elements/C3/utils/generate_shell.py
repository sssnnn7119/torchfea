from __future__ import annotations
from platform import node
from typing import Optional, TYPE_CHECKING

import torch
import numpy as np



if TYPE_CHECKING:
    from ....Main import FEA_Main

def calculate_new_nodes(node_idx_map: torch.Tensor, nodes: torch.Tensor, surface_elems: torch.Tensor, surface_node_indices: torch.Tensor, shell_thickness: float):
    """
    Calculate the averaged normal vectors for each node in a triangular mesh.
    
    Args:
        node_idx_map (torch.Tensor): Mapping from global node indices to local indices
        nodes (torch.Tensor): Tensor of node coordinates (shape: [num_nodes, 3])
        surface_elems (torch.Tensor): Tensor of surface elements (triangles) (shape: [num_elements, 3])
        surface_node_indices (torch.Tensor): Unique node indices for the surface elements
        shell_thickness (float): Thickness to offset the nodes by in the normal direction   
    Returns:
        node_normals: Tensor of normalized normal vectors for each node
    """
    # Extract vertices for each triangle
    v0 = nodes[surface_elems[:, 0]]
    v1 = nodes[surface_elems[:, 1]]
    v2 = nodes[surface_elems[:, 2]]
    
    # Calculate normal vectors using cross product (vectorized)
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)

    # Normalize the normal vectors (vectorized)
    normal_lengths = torch.norm(normals, dim=1, keepdim=True)
    mask = normal_lengths > 1e-10  # Avoid division by zero
    normals = torch.where(mask, normals / normal_lengths, normals)

    # Create the indices for our sparse accumulation matrix
    rows = torch.cat([
        node_idx_map[surface_elems[:, 0]], node_idx_map[surface_elems[:, 1]],
        node_idx_map[surface_elems[:, 2]]
    ])
    cols = torch.cat(
        [torch.arange(surface_elems.shape[0], device=nodes.device)] * 3)

    # Create values (ones) for our sparse tensor
    values = torch.ones(rows.shape[0],
                        device=nodes.device,
                        dtype=nodes.dtype)

    # Create a node-triangle sparse matrix and use it to compute node normals
    sparse_indices = torch.stack([rows, cols])
    node_triangle_map = torch.sparse_coo_tensor(
        sparse_indices,
        values,
        size=(surface_node_indices.shape[0], surface_elems.shape[0]),
        device=nodes.device)

    # Multiply the sparse matrix by the normals to get the sum of normals for each node
    node_normals = torch.sparse.mm(node_triangle_map, normals)
    
    # Normalize the averaged normals (vectorized)
    normal_lengths = torch.norm(node_normals, dim=1, keepdim=True)
    mask = normal_lengths > 1e-10  # Avoid division by zero
    node_normals = torch.where(mask, node_normals / normal_lengths, node_normals)

    # Smooth the normals by averaging over the normals of neighboring nodes
    # First, build an adjacency matrix for the surface mesh
    rows = torch.cat([
        surface_elems[:, 0], surface_elems[:, 0], 
        surface_elems[:, 1], surface_elems[:, 1],
        surface_elems[:, 2], surface_elems[:, 2]
    ])
    cols = torch.cat([
        surface_elems[:, 1], surface_elems[:, 2],
        surface_elems[:, 0], surface_elems[:, 2],
        surface_elems[:, 0], surface_elems[:, 1]
    ])

    # Map global indices to local indices
    rows_local = node_idx_map[rows]
    cols_local = node_idx_map[cols]

    # Create the adjacency matrix (symmetrical)
    values = torch.ones(rows_local.shape[0], device=nodes.device, dtype=nodes.dtype)
    sparse_indices = torch.stack([rows_local, cols_local])
    adjacency = torch.sparse_coo_tensor(
        sparse_indices,
        values,
        size=(surface_node_indices.shape[0], surface_node_indices.shape[0]),
        device=nodes.device
    )

    # Apply smoothing: average each node normal with its neighbors
    neighbor_sum = torch.sparse.mm(adjacency, node_normals)
    neighbor_count = torch.sparse.sum(adjacency, dim=1).to_dense().unsqueeze(1)
    smoothed_normals = neighbor_sum / (neighbor_count + 1e-10)  # Add small epsilon to avoid division by zero

    # Normalize the smoothed normals
    normal_lengths = torch.norm(smoothed_normals, dim=1, keepdim=True)
    mask = normal_lengths > 1e-10
    node_normals = torch.where(mask, smoothed_normals / normal_lengths, smoothed_normals)

    # Create new nodes by offsetting the original nodes by the thickness (vectorized)
    new_nodes = nodes[surface_node_indices] + node_normals * shell_thickness
    
    return new_nodes

def generate_shell_from_surface(
        fe: FEA_Main, surface_names: str, shell_thickness: float, surface_new_name: Optional[str] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Generate shell elements (C3D6) from triangular surface meshes.
    
    This function takes one or more surfaces (identified by their names in the FEA model),
    and generates wedge elements (C3D6) by extruding the triangular faces
    along their averaged normal directions by a specified thickness.
    
    Args:
        fe (FEA.Main.FEA_Main): The FEA model instance containing the surfaces
        surface_names (str): Name(s) of the surfaces to extrude (e.g., 'surface_1_All' 
                                   or ['surface_1_All', 'surface_2_All'])
        shell_thickness (float): The thickness to extrude the surface by
        surface_new_name (str, optional): New names for the offset surfaces.
    Returns:
        tuple: A tuple containing:
            - nodes_new (torch.Tensor): Combined node coordinates (original + new nodes)
            - elems_c3d6 (torch.Tensor): Element connectivity for the new C3D6 elements
            - c3d6_indices (torch.Tensor): Indices for the new C3D6 elements
            - offset_surface_sets (dict): Dictionary mapping surface names to offset surface sets
    """

    # Get and combine the triangular elements of all surfaces
    surface_elems = fe.get_surface_elements(surface_names)[0]._elems

    # Get the unique node indices from the triangular elements
    surface_node_indices = torch.unique(
        surface_elems
    )  # Calculate triangle normals for each triangle in the surface using vectorized operations

    # Create a tensor for mapping from global node indices to local indices
    max_node_idx = torch.max(surface_elems).item()
    node_idx_map = torch.full((max_node_idx + 1, ),
                            -1,
                            device=fe.nodes.device,
                            dtype=torch.int64)
    node_idx_map[surface_node_indices] = torch.arange(
        surface_node_indices.shape[0], device=fe.nodes.device)

    # Calculate new nodes by offsetting the original nodes in the normal direction
    new_nodes = calculate_new_nodes(node_idx_map=node_idx_map, nodes=fe.nodes, surface_elems=surface_elems, surface_node_indices=surface_node_indices, shell_thickness=shell_thickness)

    # Create C3D6 (wedge) elements
    # Each triangle in the surface becomes a C3D6 element
    # C3D6 connectivity: [bottom_triangle_node0, bottom_triangle_node1, bottom_triangle_node2,
    #                    top_triangle_node0, top_triangle_node1, top_triangle_node2]
    c3d6_elements = torch.zeros(
        (surface_elems.shape[0], 6), device=fe.nodes.device,
        dtype=torch.int64)  # Create the C3D6 elements (vectorized)
    # Original triangle nodes form the base (first 3 columns)
    c3d6_elements[:, 0] = surface_elems[:, 0]
    c3d6_elements[:, 1] = surface_elems[:, 1]
    c3d6_elements[:, 2] = surface_elems[:, 2]

    # New nodes form the top (last 3 columns)
    # Use the same node_idx_map from earlier for mapping node indices
    c3d6_elements[:, 3] = fe.nodes.shape[0] + node_idx_map[surface_elems[:, 0]]
    c3d6_elements[:, 4] = fe.nodes.shape[0] + node_idx_map[surface_elems[:, 1]]
    c3d6_elements[:, 5] = fe.nodes.shape[0] + node_idx_map[surface_elems[:, 2]]

    # Element indices for the new C3D6 elements - find the largest existing element index
    max_elem_index = max([(torch.max(elem_group._elems_index).item()
                           if hasattr(elem_group, '_elems_index')
                           and elem_group._elems_index.numel() > 0 else 0)
                          for elem_group in fe.elems.values()],
                         default=0)

    c3d6_indices = torch.arange(max_elem_index + 1,
                                max_elem_index + 1 + surface_elems.shape[0],
                                device=fe.nodes.device,
                                dtype=torch.int64)    # Combine the original nodes with the new nodes
    nodes_new = torch.cat([fe.nodes, new_nodes], dim=0)

    # Create a dictionary to store the offset surface sets
    # For each original surface name, we'll create a set for the top face (Surface 1) of the wedge elements
    offset_surface_sets = {}

    # Track which wedge elements correspond to each original surface
    start_idx = 0
    
    # Count triangles in this surface
    num_triangles = surface_elems.shape[0]

    # Create surface set for the offset surface (face index 1 - top triangular face)
    # The format needed is a list of [element_index, surface_index] pairs
    offset_surface = []
    for j in range(num_triangles):
        elem_index = c3d6_indices[start_idx + j].item()
        # Surface index 1 corresponds to the top triangular face (nodes 3, 4, 5) in C3D6
        surf_index = 1
        offset_surface.append(elem_index)
    
    # Store the offset surface set
    if offset_surface:
        if surface_new_name is not None:
            offset_name = surface_new_name
        else:
            offset_name = f"{surface_names}_offset"

        offset_surface_sets[offset_name] = [(np.array(offset_surface), 1)]
    
    # Update the starting index for the next surface
    start_idx += num_triangles

    return nodes_new, c3d6_elements.cpu(), c3d6_indices.cpu(), offset_surface_sets


def add_shell_elements_to_model(fe: FEA_Main, nodes_new: torch.Tensor,
                                c3d6_elements: torch.Tensor,
                                c3d6_indices: torch.Tensor,
                                name_new_elements: str = "shell_elements",
                                offset_surface_sets: dict = None):
    """
    Add the generated shell elements to the FEA model.
    
    Args:
        fe (FEA.Main.FEA_Main): The FEA model instance
        nodes_new (torch.Tensor): Combined node coordinates (original + new nodes)
        c3d6_elements (torch.Tensor): Element connectivity for the new C3D6 elements
        c3d6_indices (torch.Tensor): Indices for the new C3D6 elements
        offset_surface_sets (dict, optional): Dictionary mapping surface names to offset surface sets
        
    Returns:
        FEA.Main.FEA_Main: Updated FEA model with the shell elements added
    """
    import torchfea
    # Create a new FEA_Main instance with the updated nodes
    new_fe = torchfea.controller.FEAController(nodes_new)

    # Copy all the original elements from the old model
    for elem_name, elem_obj in fe.elems.items():
        new_fe.elems[elem_name] = elem_obj

    # Create new C3D6 elements and add them to the model
    new_fe.add_element(torchfea.elements.C3.C3D6(elems=c3d6_elements,
                                            elems_index=c3d6_indices),
                       name=name_new_elements)    # Copy node sets, element sets, and surface sets
    for name, node_set in fe.node_sets.items():
        new_fe.add_node_set(name, node_set)

    for name, elem_set in fe.element_sets.items():
        new_fe.add_element_set(name, elem_set)

    for name, surf_set in fe.surface_sets.items():
        new_fe.add_surface_set(name, surf_set)
        
    # Add the offset surface sets if provided
    if offset_surface_sets:
        for name, surf_set in offset_surface_sets.items():
            new_fe.add_surface_set(name, surf_set)

    # Copy reference points
    for rp_name, rp in fe.reference_points.items():
        new_rp = torchfea.ReferencePoint(rp.node)
        new_fe.add_reference_point(new_rp, name=rp_name)

    # Copy loads
    for load_name, load_obj in fe.loads.items():
        new_fe.loads[load_name] = load_obj

    # Copy constraints
    for const_name, const_obj in fe.constraints.items():
        new_fe.constraints[const_name] = const_obj

    return new_fe

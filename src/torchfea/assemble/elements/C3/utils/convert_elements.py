from __future__ import annotations
from calendar import c
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ....Main import FEA_Main

import torch
import numpy as np
import torchfea

from torchfea.assemble.reference_points import ReferencePoint

from .. import C3D4, C3D6, C3D8, C3D10, C3D15, C3D20


def fast_edge_lookup(edge_dict: dict, edges_tensor: torch.Tensor):
    """
    Faster lookup method using dictionary for edge to value mapping.
    
    Args:
        edge_dict (dict): Dictionary mapping edge tuples to values
        edges_tensor (torch.Tensor): Edges to look up, shape [B, 2]
        
    Returns:
        torch.Tensor: Values for each input edge, shape [B]
    """
    # Sort edges to ensure consistent ordering
    edges_sorted, _ = torch.sort(edges_tensor, dim=1)
    
    # Convert to list of tuples for dictionary lookup
    edge_tuples = [tuple(edge.tolist()) for edge in edges_sorted]
    
    # Look up values
    result = []
    for edge_tuple in edge_tuples:
        if edge_tuple in edge_dict:
            result.append(edge_dict[edge_tuple])
        else:
            result.append(-1)  # Default value
            
    return torch.tensor(result, dtype=torch.int64, device=edges_tensor.device)

def convert_to_second_order(fe: FEA_Main, element_names: list[str]=None)-> FEA_Main:
    """
    Convert first-order elements (C3D4, C3D6) to second-order elements (C3D10, C3D15).
    
    This function creates mid-edge nodes for first-order elements and updates 
    element connectivity to create their second-order equivalents.
    
    Args:
        fe (FEA.Main.FEA_Main): The FEA model instance containing elements to convert
        element_names (list, optional): List of element set names in fe.elems to convert.
                                       These are names of element sets as they appear in the fe.elems dictionary,
                                       such as 'element-0' or 'shell_elements', not the element types themselves.
                                       If None, converts all supported first-order elements.
    
    Returns:
        FEA.Main.FEA_Main: Updated FEA model with second-order elements
    """
    if element_names is None:
        # Default to converting all supported element set names that contain first-order elements
        element_names = []
        for name, elem_obj in fe.elems.items():
            if isinstance(elem_obj, C3D4) or isinstance(elem_obj, C3D6):
                element_names.append(name)
    
    # Filter to only include element names that exist in the model
    element_names = [elem_name for elem_name in element_names 
                    if elem_name in fe.elems]
    
    if not element_names:
        print("No convertible elements found in the model.")
        return fe
    
    # Store original nodes
    original_nodes = fe.nodes.clone()
    num_original_nodes = original_nodes.shape[0]
    
    # Dictionary to track mid-edge nodes (to avoid duplicates)
    # Key is a sorted tuple of two node indices, value is the new node index
    # This dictionary is shared across all element types to handle interfaces between different element types
    edge_nodes = {}
    
    # Prepare to collect all edges from all elements for pre-processing
    all_edges = []
    element_data = []    # Collect all edges from all elements using vectorized operations
    all_edges_tensor = torch.zeros((0, 2), dtype=torch.int64, device=fe.nodes.device)
    
    # Define edge pairs once for each element type
    tetra_edge_pairs = torch.tensor([
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
    ], dtype=torch.int64)
    
    wedge_edge_pairs = torch.tensor([
        [0, 1], [0, 2], [1, 2],  # Bottom triangle
        [3, 4], [3, 5], [4, 5],  # Top triangle
        [0, 3], [1, 4], [2, 5]   # Vertical edges
    ], dtype=torch.int64)
    
    # Lists to store batches of different element types for concurrent processing
    c3d4_batches = []  # List of (elems, indices, elem_name) tuples
    c3d6_batches = []  # List of (elems, indices, elem_name) tuples
    
    # Collect all element batches first
    for elem_name in element_names:
        elem_obj = fe.elems[elem_name]
        
        if isinstance(elem_obj, C3D4):
            c3d4_batches.append((elem_obj._elems, elem_obj._elems_index, elem_name))
        elif isinstance(elem_obj, C3D6):
            c3d6_batches.append((elem_obj._elems, elem_obj._elems_index, elem_name))
    
    # Process all C3D4 elements in a fully vectorized manner
    all_c3d4_edges = torch.zeros((0, 2), dtype=torch.int64, device='cpu')
    for elems, indices, elem_name in c3d4_batches:
        if elems.shape[0] > 0:
            # Put edge pairs on the correct device
            tetra_edge_pairs = tetra_edge_pairs.to(elems.device)
            
            # Get all edges at once [num_elements, 6, 2]
            # First reshape to get the node indices for each edge pair
            edge_nodes_indices = elems[:, tetra_edge_pairs].reshape(-1, 2)
            
            # Sort edges to ensure consistent ordering
            edge_nodes_sorted, _ = torch.sort(edge_nodes_indices, dim=1)
            
            # Add to our collection
            all_c3d4_edges = torch.cat([all_c3d4_edges, edge_nodes_sorted], dim=0)
            
    # Process all C3D6 elements in a fully vectorized manner
    all_c3d6_edges = torch.zeros((0, 2), dtype=torch.int64, device='cpu')
    for elems, indices, elem_name in c3d6_batches:
        if elems.shape[0] > 0:
            # Put edge pairs on the correct device
            wedge_edge_pairs = wedge_edge_pairs.to(elems.device)
            
            # Get all edges at once [num_elements, 9, 2]
            # First reshape to get the node indices for each edge pair
            edge_nodes_indices = elems[:, wedge_edge_pairs].reshape(-1, 2)
            
            # Sort edges to ensure consistent ordering
            edge_nodes_sorted, _ = torch.sort(edge_nodes_indices, dim=1)
            
            # Add to our collection
            all_c3d6_edges = torch.cat([all_c3d6_edges, edge_nodes_sorted], dim=0)
    
    # Combine all edges and find unique edges
    all_edges_tensor = torch.cat([all_c3d4_edges, all_c3d6_edges], dim=0)
    
    # Convert edges tensor to list of tuples for uniqueness checking
    all_edges = [tuple(edge.tolist()) for edge in all_edges_tensor]
      # Create unique edges and their midpoint nodes using vectorized operations
    unique_edges = list(set(all_edges))
    
    # Vectorized calculation of midpoint nodes
    if unique_edges:
        # Extract node indices from edges
        edge_nodes_indices = torch.tensor(unique_edges, dtype=torch.int64, device='cpu')
        
        # Get node positions for all edges at once
        edge_start_positions = fe.nodes[edge_nodes_indices[:, 0]]
        edge_end_positions = fe.nodes[edge_nodes_indices[:, 1]]
        
        # Calculate midpoint positions in one operation
        midpoint_positions = (edge_start_positions + edge_end_positions) / 2
        
        # Store the midpoint positions and update edge_nodes dictionary
        for i, edge in enumerate(unique_edges):
            edge_nodes[edge] = num_original_nodes + i
        
        # Store new nodes
        new_nodes = midpoint_positions
    
    # Lists to store new elements and their indices
    c3d10_elements = []
    c3d10_indices = []
    c3d15_elements = []
    c3d15_indices = []    # Process each element set
    c3d20_elements = []
    c3d20_indices = []
    for elem_name in element_names:
        elem_obj = fe.elems[elem_name]
        
        if isinstance(elem_obj, C3D4):
            # Convert C3D4 to C3D10 using fully vectorized operations
            c3d4_elems = elem_obj._elems
            c3d4_indices = elem_obj._elems_index
            num_elements = c3d4_elems.shape[0]
            
            if num_elements > 0:
                # Create C3D10 elements batch (all elements at once)
                c3d10_batch = torch.zeros((num_elements, 10), dtype=torch.int64, device=fe.nodes.device)
                
                # First 4 nodes are the same as C3D4 (corner nodes)
                c3d10_batch[:, :4] = c3d4_elems
                # Define edge pairs for tetrahedron (local indices)
                # Ordered to match faces:
                # face0: 0(6)2(5)1(4) => edges [0,2], [2,1], [0,1]
                # face1: 0(4)1(8)3(7) => edges [0,1], [1,3], [0,3]
                # face2: 1(5)2(9)3(8) => edges [1,2], [2,3], [1,3]
                # face3: 0(7)3(9)2(6) => edges [0,3], [3,2], [0,2]
                tetra_edge_pairs = torch.tensor([
                    [0, 1],  # Edge 0-1, midpoint becomes node 4
                    [1, 2],  # Edge 1-2, midpoint becomes node 5
                    [0, 2],  # Edge 0-2, midpoint becomes node 6
                    [0, 3],  # Edge 0-3, midpoint becomes node 7
                    [1, 3],  # Edge 1-3, midpoint becomes node 8
                    [2, 3]   # Edge 2-3, midpoint becomes node 9
                ], dtype=torch.int64, device=c3d4_elems.device)
                
                # Get node indices for all elements at once using advanced indexing
                # Shape: [num_elements, 6, 2] - for each element, for each of the 6 edges, the 2 nodes
                edge_nodes_indices = c3d4_elems[:, tetra_edge_pairs].reshape(num_elements, 6, 2)
                
                # Sort each edge to ensure first node < second node
                edge_nodes_sorted, _ = torch.sort(edge_nodes_indices, dim=2)
                  # Create a lookup table for all edges to their midpoint nodes
                if edge_nodes:  # Only proceed if we have edge nodes
                    # Get all edges that need to be processed
                    all_edges_flat = edge_nodes_sorted.reshape(-1, 2)  # shape: [num_elements*6, 2]
                    
                    # Use the optimized edge lookup function for better performance
                    # Process each position separately for more efficient lookup
                    c3d10_batch[:, 4:] = -1  # Initialize with an invalid index

                    # Process all 6 positions at once using vectorized operations
                    for j in range(6):
                        # Get current position edges for all elements
                        current_edges = edge_nodes_sorted[:, j, :]
                        
                        # Use our optimized lookup function
                        midpoint_indices = fast_edge_lookup(edge_nodes, current_edges)
                        
                        # Assign the midpoint indices to the correct positions in the batch
                        c3d10_batch[:, j + 4] = midpoint_indices
                
                # Store the elements and indices directly
                c3d10_elements.append(c3d10_batch)
                c3d10_indices.append(c3d4_indices)
                
        elif isinstance(elem_obj, C3D6):
            # Convert C3D6 to C3D15 using vectorized operations
            c3d6_elems = elem_obj._elems
            c3d6_indices = elem_obj._elems_index
            num_elements = c3d6_elems.shape[0]
            
            if num_elements > 0:
                # Create C3D15 elements batch (all elements at once)
                c3d15_batch = torch.zeros((num_elements, 15), dtype=torch.int64, device=fe.nodes.device)
                
                # First 6 nodes are the same as C3D6 (corner nodes)
                c3d15_batch[:, :6] = c3d6_elems                # Define edge pairs for wedge (local indices)
                # Ordered to match faces:
                # face0: 0(8)2(7)1(6) => Bottom triangle edges [0,2], [2,1], [0,1]
                # face1: 3(9)4(10)5(11) => Top triangle edges [3,4], [4,5], [3,5]
                # face2: 0(6)1(13)4(9)3(12) => Rectangle edges [0,1], [1,4], [4,3], [3,0]
                # face3: 1(7)2(14)5(10)4(13) => Rectangle edges [1,2], [2,5], [5,4], [4,1]
                # face4: 2(8)0(12)3(11)5(14) => Rectangle edges [2,0], [0,3], [3,5], [5,2]
                wedge_edge_pairs = torch.tensor([
                    [0, 1],  # Edge 0-1, midpoint becomes node 6
                    [1, 2],  # Edge 1-2, midpoint becomes node 7
                    [0, 2],  # Edge 0-2, midpoint becomes node 8
                    [3, 4],  # Edge 3-4, midpoint becomes node 9
                    [4, 5],  # Edge 4-5, midpoint becomes node 10
                    [3, 5],  # Edge 3-5, midpoint becomes node 11
                    [0, 3],  # Edge 0-3, midpoint becomes node 12
                    [1, 4],  # Edge 1-4, midpoint becomes node 13
                    [2, 5]   # Edge 2-5, midpoint becomes node 14
                ], dtype=torch.int64, device=c3d6_elems.device)
                
                # Get node indices for all elements at once using advanced indexing
                # Shape: [num_elements, 9, 2] - for each element, for each of the 9 edges, the 2 nodes
                edge_nodes_indices = c3d6_elems[:, wedge_edge_pairs].reshape(num_elements, 9, 2)
                
                # Sort each edge to ensure first node < second node
                edge_nodes_sorted, _ = torch.sort(edge_nodes_indices, dim=2)
                
                # Initialize the mid-edge nodes with -1 (invalid index)
                c3d15_batch[:, 6:] = -1
                  # Process all 9 positions at once using vectorized operations
                for j in range(9):
                    # Get current position edges for all elements
                    current_edges = edge_nodes_sorted[:, j, :]
                    
                    # Use our optimized lookup function
                    midpoint_indices = fast_edge_lookup(edge_nodes, current_edges)
                    
                    # Assign the midpoint indices to the correct positions in the batch
                    c3d15_batch[:, j + 6] = midpoint_indices
                
                # Store the batch directly
                c3d15_elements.append(c3d15_batch)
                c3d15_indices.append(c3d6_indices)

        elif isinstance(elem_obj, C3D8):
            # Convert C3D8 to C3D20 using vectorized operations
            c3d8_elems = elem_obj._elems
            c3d8_indices = elem_obj._elems_index
            num_elements = c3d8_elems.shape[0]
            
            if num_elements > 0:
                # Create C3D20 elements batch (all elements at once)
                c3d20_batch = torch.zeros((num_elements, 20), dtype=torch.int64, device=fe.nodes.device)
                
                # First 8 nodes are the same as C3D8 (corner nodes)
                c3d20_batch[:, :8] = c3d8_elems
                
                # Define edge pairs for hexahedron (local indices)
                hex_edge_pairs = torch.tensor([
                    [0, 1],  # Edge 0-1, midpoint becomes node 8
                    [1, 2],  # Edge 1-2, midpoint becomes node 9
                    [2, 3],  # Edge 2-3, midpoint becomes node 10
                    [3, 0],  # Edge 3-0, midpoint becomes node 11
                    [4, 5],  # Edge 4-5, midpoint becomes node 12
                    [5, 6],  # Edge 5-6, midpoint becomes node 13
                    [6, 7],  # Edge 6-7, midpoint becomes node 14
                    [7, 4],  # Edge 7-4, midpoint becomes node 15
                    [0, 4],  # Edge 0-4, midpoint becomes node 16
                    [1, 5],  # Edge 1-5, midpoint becomes node 17
                    [2, 6],  # Edge 2-6, midpoint becomes node 18
                    [3, 7]   # Edge 3-7, midpoint becomes node 19
                ], dtype=torch.int64, device=c3d8_elems.device)
                
                # Get node indices for all elements at once
                edge_nodes_indices = c3d8_elems[:, hex_edge_pairs].reshape(num_elements, 12, 2)
                
                # Sort each edge to ensure first node < second node
                edge_nodes_sorted, _ = torch.sort(edge_nodes_indices, dim=2)
                
                # Initialize the mid-edge nodes with -1 (invalid index)
                c3d20_batch[:, 8:] = -1
                
                # Process all 12 positions at once using vectorized operations
                for j in range(12):
                    # Get current position edges for all elements
                    current_edges = edge_nodes_sorted[:, j, :]
                    
                    # Use our optimized lookup function
                    midpoint_indices = fast_edge_lookup(edge_nodes, current_edges)
                    
                    # Assign the midpoint indices to the correct positions in the batch
                    c3d20_batch[:, j + 8] = midpoint_indices
                
                # Store the batch directly
                c3d20_elements.append(c3d20_batch)
                c3d20_indices.append(c3d8_indices)



    # Combine original and new nodes
    if not unique_edges:
        # No new nodes were created
        combined_nodes = original_nodes
    else:
        # new_nodes is a tensor from our vectorized operations
        combined_nodes = torch.cat([original_nodes, new_nodes])
    
    # Create a new FEA model with the updated nodes
    new_fe = torchfea.controller.FEAController(combined_nodes)
    
    # Copy non-converted elements from the original model
    for elem_name, elem_obj in fe.elems.items():
        if elem_name not in element_names:
            new_fe.elems[elem_name] = elem_obj
    
    # Add the new second-order elements
    for i in range(len(element_names)):
        elem_name = element_names[i]
        elem_obj = fe.elems[elem_name]
        
        # Determine what kind of element this was and add the converted elements
        if isinstance(elem_obj, C3D4):
            if len(c3d10_elements) > 0:
                new_fe.add_element(
                    C3D10(elems=c3d10_elements[i].cpu(), elems_index=c3d10_indices[i].cpu()),
                    name=elem_name
                )
        elif isinstance(elem_obj, C3D6):
            if len(c3d15_elements) > 0:
                new_fe.add_element(
                    C3D15(elems=c3d15_elements[i].cpu(), elems_index=c3d15_indices[i].cpu()),
                    name=elem_name
                )
        elif isinstance(elem_obj, C3D8):
            if len(c3d20_elements) > 0:
                new_fe.add_element(
                    C3D20(elems=c3d20_elements[i].cpu(), elems_index=c3d20_indices[i].cpu()),
                    name=elem_name
                )

    
    # Copy node sets, element sets, and surface sets
    for name, node_set in fe.node_sets.items():
        new_fe.add_node_set(name, node_set)
    
    for name, elem_set in fe.element_sets.items():
        new_fe.add_element_set(name, elem_set)
    
    for name, surf_set in fe.surface_sets._surface_dict.items():
        new_fe.add_surface_set(name, surf_set)
    
    # Copy reference points
    for rp_name, rp in fe.reference_points.items():
        new_rp = torchfea.ReferencePoint(rp.node)
        new_fe.add_reference_point(new_rp, name=rp_name)
    
    # Copy loads
    for load_name, load_obj in fe.loads.items():
        new_fe.loads[load_name] = load_obj
    
    return new_fe


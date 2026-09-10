from .part import Part
from .elements import initialize_element
import torch
import numpy as np
def create_box(xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float, nx: int, ny: int, nz: int, element_name: str = 'C3D8'):
    """
    Create a box geometry defined by the given dimensions and number of elements.

    Args:
        xmin (float): Minimum x-coordinate of the box.
        xmax (float): Maximum x-coordinate of the box.
        ymin (float): Minimum y-coordinate of the box.
        ymax (float): Maximum y-coordinate of the box.
        zmin (float): Minimum z-coordinate of the box.
        zmax (float): Maximum z-coordinate of the box.
        nx (int): Number of nodes along the x-axis.
        ny (int): Number of nodes along the y-axis.
        nz (int): Number of nodes along the z-axis.

    Returns:
        nodes (torch.Tensor): Tensor of shape (num_nodes, 3) containing the coordinates            of the nodes.
        elements (torch.Tensor): Tensor of shape (num_elements, 8) containing the node            indices for each element.
    """

    assert xmin < xmax, "xmin must be less than xmax"
    assert ymin < ymax, "ymin must be less than ymax"
    assert zmin < zmax, "zmin must be less than zmax"
    assert nx > 0, "nx must be greater than 0"
    assert ny > 0, "ny must be greater than 0"
    assert nz > 0, "nz must be greater than 0"

    x = torch.linspace(xmin, xmax, nx)
    y = torch.linspace(ymin, ymax, ny)
    z = torch.linspace(zmin, zmax, nz)

    nodes = torch.stack(torch.meshgrid(x, y, z), dim=-1).reshape(-1, 3)

    # Strides
    stride_x = ny * nz
    stride_y = nz
    stride_z = 1
    
    # Create elements in Abaqus node order
    elements = torch.zeros(nx-1, ny-1, nz-1, 8, dtype=torch.long)
    
    # Base nodes (local node 0, corresponding to Abaqus node 1)
    base = torch.arange(0, (nx-1) * stride_x, stride_x, dtype=torch.long)[:, None, None] \
        + torch.arange(0, (ny-1) * stride_y, stride_y, dtype=torch.long)[None, :, None] \
        + torch.arange(0, nz-1, dtype=torch.long)[None, None, :]
    
    # Map to Abaqus node order: 1, 2, 3, 4, 5, 6, 7, 8
    elements[:, :, :, 0] = base                    # Node 1: bottom-front-left
    elements[:, :, :, 1] = base + stride_x         # Node 2: bottom-front-right
    elements[:, :, :, 2] = base + stride_x + stride_y  # Node 3: bottom-back-right
    elements[:, :, :, 3] = base + stride_y         # Node 4: bottom-back-left
    elements[:, :, :, 4] = base + stride_z         # Node 5: top-front-left
    elements[:, :, :, 5] = base + stride_x + stride_z   # Node 6: top-front-right
    elements[:, :, :, 6] = base + stride_x + stride_y + stride_z  # Node 7: top-back-right
    elements[:, :, :, 7] = base + stride_y + stride_z   # Node 8: top-back-left
    
    part = Part(nodes=nodes)

    node_per_elem = 8
    elem_type = f'C3D{node_per_elem}'

    if element_name is None:
        element_name = elem_type

    elems = initialize_element(element_type=elem_type, elems_index=torch.arange((nx-1)*(ny-1)*(nz-1)), elems=elements.reshape(-1, 8))
    
    part.add_element(elems, name=element_name)

    # surface_xmin: x=0
    elem_indices_xmin = np.arange((nx-1)*(ny-1)*(nz-1)).reshape(nx-1, ny-1, nz-1)[0, :, :].reshape(-1)
    part.add_surface_set(
        name='surface_xmin', 
        elements=[(elem_indices_xmin, 5)]
    )

    # surface_xmax: x=nx-1
    elem_indices_xmax = np.arange((nx-1)*(ny-1)*(nz-1)).reshape(nx-1, ny-1, nz-1)[-1, :, :].reshape(-1)
    part.add_surface_set(
        name='surface_xmax', 
        elements=[(elem_indices_xmax, 3)]
    )

    # surface_ymin: y=0
    elem_indices_ymin = np.arange((nx-1)*(ny-1)*(nz-1)).reshape(nx-1, ny-1, nz-1)[:, 0, :].reshape(-1)
    part.add_surface_set(
        name='surface_ymin', 
        elements=[(elem_indices_ymin, 2)]
    )

    # surface_ymax: y=ny-1
    elem_indices_ymax = np.arange((nx-1)*(ny-1)*(nz-1)).reshape(nx-1, ny-1, nz-1)[:, -1, :].reshape(-1)
    part.add_surface_set(
        name='surface_ymax', 
        elements=[(elem_indices_ymax, 4)]
    )

    # surface_zmin: z=0
    elem_indices_zmin = np.arange((nx-1)*(ny-1)*(nz-1)).reshape(nx-1, ny-1, nz-1)[:, :, 0].reshape(-1)
    part.add_surface_set(
        name='surface_zmin', 
        elements=[(elem_indices_zmin, 0)]
    )

    # surface_zmax: z=nz-1
    elem_indices_zmax = np.arange((nx-1)*(ny-1)*(nz-1)).reshape(nx-1, ny-1, nz-1)[:, :, -1].reshape(-1)
    part.add_surface_set(
        name='surface_zmax', 
        elements=[(elem_indices_zmax, 1)]
    )

    # define node sets for boundary conditions
    total_nodes = nx * ny * nz

    nodes_grid = np.arange(total_nodes).reshape(nx, ny, nz)

    # xmin and xmax: fixed x indices
    nodes_xmin = nodes_grid[0, :, :].reshape(-1)      # All nodes at x=0
    nodes_xmax = nodes_grid[-1, :, :].reshape(-1)     # All nodes at x=nx-1

    # ymin and ymax: fixed y indices
    nodes_ymin = nodes_grid[:, 0, :].reshape(-1)      # All nodes at y=0
    nodes_ymax = nodes_grid[:, -1, :].reshape(-1)     # All nodes at y=ny-1

    # zmin and zmax: fixed z indices
    nodes_zmin = nodes_grid[:, :, 0].reshape(-1)      # All nodes at z=0
    nodes_zmax = nodes_grid[:, :, -1].reshape(-1)     # All nodes at z=nz-1

    part.add_node_set(name='nodes_xmin', node_indices=nodes_xmin)
    part.add_node_set(name='nodes_xmax', node_indices=nodes_xmax)
    part.add_node_set(name='nodes_ymin', node_indices=nodes_ymin)
    part.add_node_set(name='nodes_ymax', node_indices=nodes_ymax)
    part.add_node_set(name='nodes_zmin', node_indices=nodes_zmin)
    part.add_node_set(name='nodes_zmax', node_indices=nodes_zmax)

    part.add_node_set(name='all', node_indices=np.arange(total_nodes))

    return part


def create_cylinder(
    x: float, y: float, z: float,
    dx: float, dy: float, dz: float,
    radius: float,
    mesh_size: float = 1.5,
    mesh_size_min: float | None = None,
    element_name: str = 'C3D4',
) -> Part:
    """
    Create a cylinder with gmsh's OCC kernel and generate a tetrahedral (C3D4) mesh.

    The cylinder base center is at (x, y, z), and its axis is defined by
    (dx, dy, dz). The norm of this vector is the cylinder height.

    Args:
        x (float): x-coordinate of the base center.
        y (float): y-coordinate of the base center.
        z (float): z-coordinate of the base center.
        dx (float): x component of the axis vector.
        dy (float): y component of the axis vector.
        dz (float): z component of the axis vector.
        radius (float): Cylinder radius; must be positive.
        mesh_size (float): Target gmsh mesh size (Mesh.MeshSizeMax).
        mesh_size_min (float | None): Minimum gmsh mesh size (Mesh.MeshSizeMin).
            Defaults to mesh_size for a uniform mesh.
        element_name (str): Name used to register the element in part.elems.

    Returns:
                Part: A Part containing C3D4 elements and the following sets:
                        - node sets: 'all', 'nodes_cap0' (base), 'nodes_cap1' (top)
                        - surface sets: 'extern' (entire exterior), 'surface_cap0',
                            'surface_cap1', and 'surface_lateral'

    Note:
        The gmsh tetrahedron node order matches torchfea's C3D4 convention
        (right-handed/positive volume), so no node reordering is required.
    """

    assert radius > 0, "radius must be greater than 0"
    assert mesh_size > 0, "mesh_size must be greater than 0"

    base = np.array([x, y, z], dtype=np.float64)
    ax = np.array([dx, dy, dz], dtype=np.float64)
    length = float(np.linalg.norm(ax))
    assert length > 0, "cylinder axis (dx, dy, dz) must have non-zero length"

    import gmsh  # Import lazily because loading gmsh is relatively slow.

    gmsh.initialize()
    try:
        gmsh.clear()
        gmsh.model.add("cylinder")
        gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, radius, tag=1)
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber(
            "Mesh.MeshSizeMin",
            mesh_size_min if mesh_size_min is not None else mesh_size,
        )
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.model.mesh.generate(3)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()
    finally:
        gmsh.finalize()

    # gmsh node tags are not guaranteed to be contiguous or start at 1.
    node_tags = np.asarray(node_tags, dtype=np.int64)
    coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
    order = np.argsort(node_tags)
    tags_sorted = node_tags[order]
    coords = coords[order]

    def _to_local(global_tags: np.ndarray) -> np.ndarray:
        return np.searchsorted(tags_sorted, np.asarray(global_tags, dtype=np.int64))

    # Extract four-node tetrahedra (type 4); their order matches torchfea C3D4.
    conn = None
    for etype, _etags, ntag in zip(element_types, element_tags, element_node_tags):
        if etype == 4:
            conn = _to_local(np.asarray(ntag, dtype=np.int64).reshape(-1, 4))
            break
    if conn is None:
        raise RuntimeError("No 4-node tetrahedral (type 4) elements found in the gmsh mesh.")

    nodes = torch.tensor(coords, dtype=torch.float64)
    elems = torch.tensor(conn, dtype=torch.long)

    part = Part(nodes=nodes)
    elem = initialize_element(
        element_type='C3D4',
        elems_index=torch.arange(conn.shape[0]),
        elems=elems,
    )
    part.add_element(elem, name=element_name)

    # Exterior boundary: the lateral surface and both end caps.
    boundary = elem.extract_boundary_surface_set()
    part.add_surface_set(name='extern', elements=boundary)

    # Classify boundary triangles as base, top, or lateral faces by axis projection.
    ah = ax / length
    t = (coords - base) @ ah  # Projection of each node onto the cylinder axis.
    coord_scale = max(1.0, float(np.abs(np.concatenate([base, ax, [radius]])).max()))
    tol = 1e-8 * coord_scale

    face_defs = elem.surfaceid_map
    cap0, cap1, lateral = [], [], []
    for arr, sfidx in boundary:
        if arr.size == 0:
            continue
        face_t = t[conn[arr][:, face_defs[sfidx]]]  # (n_face, 3) projected positions.
        is_cap0 = np.all(np.abs(face_t) <= tol, axis=1)
        is_cap1 = np.all(np.abs(face_t - length) <= tol, axis=1)
        if np.any(is_cap0):
            cap0.append((arr[is_cap0], sfidx))
        if np.any(is_cap1):
            cap1.append((arr[is_cap1], sfidx))
        is_lateral = ~(is_cap0 | is_cap1)
        if np.any(is_lateral):
            lateral.append((arr[is_lateral], sfidx))

    if cap0:
        part.add_surface_set(name='surface_cap0', elements=cap0)
    if cap1:
        part.add_surface_set(name='surface_cap1', elements=cap1)
    if lateral:
        part.add_surface_set(name='surface_lateral', elements=lateral)

    # Node sets for the base and top caps, useful for boundary conditions.
    part.add_node_set(name='all', node_indices=np.arange(coords.shape[0]))
    part.add_node_set(name='nodes_cap0', node_indices=np.where(np.abs(t) <= tol)[0])
    part.add_node_set(name='nodes_cap1', node_indices=np.where(np.abs(t - length) <= tol)[0])

    return part
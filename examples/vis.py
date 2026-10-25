import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

def create_figure(title, subplot_size=(1,1)):
    """Create a 3D figure with the given title."""
    fig = plt.figure(figsize=(8.27, 11.69))
    if subplot_size == (1, 1):
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
    else:
        ax = []
        for i in range(subplot_size[0]):
            for j in range(subplot_size[1]):
                ax.append(fig.add_subplot(subplot_size[0], subplot_size[1], i * subplot_size[1] + j + 1, projection='3d'))
                ax[-1].set_title(f"{title} ({i+1},{j+1})", fontsize=16)
                ax[-1].set_xlabel('X', fontsize=12)
                ax[-1].set_ylabel('Y', fontsize=12)
                ax[-1].set_zlabel('Z', fontsize=12)
    return fig, ax

def set_equal_aspect_3d(ax):
    """Set equal aspect ratio for 3D plot."""
    extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in ['x', 'y', 'z']])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    radius = 0.5 * np.max(sz)
    for i in range(3):
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)

def plot_nodes(ax, coords, labels=True, color='blue', marker='o', s=10):
    """Plot nodes with optional labels."""
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color=color, marker=marker, s=s)
    if labels:
        for i, (x, y, z) in enumerate(coords):
            ax.text(x + 0.05, y + 0.05, z + 0.05, f"{i}", color='black', fontsize=10)

def plot_edges(ax, coords, edges, color='gray', linewidth=1):
    """Plot edges between nodes."""
    for edge in edges:
        i, j = edge
        ax.plot([coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                [coords[i, 2], coords[j, 2]], color=color, linewidth=linewidth)

def plot_faces(ax, coords, faces, color='lightblue', alpha=0.3):
    """Plot faces to visualize the element surface."""
    poly3d = [[coords[idx] for idx in face] for face in faces]
    collection = Poly3DCollection(poly3d, alpha=alpha, facecolor=color, edgecolor='gray')
    ax.add_collection3d(collection)

def plot_integration_points(ax, points, weights=None, color='red', marker='x', s=80):
    """Plot integration points with optional size based on weights."""
    if weights is not None:
        # Normalize weights for visualization
        norm_weights = 20 + 50 * (weights - min(weights)) / (max(weights) - min(weights) + 1e-10)
        for (x, y, z), w, nw in zip(points, weights, norm_weights):
            ax.scatter(x, y, z, color=color, marker=marker, s=nw)
            ax.text(x, y, z, f"{w:.4f}", color='darkred', fontsize=6, ha='right')
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, marker=marker, s=s)
        for i, (x, y, z) in enumerate(points):
            ax.text(x, y, z, f"IP{i}", color='darkred', fontsize=6, ha='right')

def plot_face_labels(ax, coords, faces, face_names=None):
    """Plot labels for each face at the face center."""
    for i, face in enumerate(faces):
        # Calculate face center by averaging node coordinates
        face_nodes = np.array([coords[idx] for idx in face])
        face_center = np.mean(face_nodes, axis=0)
        
        # Calculate face normal vector
        if len(face) == 3:  # Triangle face
            v1 = face_nodes[1] - face_nodes[0]
            v2 = face_nodes[2] - face_nodes[0]
            normal = np.cross(v1, v2)
        else:  # Quadrilateral face
            v1 = face_nodes[2] - face_nodes[0]
            v2 = face_nodes[3] - face_nodes[1]
            normal = np.cross(v1, v2)
            
        # Normalize the normal vector
        normal_length = np.linalg.norm(normal)
        if normal_length > 1e-10:  # Avoid division by zero
            normal = normal / normal_length
            
        # Offset face center along normal vector
        offset = 0  # Adjust this value to control the offset distance
        face_center = face_center + offset * normal
        
        # Create the face label
        face_label = f"Face {i}" if face_names is None else f"{face_names[i]}\n(Face {i})"
        
        # Add label at face center
        ax.text(face_center[0], face_center[1], face_center[2], 
                face_label, color='darkblue', fontsize=6, 
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.4, boxstyle='round'),
                zorder=100)  # Higher zorder makes the text appear in front
        # add a line to link the label to the face center
        ax.plot([face_center[0] - offset * normal[0], face_center[0]],
            [face_center[1] - offset * normal[1], face_center[1]],
            [face_center[2] - offset * normal[2], face_center[2]],
            'k-', linewidth=0.8, alpha=1.) 
        ax.scatter(face_center[0] - offset * normal[0],
                    face_center[1] - offset * normal[1],
                    face_center[2] - offset * normal[2],
                    color='black', marker='o', s=5, zorder=100)

# Define elements
def define_tet4():
    """Define a 4-node tetrahedral element (C3D4)."""
    # Natural coordinates (g,h,r)
    natural_coords = np.array([
        [0, 0, 0],  # Node 0
        [1, 0, 0],  # Node 1
        [0, 1, 0],  # Node 2
        [0, 0, 1]   # Node 3
    ])
  
    # Convert to Cartesian for visualization
    cart_coords = natural_coords.copy()
  
    edges = [
        (0, 1), (1, 2), (2, 0),  # Base
        (0, 3), (1, 3), (2, 3)   # Edges to apex
    ]
  
    faces = [
        [0, 2, 1],  # Base
        [0, 1, 3],  # Side face
        [1, 2, 3],  # Side face
        [0, 3, 2]   # Side face
    ]
  
    # Integration point (natural coordinates)
    int_points = np.array([[0.25, 0.25, 0.25]])
    int_weights = np.array([1/6])
  
    return {
        'name': 'C3D4 (4-node Tetrahedron)',
        'natural_coords': natural_coords,
        'cart_coords': cart_coords,
        'edges': edges,
        'faces': faces,
        'int_points': int_points,
        'int_weights': int_weights
    }

def define_wedge6():
    """Define a 6-node wedge (prism) element (C3D6)."""
    # Natural coordinates (g,h,r)
    natural_coords = np.array([
        [0, 0, -1],  # Node 0
        [1, 0, -1],  # Node 1
        [0, 1, -1],  # Node 2
        [0, 0, 1],   # Node 3
        [1, 0, 1],   # Node 4
        [0, 1, 1]    # Node 5
    ])
  
    # Convert to Cartesian for better visualization
    cart_coords = np.zeros_like(natural_coords)
    cart_coords[:, 0] = natural_coords[:, 0]
    cart_coords[:, 1] = natural_coords[:, 1]
    cart_coords[:, 2] = natural_coords[:, 2]
  
    edges = [
        (0, 1), (1, 2), (2, 0),  # Bottom triangle
        (3, 4), (4, 5), (5, 3),  # Top triangle
        (0, 3), (1, 4), (2, 5)   # Connecting edges
    ]
  
    faces = [
        [0, 2, 1],      # Bottom face
        [3, 4, 5],      # Top face
        [0, 1, 4, 3],   # Side face
        [1, 2, 5, 4],   # Side face
        [2, 0, 3, 5]    # Side face
    ]
  
    # 6-point integration scheme (approximate values)
    g_coords = np.array([1/6, 2/3, 1/6, 1/6, 2/3, 1/6])
    h_coords = np.array([1/6, 1/6, 2/3, 1/6, 1/6, 2/3])
    r_coords = np.array([-np.sqrt(1/3), -np.sqrt(1/3), -np.sqrt(1/3), 
                         np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)])
  
    int_points = np.column_stack((g_coords, h_coords, r_coords))
    int_weights = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])  # Approximate
  
    return {
        'name': 'C3D6 (6-node Wedge)',
        'natural_coords': natural_coords,
        'cart_coords': cart_coords,
        'edges': edges,
        'faces': faces,
        'int_points': int_points,
        'int_weights': int_weights
    }

def define_hex8():
    """Define an 8-node hexahedral (brick) element (C3D8)."""
    # Natural coordinates (g,h,r)
    natural_coords = np.array([
        [-1, -1, -1],  # Node 0
        [1, -1, -1],   # Node 1
        [1, 1, -1],    # Node 2
        [-1, 1, -1],   # Node 3
        [-1, -1, 1],   # Node 4
        [1, -1, 1],    # Node 5
        [1, 1, 1],     # Node 6
        [-1, 1, 1]     # Node 7
    ])
  
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
  
    faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Front face
        [1, 2, 6, 5],   # Right face
        [3, 2, 6, 7],  # Back face
        [0, 3, 7, 4],  # Left face
    ]
  
    # 8 integration points for full integration (2x2x2)
    gauss_points = np.array([
        [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)],
        [1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)],
        [1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)],
        [-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)],
        [-1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)],
        [1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)],
        [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
        [-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
    ])
  
    int_points_full = gauss_points
    int_weights_full = np.ones(8)  # Each point has weight 1
  
    # 1 integration point for reduced integration (C3D8R)
    int_point_reduced = np.array([[0, 0, 0]])
    int_weight_reduced = np.array([8])  # Weight is 8
  
    return {
        'name': 'C3D8/C3D8R (8-node Hex)',
        'natural_coords': natural_coords,
        'cart_coords': natural_coords,  # Same for this element
        'edges': edges,
        'faces': faces,
        'int_points_full': int_points_full,
        'int_weights_full': int_weights_full,
        'int_point_reduced': int_point_reduced,
        'int_weight_reduced': int_weight_reduced
    }

def define_tet10():
    """Define a 10-node tetrahedral element (C3D10)."""
    # Natural coordinates (g,h,r)
    natural_coords = np.array([
        [0, 0, 0],      # Node 0 (vertex)
        [1, 0, 0],      # Node 1 (vertex)
        [0, 1, 0],      # Node 2 (vertex)
        [0, 0, 1],      # Node 3 (vertex)
        [0.5, 0, 0],    # Node 4 (mid-edge 0-1)
        [0.5, 0.5, 0],  # Node 5 (mid-edge 1-2)
        [0, 0.5, 0],    # Node 6 (mid-edge 2-0)
        [0, 0, 0.5],    # Node 7 (mid-edge 0-3)
        [0.5, 0, 0.5],  # Node 8 (mid-edge 1-3)
        [0, 0.5, 0.5]   # Node 9 (mid-edge 2-3)
    ])
  
    # Use same coordinates for Cartesian visualization
    cart_coords = natural_coords.copy()
  
    edges = [
        (0, 4), (4, 1),  # Edge 0-1 with midpoint
        (1, 5), (5, 2),  # Edge 1-2 with midpoint
        (2, 6), (6, 0),  # Edge 2-0 with midpoint
        (0, 7), (7, 3),  # Edge 0-3 with midpoint
        (1, 8), (8, 3),  # Edge 1-3 with midpoint
        (2, 9), (9, 3)   # Edge 2-3 with midpoint
    ]
  
    faces = [
        [0, 6, 2, 5, 1, 4],     # Base face with midpoints
        [0, 4, 1, 8, 3, 7],     # Side face with midpoints
        [1, 5, 2, 9, 3, 8],     # Side face with midpoints
        [0, 7, 3, 9, 2, 6]      # Side face with midpoints
    ]
  
    # 4 integration points (approximate)
    a = 0.58541020
    b = 0.13819660
    int_points = np.array([
        [a, b, b],
        [b, a, b],
        [b, b, a],
        [b, b, b]
    ])
    int_weights = np.array([1/24, 1/24, 1/24, 1/24])
  
    return {
        'name': 'C3D10 (10-node Tetrahedron)',
        'natural_coords': natural_coords,
        'cart_coords': cart_coords,
        'edges': edges,
        'faces': faces,
        'int_points': int_points,
        'int_weights': int_weights
    }

def define_wedge15():
    """Define a 15-node wedge element (C3D15)."""
    # Natural coordinates
    natural_coords = np.array([
        [0, 0, -1],      # Node 0 (vertex)
        [1, 0, -1],      # Node 1 (vertex)
        [0, 1, -1],      # Node 2 (vertex)
        [0, 0, 1],       # Node 3 (vertex)
        [1, 0, 1],       # Node 4 (vertex)
        [0, 1, 1],       # Node 5 (vertex)
        [0.5, 0, -1],    # Node 6 (mid-edge base)
        [0.5, 0.5, -1],  # Node 7 (mid-edge base)
        [0, 0.5, -1],    # Node 8 (mid-edge base)
        [0.5, 0, 1],     # Node 9 (mid-edge top)
        [0.5, 0.5, 1],   # Node 10 (mid-edge top)
        [0, 0.5, 1],     # Node 11 (mid-edge top)
        [0, 0, 0],       # Node 12 (mid-edge height)
        [1, 0, 0],       # Node 13 (mid-edge height)
        [0, 1, 0]        # Node 14 (mid-edge height)
    ])
  
    # Convert to Cartesian for better visualization
    cart_coords = natural_coords.copy()
  
    edges = [
        (0, 6), (6, 1),     # Edge 0-1 with midpoint
        (1, 7), (7, 2),     # Edge 1-2 with midpoint
        (2, 8), (8, 0),     # Edge 2-0 with midpoint
        (3, 9), (9, 4),     # Edge 3-4 with midpoint
        (4, 10), (10, 5),   # Edge 4-5 with midpoint
        (5, 11), (11, 3),   # Edge 5-3 with midpoint
        (0, 12), (12, 3),   # Edge 0-3 with midpoint
        (1, 13), (13, 4),   # Edge 1-4 with midpoint
        (2, 14), (14, 5)    # Edge 2-5 with midpoint
    ]
  
    faces = [
        [0, 8, 2, 7, 1, 6],      # Bottom face with midpoints
        [3, 9, 4, 10, 5, 11],    # Top face with midpoints
        [0, 6, 1, 13, 4, 9, 3, 12],   # Side face with midpoints
        [1, 7, 2, 14, 5, 10, 4, 13],  # Side face with midpoints
        [2, 8, 0, 12, 3, 11, 5, 14]   # Side face with midpoints
    ]
  
    # 9-point integration scheme (approximate)
    # 3 points on triangle base, 3 heights
    triangle_base = np.array([
        [1/6, 1/6],
        [2/3, 1/6],
        [1/6, 2/3]
    ])
  
    heights = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    weights_1d = np.array([5/9, 8/9, 5/9])
  
    int_points = []
    int_weights = []
  
    for i, (g, h) in enumerate(triangle_base):
        for j, r in enumerate(heights):
            int_points.append([g, h, r])
            int_weights.append((1/6) * weights_1d[j])  # Normalize by triangle area (1/2)
  
    int_points = np.array(int_points)
    int_weights = np.array(int_weights)
  
    return {
        'name': 'C3D15 (15-node Wedge)',
        'natural_coords': natural_coords,
        'cart_coords': cart_coords,
        'edges': edges,
        'faces': faces,
        'int_points': int_points,
        'int_weights': int_weights
    }

def define_hex20():
    """Define a 20-node hexahedral element (C3D20/C3D20R)."""
    # Natural coordinates (g,h,r)
    natural_coords = np.array([
        [-1, -1, -1],  # Node 0 (corner)
        [1, -1, -1],   # Node 1 (corner)
        [1, 1, -1],    # Node 2 (corner)
        [-1, 1, -1],   # Node 3 (corner)
        [-1, -1, 1],   # Node 4 (corner)
        [1, -1, 1],    # Node 5 (corner)
        [1, 1, 1],     # Node 6 (corner)
        [-1, 1, 1],    # Node 7 (corner)
        [0, -1, -1],   # Node 8 (mid-edge bottom)
        [1, 0, -1],    # Node 9 (mid-edge bottom)
        [0, 1, -1],    # Node 10 (mid-edge bottom)
        [-1, 0, -1],   # Node 11 (mid-edge bottom)
        [0, -1, 1],    # Node 12 (mid-edge top)
        [1, 0, 1],     # Node 13 (mid-edge top)
        [0, 1, 1],     # Node 14 (mid-edge top)
        [-1, 0, 1],    # Node 15 (mid-edge top)
        [-1, -1, 0],   # Node 16 (mid-edge side)
        [1, -1, 0],    # Node 17 (mid-edge side)
        [1, 1, 0],     # Node 18 (mid-edge side)
        [-1, 1, 0]     # Node 19 (mid-edge side)
    ])
  
    edges = []
    # Bottom face edges with midpoints
    edges.extend([(0, 8), (8, 1), (1, 9), (9, 2), (2, 10), (10, 3), (3, 11), (11, 0)])
    # Top face edges with midpoints
    edges.extend([(4, 12), (12, 5), (5, 13), (13, 6), (6, 14), (14, 7), (7, 15), (15, 4)])
    # Vertical edges with midpoints
    edges.extend([(0, 16), (16, 4), (1, 17), (17, 5), (2, 18), (18, 6), (3, 19), (19, 7)])
  
    faces = [
        [0, 8, 1, 9, 2, 10, 3, 11],    # Bottom face
        [4, 12, 5, 13, 6, 14, 7, 15],  # Top face
        [0, 8, 1, 17, 5, 12, 4, 16],   # Front face
        [1, 9, 2, 18, 6, 13, 5, 17],    # Right face
        [3, 10, 2, 18, 6, 14, 7, 19],  # Back face
        [0, 11, 3, 19, 7, 15, 4, 16],  # Left face
    ]
  
    # 27 integration points for full integration (3x3x3)
    gauss_points_1d = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    weights_1d = np.array([5/9, 8/9, 5/9])
  
    int_points_full = []
    int_weights_full = []
  
    for i, g in enumerate(gauss_points_1d):
        for j, h in enumerate(gauss_points_1d):
            for k, r in enumerate(gauss_points_1d):
                int_points_full.append([g, h, r])
                int_weights_full.append(weights_1d[i] * weights_1d[j] * weights_1d[k])
  
    int_points_full = np.array(int_points_full)
    int_weights_full = np.array(int_weights_full)
  
    # 8 integration points for reduced integration (2x2x2)
    reduced_points_1d = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
  
    int_points_reduced = []
  
    for i in reduced_points_1d:
        for j in reduced_points_1d:
            for k in reduced_points_1d:
                int_points_reduced.append([i, j, k])
  
    int_points_reduced = np.array(int_points_reduced)
    int_weights_reduced = np.ones(8)
  
    return {
        'name': 'C3D20/C3D20R (20-node Hex)',
        'natural_coords': natural_coords,
        'cart_coords': natural_coords,  # Same for this element
        'edges': edges,
        'faces': faces,
        'int_points_full': int_points_full,
        'int_weights_full': int_weights_full,
        'int_points_reduced': int_points_reduced,
        'int_weights_reduced': int_weights_reduced
    }

def visualize_nodes():
    """Visualize node arrangements for first and second order elements."""
    # First order elements
    first_order_elements = [define_tet4(), define_wedge6(), define_hex8()]
    
    # Define face names for each element type
    tet4_face_names = ["Base", "Side 1", "Side 2", "Side 3"]
    wedge6_face_names = ["Bottom", "Top", "Side 1", "Side 2", "Side 3"]
    hex8_face_names = ["Bottom (Z-)", "Top (Z+)", "Front (Y-)", "Right (X+)", "Back (Y+)", "Left (X-)"]
    first_order_face_names = [tet4_face_names, wedge6_face_names, hex8_face_names]
    
    fig, ax_total = create_figure(f"Node Numbering", subplot_size=(3, 2))
    for i, element in enumerate(first_order_elements):
        coords = element['cart_coords']
        ax = ax_total[2*i]
        plot_nodes(ax, coords)
        plot_edges(ax, coords, element['edges'])
        plot_faces(ax, coords, element['faces'])
        plot_face_labels(ax, coords, element['faces'], first_order_face_names[i])
        # set_equal_aspect_3d(ax)
        plt.tight_layout()
        
  
    # Second order elements
    second_order_elements = [define_tet10(), define_wedge15(), define_hex20()]
    
    # Same face names for second order elements
    second_order_face_names = [tet4_face_names, wedge6_face_names, hex8_face_names]
  
    for i, element in enumerate(second_order_elements):
        coords = element['cart_coords']
        ax = ax_total[2*i+1]
        plot_nodes(ax, coords)
        plot_edges(ax, coords, element['edges'])
        plot_faces(ax, coords, element['faces'])
        plot_face_labels(ax, coords, element['faces'], second_order_face_names[i])
        # set_equal_aspect_3d(ax)
        plt.tight_layout()

    plt.savefig(f"nodes_orders.png", dpi=300, bbox_inches='tight')
      
    plt.close('all')

def visualize_integration_points():
    """Visualize integration points for first and second order elements."""
    # First order elements
    elements = [define_tet4(), define_wedge6(), define_hex8()]
    
    # Define face names for each element type
    tet4_face_names = ["Base", "Side 1", "Side 2", "Side 3"]
    wedge6_face_names = ["Bottom", "Top", "Side 1", "Side 2", "Side 3"]
    hex8_face_names = ["Bottom (Z-)", "Top (Z+)", "Front (Y-)", "Back (Y+)", "Left (X-)", "Right (X+)"]
    first_order_face_names = [tet4_face_names, wedge6_face_names, hex8_face_names]
    
    fig, ax_set = create_figure(f"Integration Points", (3, 2))
    for i, element in enumerate(elements):
        
        coords = element['cart_coords']
        ax = ax_set[2*i]
        # Plot transparent element
        plot_nodes(ax, coords, labels=False, color='lightgray', s=50)
        plot_edges(ax, coords, element['edges'], color='lightgray')
        plot_faces(ax, coords, element['faces'], color='lightgray', alpha=0.1)
        # plot_face_labels(ax, coords, element['faces'], first_order_face_names[i])

        # Plot integration points
        if 'int_points' in element and 'int_weights' in element:
            plot_integration_points(ax, element['int_points'], element['int_weights'])
        elif 'int_points_full' in element:
            # For hex8, show both full and reduced integration
            # ax.text2D(0.05, 0.95, "Full Integration (8 points)", transform=ax.transAxes, color='red')
            plot_integration_points(ax, element['int_points_full'], element['int_weights_full'])
          
            # # Create second plot for reduced integration
            # fig2, ax2 = create_figure(f"First Order Element: {element['name']} Reduced Integration")
            # plot_nodes(ax2, coords, labels=False, color='lightgray', s=50)
            # plot_edges(ax2, coords, element['edges'], color='lightgray')
            # plot_faces(ax2, coords, element['faces'], color='lightgray', alpha=0.1)
            # # plot_face_labels(ax2, coords, element['faces'], first_order_face_names[i])
            # ax2.text2D(0.05, 0.95, "Reduced Integration (1 point)", transform=ax2.transAxes, color='red')
            # plot_integration_points(ax2, element['int_point_reduced'], element['int_weight_reduced'])
            # set_equal_aspect_3d(ax2)
            # plt.tight_layout()
            # plt.savefig(f"first_order_int_points_{i}_reduced.png", dpi=300, bbox_inches='tight')
      
        set_equal_aspect_3d(ax)
        plt.tight_layout()

    # Second order elements
    elements = [define_tet10(), define_wedge15(), define_hex20()]
    
    # Same face names for second order elements
    second_order_face_names = [tet4_face_names, wedge6_face_names, hex8_face_names]
  
    for i, element in enumerate(elements):

        coords = element['cart_coords']
        ax = ax_set[2*i+1]
        # Plot transparent element
        plot_nodes(ax, coords, labels=False, color='lightgray', s=50)
        plot_edges(ax, coords, element['edges'], color='lightgray')
        plot_faces(ax, coords, element['faces'], color='lightgray', alpha=0.1)
        # plot_face_labels(ax, coords, element['faces'], second_order_face_names[i])
      
        # Plot integration points
        if 'int_points' in element and 'int_weights' in element:
            plot_integration_points(ax, element['int_points'], element['int_weights'])
        elif 'int_points_full' in element:
            # For hex20, show both full and reduced integration
            # ax.text2D(0.05, 0.95, "Full Integration (27 points)", transform=ax.transAxes, color='red')
            plot_integration_points(ax, element['int_points_full'], element['int_weights_full'])
          
            # # Create second plot for reduced integration
            # fig2, ax2 = create_figure(f"Second Order Element: {element['name']} Reduced Integration")
            # plot_nodes(ax2, coords, labels=False, color='lightgray', s=50)
            # plot_edges(ax2, coords, element['edges'], color='lightgray')
            # plot_faces(ax2, coords, element['faces'], color='lightgray', alpha=0.1)
            # # plot_face_labels(ax2, coords, element['faces'], second_order_face_names[i])
            # ax2.text2D(0.05, 0.95, "Reduced Integration (8 points)", transform=ax2.transAxes, color='red')
            # plot_integration_points(ax2, element['int_points_reduced'], element['int_weights_reduced'])
            # set_equal_aspect_3d(ax2)
            # plt.tight_layout()
            # plt.savefig(f"second_order_int_points_{i}_reduced.png", dpi=300, bbox_inches='tight')
      
        set_equal_aspect_3d(ax)
        plt.tight_layout()
    plt.savefig(f"int_points_{i}.png", dpi=300, bbox_inches='tight')
      
    plt.close('all')

def main():
    print("Generating visualizations for torch_fea elements...")
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists("element_visualizations"):
        os.makedirs("element_visualizations")
  
    # Change working directory to output folder
    os.chdir("element_visualizations")
  
    # Visualize nodes and integration points
    visualize_nodes()
    visualize_integration_points()
  
    print("Visualizations completed! Check the 'element_visualizations' folder.")

if __name__ == "__main__":
    main()

import numpy as np
import time
class Tomb:
    def __init__(self, id, lat, lon, elevation):
        self.id = id
        self.pos = np.array([lat, lon])  # Position vector (Lat/Lon)
        self.ele = elevation             # Z value
        self.result = None               # Store the calculation result here\


class Node:
    def __init__(self, center, size):
        self.center = center  # Center of this quadrant (Lat/Lon)
        self.size = size  # Width of the quadrant (degrees)
        self.children = None  # 4 children nodes (NW, NE, SW, SE)
        self.tombs = []  # Tombs contained in this specific node

        # Aggregate Statistics (Virtual Tomb)
        self.centroid = np.zeros(2)  # Geometric Centroid
        self.min_z = float('inf')  # Lowest elevation in this cluster
        self.max_z = float('-inf')  # Highest elevation in this cluster
        self.count = 0  # Number of tombs in this subtree

    def is_leaf(self):
        return self.children is None

    def compute_statistics(self):
        """
        Step 3: Bottom-Up Statistics [3]
        Recursively calculates the centroid and elevation range for this node
        based on its children or the tombs it holds.
        """
        # Base case: Leaf node with tombs
        if self.is_leaf():
            if not self.tombs: return

            # Simple average for centroid, min/max for elevation
            positions = np.array([t.pos for t in self.tombs])
            elevations = np.array([t.ele for t in self.tombs])

            self.count = len(self.tombs)
            self.centroid = np.mean(positions, axis=0)
            self.min_z = np.min(elevations)
            self.max_z = np.max(elevations)

        # Recursive step: Internal node
        else:
            total_pos = np.zeros(2)
            z_min_list = []
            z_max_list = []

            for child in self.children:
                child.compute_statistics()  # Recurse down
                if child.count > 0:
                    self.count += child.count
                    total_pos += child.centroid * child.count  # Weighted sum
                    z_min_list.append(child.min_z)
                    z_max_list.append(child.max_z)

            if self.count > 0:
                self.centroid = total_pos / self.count
                self.min_z = min(z_min_list)
                self.max_z = max(z_max_list)


def build_quadtree(tombs, center, size):
    node = Node(center, size)

    # Stop condition: If few tombs, keep as leaf (Bucket size)
    # You can tune this (e.g., stop if <= 1 tomb)
    if len(tombs) <= 1:
        node.tombs = tombs
        return node

    # Split into 4 quadrants
    half_size = size / 2
    step = size / 4

    # Define centers for NW, NE, SW, SE
    offsets = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
    children = []

    for dx, dy in offsets:
        child_center = node.center + np.array([dx * step, dy * step])

        # Filter tombs belonging to this quadrant
        # Corrected filtering: check if tomb falls within child quadrant boundaries
        child_tombs = [
            t for t in tombs
            if abs(t.pos[0] - child_center[0]) <= step
               and abs(t.pos[1] - child_center[1]) <= step
        ]

        children.append(build_quadtree(child_tombs, child_center, half_size))

    node.children = children
    return node


def solve_dataset(root_node, target_point, theta=0.5, epsilon=20):
    """
    Step 4: The Solver (Top-Down) [3]
    """

    # 1. Distance Check (s/d < theta)
    # Calculate distance from target to this node's centroid
    dist = np.linalg.norm(root_node.centroid - target_point)
    if dist == 0: dist = 0.0001  # Avoid division by zero

    is_far_enough = (root_node.size / dist) < theta

    # 2. Topographic Check (Z_max - Z_min < epsilon)
    elevation_diff = root_node.max_z - root_node.min_z
    is_flat_enough = elevation_diff < epsilon

    # DECISION LOGIC [7, 8]

    # CASE A: It's a single tomb (Leaf) -> Must calculate exact
    if root_node.is_leaf():
        for tomb in root_node.tombs:
            # RUN YOUR HEAVY SCRIPT HERE
            tomb.result = run_heavy_script(tomb.pos, tomb.ele)
            print(f"Calculated exact for Tomb {tomb.id}")

    # CASE B: Cluster is valid -> Approximation
    elif is_far_enough and is_flat_enough:
        # Run heavy script ONCE on the virtual centroid
        cluster_result = run_heavy_script(root_node.centroid, root_node.min_z)  # or avg z

        # Apply result to all children (Virtual Tomb)
        apply_to_all(root_node, cluster_result)
        print(f"Approximated {root_node.count} tombs at distance {dist:.2f}")

    # CASE C: Cluster invalid -> Dig Deeper
    else:
        for child in root_node.children:
            if child.count > 0:
                solve_dataset(child, target_point, theta, epsilon)


def apply_to_all(node, result):
    """Helper to propagate results down to all tombs in a cluster"""
    if node.is_leaf():
        for tomb in node.tombs:
            tomb.result = result
    else:
        for child in node.children:
            apply_to_all(child, result)


# Mock version of your 3-second script
def run_heavy_script(pos, ele):
    return f"Horizon_Angle_at_{pos}"


if __name__ == "__main__":
    # 1. Generate Mock Data (30 random tombs in a small area)
    # In reality, you'd load your CSV or Shapefile here

    start = time.time()
    num_tombs = 100
    np.random.seed(42)

    mock_tombs = [
        Tomb(
            id=i,
            lat=np.random.uniform(30.0, 30.1),
            lon=np.random.uniform(31.0, 31.1),
            elevation=np.random.uniform(10, 100)
        ) for i in range(num_tombs)
    ]

    # 2. Determine Tree Parameters
    # We find the bounding box of all tombs to center the Quadtree
    all_pos = np.array([t.pos for t in mock_tombs])
    min_coords = np.min(all_pos, axis=0)
    max_coords = np.max(all_pos, axis=0)

    center = (min_coords + max_coords) / 2
    # Size is the maximum span of the data
    size = np.max(max_coords - min_coords) + 0.01

    # 3. Build the Tree
    print("--- Building Quadtree ---")
    # Note: I've updated the logic slightly to ensure we don't miss tombs
    root = build_quadtree(mock_tombs, center, size)

    # 4. Compute Statistics (Bottom-Up)
    print("--- Computing Cluster Statistics ---")
    root.compute_statistics()

    # 5. Run the Solver
    # Let's say we are looking at the horizon from this specific viewpoint:
    viewpoint = np.array([30.05, 31.05])

    print(f"--- Running Solver (Total Tombs: {num_tombs}) ---")
    # theta=0.5: higher means more approximation
    # epsilon=20: max elevation difference allowed to treat a cluster as "flat"
    solve_dataset(root, target_point=viewpoint, theta=0.5, epsilon=20)

    # 6. Verify Results
    print("\n--- Final Results Check ---")
    for i, t in enumerate(mock_tombs[:5]):  # Show first 5
        print(f"Tomb {t.id} (Ele: {t.ele:.1f}): {t.result}")

    print(f"Done in {time.time() - start}s")
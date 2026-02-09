import numpy as np
import time


class Node:
    def __init__(self, x, y, px, py, m, size):
        self.m = m
        self.pos = np.array([float(x), float(y)])
        self.mom = np.array([float(px), float(py)])
        self.child = None
        self.s = size

    def dist(self, other):
        return np.linalg.norm(self.pos - other.pos)

    def force_applied(self, other, G):
        d = self.dist(other)
        epsilon = 1e-3  # Softening factor
        if d < epsilon:
            return np.zeros(2)
        return (self.pos - other.pos) * (self.m * other.m / (d ** 3 + epsilon)) * G


def get_quadrant_index(body, center):
    west = body.pos[0] < center[0]
    south = body.pos[1] < center[1]
    if west and south: return 0
    if not west and south: return 1
    if west and not south: return 2
    return 3


def get_quadrant_center_and_size(center, size, idx):
    new_size = size / 2.0
    offset = size / 4.0
    signs = [np.array([-1, -1]), np.array([1, -1]), np.array([-1, 1]), np.array([1, 1])]
    new_center = center + (signs[idx] * offset)
    return new_center, new_size


def add_body(body, node, center, size, depth=0):
    # Prevent infinite recursion if points are identical
    if depth > 50:
        return node

    if node is None:
        return Node(body.pos[0], body.pos[1], body.mom[0], body.mom[1], body.m, size)

    if node.child is not None:
        total_mass = node.m + body.m
        node.pos = (node.pos * node.m + body.pos * body.m) / total_mass
        node.m = total_mass
        idx = get_quadrant_index(body, center)
        c, s = get_quadrant_center_and_size(center, size, idx)
        node.child[idx] = add_body(body, node.child[idx], c, s, depth + 1)
        return node
    else:
        new_node = Node(0, 0, 0, 0, 0, size)
        new_node.child = [None] * 4

        idx_old = get_quadrant_index(node, center)
        c_old, s_old = get_quadrant_center_and_size(center, size, idx_old)
        new_node.child[idx_old] = add_body(node, None, c_old, s_old, depth + 1)

        idx_new = get_quadrant_index(body, center)
        c_new, s_new = get_quadrant_center_and_size(center, size, idx_new)
        new_node.child[idx_new] = add_body(body, new_node.child[idx_new], c_new, s_new, depth + 1)

        total_m = node.m + body.m
        new_node.pos = (node.pos * node.m + body.pos * body.m) / total_m
        new_node.m = total_m
        return new_node


def calculate_force(body, node, theta, G):
    if node is None or node is body:
        return np.zeros(2)
    if node.child is None:
        return node.force_applied(body, G)
    d = node.dist(body)
    if d > 0 and (node.s / d < theta):
        return node.force_applied(body, G)
    return sum((calculate_force(body, c, theta, G) for c in node.child if c is not None), np.zeros(2))


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    N_BODIES = 100
    N_STEPS = 50
    THETA = 0.5
    G = 1.0
    DT = 0.01
    LIMITS = 10.0
    CENTER = np.array([5.0, 5.0])

    bodies = [Node(np.random.uniform(2, 8), np.random.uniform(2, 8),
                   np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1),
                   np.random.uniform(0.5, 2.0), 0) for _ in range(N_BODIES)]

    print(f"Starting simulation with {N_BODIES} bodies...")
    start_total = time.time()

    for t in range(N_STEPS):
        step_start = time.time()

        # 1. Build Tree
        root = None
        for b in bodies:
            root = add_body(b, root, CENTER, LIMITS)

        # 2. Physics Update
        for b in bodies:
            force = calculate_force(b, root, THETA, G)
            b.mom += force * DT
            b.pos += (b.mom / b.m) * DT

        step_end = time.time()
        if t % 10 == 0:
            print(f"Step {t} | Time: {step_end - step_start:.4f}s")

    end_total = time.time()
    print("-" * 30)
    print(f"Total Simulation Time: {end_total - start_total:.4f} seconds")
    print(f"Average time per step: {(end_total - start_total) / N_STEPS:.4f} seconds")
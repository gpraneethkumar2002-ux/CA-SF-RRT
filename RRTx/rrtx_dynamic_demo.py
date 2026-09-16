import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as geom

from rrtx_core_final import RRT_X, RRT_Node, VehicleGeometry


# ============================================================
# CONFIGURATION
# ============================================================

START = (1.0, 1.0)
GOAL = (9.0, 9.0)

X_BOUNDS = (0.0, 10.0)
Y_BOUNDS = (0.0, 10.0)

# Number of samples used to construct the initial RRTX graph
NUM_ITERATIONS = 500

# ------------------------------------------------------------
# Static obstacle
#
# We keep one dummy static obstacle far outside the workspace.
# The current RRTX implementation expects at least one static
# obstacle because _get_obst_clearance() uses STRtree.nearest().
# ------------------------------------------------------------
static_obstacles = [
    geom.box(50.0, 50.0, 51.0, 51.0)
]

# Large boundary so that it does not interfere with the demo
lanelet_boundary = geom.box(
    -100.0,
    -100.0,
    100.0,
    100.0
)

# Vehicle parameters
vehicle = VehicleGeometry(
    w_half=0.5,
    lf=1.0,
    lr=1.0,
    bumpers_length=(0.1, 0.1)
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_path(rrt):
    """
    Extract the current path from the robot/start node
    to the goal by following p_T_out.
    """

    if rrt.v_bot is None:
        return None

    path_nodes = []

    current = rrt.v_bot
    visited = set()

    while current is not None:

        # Prevent accidental infinite loops
        if id(current) in visited:
            print("ERROR: Loop detected while extracting path.")
            return None

        visited.add(id(current))
        path_nodes.append(current)

        if current is rrt.goal_node:
            break

        current = current.p_T_out

    # Check whether goal was reached
    if len(path_nodes) == 0:
        return None

    if path_nodes[-1] is not rrt.goal_node:
        return None

    return path_nodes


def path_length(path_nodes):
    """
    Calculate total geometric path length.
    """

    if path_nodes is None or len(path_nodes) < 2:
        return np.inf

    total = 0.0

    for i in range(len(path_nodes) - 1):

        x1 = path_nodes[i].x
        y1 = path_nodes[i].y

        x2 = path_nodes[i + 1].x
        y2 = path_nodes[i + 1].y

        total += np.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

    return total


def print_path(title, path_nodes):
    """
    Print the path nodes and their RRTX values.
    """

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    if path_nodes is None:
        print("NO VALID PATH")
        return

    for i, node in enumerate(path_nodes):

        print(
            f"{i:3d}: "
            f"({node.x:7.2f}, {node.y:7.2f}) "
            f"g={node.g:10.3f} "
            f"lmc={node.lmc:10.3f}"
        )

    print("-" * 60)
    print(f"Number of path nodes : {len(path_nodes)}")
    print(f"Path length          : {path_length(path_nodes):.4f}")


def plot_result(
    rrt,
    initial_path,
    repaired_path,
    dynamic_obstacle,
    title
):
    """
    Visualize the RRTX graph, initial path,
    dynamic obstacle and repaired path.
    """

    plt.figure(figsize=(9, 9))

    # --------------------------------------------------------
    # Plot all RRTX nodes
    # --------------------------------------------------------

    xs = []
    ys = []

    for node in rrt.all_nodes:

        xs.append(node.x)
        ys.append(node.y)

    if len(xs) > 0:
        plt.scatter(
            xs,
            ys,
            s=8,
            alpha=0.35,
            label="RRTX nodes"
        )

    # --------------------------------------------------------
    # Plot initial path
    # --------------------------------------------------------

    if initial_path is not None:

        ix = [node.x for node in initial_path]
        iy = [node.y for node in initial_path]

        plt.plot(
            ix,
            iy,
            linewidth=3,
            label="Initial path"
        )

    # --------------------------------------------------------
    # Plot repaired path
    # --------------------------------------------------------

    if repaired_path is not None:

        rx = [node.x for node in repaired_path]
        ry = [node.y for node in repaired_path]

        plt.plot(
            rx,
            ry,
            linewidth=3,
            linestyle="--",
            label="Repaired path"
        )

    # --------------------------------------------------------
    # Plot dynamic obstacle
    # --------------------------------------------------------

    if dynamic_obstacle is not None:

        ox, oy = dynamic_obstacle.exterior.xy

        plt.fill(
            ox,
            oy,
            alpha=0.5,
            label="Dynamic obstacle"
        )

    # --------------------------------------------------------
    # Start and goal
    # --------------------------------------------------------

    plt.scatter(
        START[0],
        START[1],
        s=120,
        marker="o",
        label="START"
    )

    plt.scatter(
        GOAL[0],
        GOAL[1],
        s=120,
        marker="*",
        label="GOAL"
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    plt.xlim(X_BOUNDS)
    plt.ylim(Y_BOUNDS)

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title(title)

    plt.grid(True)
    plt.legend()

    plt.gca().set_aspect("equal", adjustable="box")

    plt.show()


# ============================================================
# CREATE RRTX
# ============================================================

print()
print("=" * 60)
print("              RRTX DYNAMIC OBSTACLE DEMO")
print("=" * 60)

goal_node = RRT_Node(
    GOAL[0],
    GOAL[1]
)

rrt = RRT_X(
    v_goal=goal_node,
    static_obs=static_obstacles,
    lanelet_poly_exterior=lanelet_boundary,
    x_bounds=X_BOUNDS,
    y_bounds=Y_BOUNDS,
    agent_params=vehicle,
    edge_coll_rad=0.2
)


# ============================================================
# PHASE 1
# BUILD INITIAL RRTX GRAPH
# ============================================================

print()
print("=" * 60)
print("PHASE 1: BUILDING INITIAL RRTX GRAPH")
print("=" * 60)

for i in range(NUM_ITERATIONS):

    rrt.step(
        v_curr=None,
        new_obs=None
    )

print()
print("Initial graph construction complete.")
print(f"Total nodes: {rrt.num_nodes}")


# ============================================================
# PHASE 2
# CONNECT START
# ============================================================

print()
print("=" * 60)
print("PHASE 2: CONNECTING START TO RRTX GRAPH")
print("=" * 60)

rrt.step(
    v_curr=(START[0], START[1], 0.0),
    new_obs=None
)

initial_path = extract_path(rrt)

if initial_path is None:

    print()
    print("ERROR: Could not obtain initial START -> GOAL path.")
    print("Increase NUM_ITERATIONS and try again.")

    raise SystemExit


print_path(
    "INITIAL RRTX PATH",
    initial_path
)


# ============================================================
# PHASE 3
# CREATE DYNAMIC OBSTACLE
# ============================================================

print()
print("=" * 60)
print("PHASE 3: INSERTING DYNAMIC OBSTACLE")
print("=" * 60)


# ------------------------------------------------------------
# Find a point approximately halfway along the initial path.
# This makes the obstacle placement independent of the exact
# random RRTX path generated.
# ------------------------------------------------------------

middle_index = len(initial_path) // 2

middle_node = initial_path[middle_index]

obstacle_x = middle_node.x
obstacle_y = middle_node.y

# Dynamic obstacle size
obstacle_half_size = 0.7

dynamic_obstacle = geom.box(
    obstacle_x - obstacle_half_size,
    obstacle_y - obstacle_half_size,
    obstacle_x + obstacle_half_size,
    obstacle_y + obstacle_half_size
)

print(
    f"Dynamic obstacle inserted around "
    f"({obstacle_x:.2f}, {obstacle_y:.2f})"
)

print(
    f"Obstacle bounds: "
    f"{dynamic_obstacle.bounds}"
)


# ============================================================
# CHECK WHETHER THE INITIAL PATH INTERSECTS OBSTACLE
# ============================================================

initial_line = geom.LineString(
    [(node.x, node.y) for node in initial_path]
)

print()

if initial_line.intersects(dynamic_obstacle):

    print(
        "CONFIRMED: Dynamic obstacle intersects "
        "the initial RRTX path."
    )

else:

    print(
        "WARNING: Dynamic obstacle does not intersect "
        "the initial path."
    )

    print(
        "The obstacle may be too small or the path geometry "
        "may not be sufficiently covered."
    )


# ============================================================
# PHASE 4
# RRTX ENVIRONMENT UPDATE
# ============================================================

print()
print("=" * 60)
print("PHASE 4: RRTX GRAPH REPAIR")
print("=" * 60)

print()
print("Calling RRTX with the new dynamic obstacle...")

rrt.step(
    v_curr=None,
    new_obs=[dynamic_obstacle]
)


# ============================================================
# PHASE 5
# EXTRACT REPAIRED PATH
# ============================================================

print()
print("=" * 60)
print("PHASE 5: EXTRACTING REPAIRED PATH")
print("=" * 60)

repaired_path = extract_path(rrt)

print_path(
    "REPAIRED RRTX PATH",
    repaired_path
)


# ============================================================
# VALIDATE REPAIRED PATH
# ============================================================

print()
print("=" * 60)
print("RRTX DYNAMIC OBSTACLE VALIDATION")
print("=" * 60)


if repaired_path is None:

    print()
    print("RESULT: No repaired path was found.")

else:

    repaired_line = geom.LineString(
        [(node.x, node.y) for node in repaired_path]
    )

    intersects = repaired_line.intersects(
        dynamic_obstacle
    )

    print()
    print(f"Initial path length : {path_length(initial_path):.4f}")
    print(f"Repaired path length: {path_length(repaired_path):.4f}")

    print()

    if intersects:

        print(
            "WARNING: Repaired path still intersects "
            "the dynamic obstacle."
        )

    else:

        print(
            "SUCCESS: Repaired path avoids "
            "the dynamic obstacle."
        )

    if repaired_path != initial_path:

        print(
            "SUCCESS: RRTX generated a different path "
            "after the environment changed."
        )

    else:

        print(
            "WARNING: Path structure did not change."
        )


# ============================================================
# PRINT RRTX INTERNAL STATE
# ============================================================

print()
print("=" * 60)
print("RRTX INTERNAL STATE")
print("=" * 60)

print(f"Total graph nodes       : {rrt.num_nodes}")
print(f"Dynamic obstacles       : {len(rrt.X_obs_d)}")
print(f"Nodes in collision      : {len(rrt.nodes_in_collision)}")
print(f"Orphan nodes            : {len(rrt.orphan_nodes)}")
print(f"Priority queue size     : {len(rrt.Q)}")

if rrt.v_bot is not None:

    print(
        f"Start g-value          : {rrt.v_bot.g}"
    )

    print(
        f"Start lmc-value        : {rrt.v_bot.lmc}"
    )


# ============================================================
# VISUALIZATION
# ============================================================

print()
print("Displaying RRTX dynamic-obstacle experiment...")

plot_result(
    rrt,
    initial_path,
    repaired_path,
    dynamic_obstacle,
    "RRTX: Dynamic Obstacle Insertion and Graph Repair"
)

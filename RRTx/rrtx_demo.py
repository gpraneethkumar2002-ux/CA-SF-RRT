import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as geom

from rrtx_core_final import RRT_X, RRT_Node, VehicleGeometry


# ============================================================
# SIMPLE RRTX BASELINE TEST
# ============================================================

START = (1.0, 1.0)
GOAL = (9.0, 9.0)

X_BOUNDS = (0.0, 10.0)
Y_BOUNDS = (0.0, 10.0)


# ============================================================
# STATIC OBSTACLES
# ============================================================
#
# The current baseline is an obstacle-free environment.
#
# A dummy obstacle is placed far outside the planning region
# because the current RRTX implementation expects at least
# one static obstacle when calculating obstacle clearance.
#

static_obstacles = [
    geom.box(50.0, 50.0, 51.0, 51.0)
]


# ============================================================
# PLANNING BOUNDARY
# ============================================================
#
# The RRTX implementation expects lanelet_poly_exterior
# to represent the boundary of the drivable region.
#
# We make the boundary much larger than the actual sampling
# region so that the vehicle never interacts with it.
#

lanelet_boundary = geom.box(
    -100.0,
    -100.0,
    100.0,
    100.0
)


# ============================================================
# VEHICLE GEOMETRY
# ============================================================

vehicle = VehicleGeometry(
    w_half=0.5,
    lf=1.0,
    lr=1.0,
    bumpers_length=(0.1, 0.1),
)


# ============================================================
# CREATE GOAL NODE
# ============================================================
#
# RRTX is initialized from the GOAL and grows the graph
# outward from the goal.
#

goal_node = RRT_Node(
    GOAL[0],
    GOAL[1]
)


# ============================================================
# CREATE RRTX PLANNER
# ============================================================

print()
print("==============================================")
print("       STANDALONE RRTX BASELINE")
print("==============================================")
print()

print("Creating RRTX planner...")

rrt = RRT_X(
    v_goal=goal_node,
    static_obs=static_obstacles,
    lanelet_poly_exterior=lanelet_boundary,
    x_bounds=X_BOUNDS,
    y_bounds=Y_BOUNDS,
    agent_params=vehicle,
    edge_coll_rad=0.2,
)

print("RRTX planner created successfully.")
print()


# ============================================================
# BUILD GOAL-ROOTED RRTX TREE
# ============================================================

NUM_ITERATIONS = 500

print("Building goal-rooted RRTX tree...")
print()

for i in range(NUM_ITERATIONS):

    rrt.step(
        v_curr=None,
        new_obs=None
    )

    if (i + 1) % 100 == 0:

        print(
            f"Iteration {i + 1}/{NUM_ITERATIONS} | "
            f"Nodes: {rrt.num_nodes}"
        )


# ============================================================
# TREE INFORMATION
# ============================================================

print()
print("==============================================")
print("       TREE CONSTRUCTION COMPLETE")
print("==============================================")
print()

print("Total tree nodes:", rrt.num_nodes)

print(
    "Goal position:",
    f"({goal_node.x:.2f}, {goal_node.y:.2f})"
)

print("Goal g-value:", goal_node.g)
print("Goal lmc-value:", goal_node.lmc)

print()


# ============================================================
# CONNECT START / ROBOT TO THE RRTX TREE
# ============================================================
#
# IMPORTANT:
#
# The original implementation uses _update_robot()
# to introduce the current robot position into the
# goal-rooted RRTX graph.
#
# step(v_curr=...) internally calls _update_robot().
#

print("==============================================")
print("       CONNECTING START TO RRTX TREE")
print("==============================================")
print()

print(
    "Start position:",
    f"({START[0]:.2f}, {START[1]:.2f})"
)

print("Connecting start node...")

rrt.step(
    v_curr=(START[0], START[1], 0.0),
    new_obs=None
)

print()

if rrt.v_bot is not None:

    print("Start node successfully connected.")
    print(
        "Start node:",
        f"({rrt.v_bot.x:.2f}, {rrt.v_bot.y:.2f})"
    )

    print("Start g-value:", rrt.v_bot.g)
    print("Start lmc-value:", rrt.v_bot.lmc)

else:

    print("ERROR: Start node was not created.")


# ============================================================
# EXTRACT START -> GOAL PATH
# ============================================================
#
# RRTX is goal-rooted.
#
# Therefore:
#
#     START
#       |
#       v
#     RRTX NODE
#       |
#       v
#     RRTX NODE
#       |
#       v
#     GOAL
#
# We therefore start from v_bot and follow p_T_out.
#

print()
print("==============================================")
print("       EXTRACTING RRTX PATH")
print("==============================================")
print()

path_nodes = []

current = rrt.v_bot

visited = set()

while current is not None:

    node_id = id(current)

    if node_id in visited:

        print("WARNING: Cycle detected in RRTX tree.")
        break

    visited.add(node_id)

    path_nodes.append(current)

    if current is goal_node:
        break

    current = current.p_T_out


# ============================================================
# PATH INFORMATION
# ============================================================

print()
print("Number of path nodes:", len(path_nodes))
print()

if len(path_nodes) <= 1:

    print("No START -> GOAL path was established.")

else:

    print("RRTX PATH")
    print("----------------------------------------------")

    for i, node in enumerate(path_nodes):

        print(
            f"{i:3d}: "
            f"({node.x:7.2f}, {node.y:7.2f}) "
            f"g={node.g:10.3f} "
            f"lmc={node.lmc:10.3f}"
        )

    print("----------------------------------------------")


# ============================================================
# CHECK WHETHER PATH REACHES GOAL
# ============================================================

reaches_goal = (
    len(path_nodes) > 0
    and path_nodes[-1] is goal_node
)

print()

if reaches_goal:

    print("SUCCESS: START -> GOAL path found.")

else:

    print("WARNING: Path does not reach the goal.")


# ============================================================
# CALCULATE PATH LENGTH
# ============================================================

path_length = 0.0

for i in range(len(path_nodes) - 1):

    x1 = path_nodes[i].x
    y1 = path_nodes[i].y

    x2 = path_nodes[i + 1].x
    y2 = path_nodes[i + 1].y

    path_length += np.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


print()
print("Path length:", path_length)
print()


# ============================================================
# VISUALIZATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 9)
)


# ============================================================
# RRTX TREE
# ============================================================

for node in rrt.all_nodes:

    if node.p_T_out is not None:

        parent = node.p_T_out

        ax.plot(
            [node.x, parent.x],
            [node.y, parent.y],
            linewidth=0.5,
            alpha=0.35
        )


# ============================================================
# START
# ============================================================

ax.scatter(
    START[0],
    START[1],
    s=120,
    marker="o",
    label="Start"
)


# ============================================================
# GOAL
# ============================================================

ax.scatter(
    GOAL[0],
    GOAL[1],
    s=180,
    marker="*",
    label="Goal"
)


# ============================================================
# RRTX PATH
# ============================================================

if len(path_nodes) > 1:

    path_x = [
        node.x
        for node in path_nodes
    ]

    path_y = [
        node.y
        for node in path_nodes
    ]

    ax.plot(
        path_x,
        path_y,
        linewidth=3,
        label="RRTX path"
    )


# ============================================================
# GRAPH SETTINGS
# ============================================================

ax.set_xlim(
    X_BOUNDS[0],
    X_BOUNDS[1]
)

ax.set_ylim(
    Y_BOUNDS[0],
    Y_BOUNDS[1]
)

ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.set_title(
    "RRTX Baseline - Empty Environment"
)

ax.grid(True)

ax.legend()

ax.set_aspect("equal")

plt.tight_layout()


# ============================================================
# DISPLAY
# ============================================================

print("Displaying visualization...")

plt.show()
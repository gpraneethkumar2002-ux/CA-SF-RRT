from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set
import heapq
import numpy as np
from numpy.typing import NDArray
import scipy.spatial
import shapely.geometry as geom
from shapely.strtree import STRtree

ROUND_DP = 2
KDState = Tuple[float, float]

@dataclass
class VehicleGeometry:
    w_half: float
    lf: float
    lr: float
    bumpers_length: Tuple[float, float]

@dataclass
class Path:
    path: geom.LineString
    obs_clearance: float
    cost: float = np.inf

    def __post_init__(self):
        self.cost = self.path.length + 5 / self.obs_clearance

class RRT_Node:
    '''
    A node in the tree structure of RRT.

    It exists in the state space of the robot: (x, y, theta) wrt the world frame.
    '''
    def __init__(self, x:float, y:float) -> None:
        self.x = x                  # X pos of COG in world frame
        self.y = y                  # Y pos of COG in world frame

        self.g = np.inf
        '''The e-consistent cost-to-goal of reaching the goal from V'''
        self.lmc = np.inf
        '''The lookahead estimate of cost-to-goal'''

        self.N_o_inc: Dict["RRT_Node",float] = {}
        '''original list of incoming nodes. These map to their calculated distance.'''
        self.N_o_out: Dict["RRT_Node",float] = {}
        '''original list of outgoing nodes. These map to their calculated distance.'''
        self.N_r_inc: Dict["RRT_Node",float] = {}
        '''running list of incoming nodes. These map to their calculated distance.'''
        self.N_r_out: Dict["RRT_Node",float] = {}
        '''running list of outgoing nodes. These map to their calculated distance.'''

        self.p_T_out: Optional["RRT_Node"] = None
        '''Parent node for tree to goal'''
        self.C_T_inc: Set["RRT_Node"] = set()
        '''Child nodes for tree to goal.'''

        self.obst_clearance:Optional[float] = None
        '''The distance to the nearest obstacle for this node.'''

        self.is_in_collision = False
        self.is_v_bot = False

    def __lt__(self, other:"RRT_Node"):
        ''' This lets us use a heap-queue `heapq` of RRT_Node objects. '''
        return (min(self.g, self.lmc), self.g) < (min(other.g, other.lmc), other.g)

    def kd(self) -> KDState:
        '''
        Expression of node state as an array for use in KD_tree.

        `(x,y)` rounded off to ROUND_DP precision
        '''
        return (
            np.round(self.x, ROUND_DP),
            np.round(self.y, ROUND_DP),
                )

    def __repr__(self) -> str:
        return f"RRT_Node(x:{self.x:.2f}, y:{self.y:.2f})"

class RRT_X:
    def __init__(self, v_goal: RRT_Node, static_obs: List[geom.Polygon],
        lanelet_poly_exterior:geom.Polygon,
        x_bounds:Tuple[float,float], y_bounds:Tuple[float,float],
        agent_params:VehicleGeometry, edge_coll_rad:float
    ) -> None:

        ########################
        ## Tunable parameters ##
        ########################

        self.delta = 2.5
        '''Max connection distance between nodes'''
        self.r = self.delta*2.7
        '''Default distance to search for connection'''
        self.max_search_radius = self.r
        '''Max search radius. Used in `shrinking_ball_rad()`'''
        self.search_rad = self.r*4.5
        '''Multiplier for log term. Used in `shrinking_ball_rad()`'''
        self.kd_tree_rebuild_interval = 25
        '''How often to rebuild the KD-trees'''
        self.epsilon = 1.0
        '''threshold for ϵ-consistency'''
        self.curr_it = 0
        self.OBST_CLEARANCE = 0.5
        ''' An acceptable distance away from obstacles for the car to keep.
            This should be added with a representative distance from the car, eg. self.lf.
        '''
        self.EDGE_COLL_RAD = edge_coll_rad
        ''' how far around a dynamic obstacle's centroid should we look for nodes to check
            if they will collide?
        '''

        ## outward_facing status pointers ##
        self.curr_it = 0
        '''Current RRT iteration'''
        self.num_nodes = 0
        '''Current number of nodes in the RRT graph'''

        # Dimensions of robot
        self.hw = agent_params.w_half
        self.lf = agent_params.lf + agent_params.bumpers_length[0]
        self.lr = agent_params.lr + agent_params.bumpers_length[0]

        # initialise goal_node information
        self.goal_node = v_goal
        self.goal_node.g = 0
        self.goal_node.lmc = 0
        self.goal_node.p_T_out = None

        ######################
        ## Class attributes ##
        ######################

        self.V = scipy.spatial.KDTree( [self.goal_node.kd()] )
        '''KD-tree of nodes'''
        self.V_l: List[KDState] = []
        '''List to periodically update the KDTree'''
        self.Vq: Dict[KDState, RRT_Node] = {self.goal_node.kd(): self.goal_node}
        '''Mapping the KD tree coords to an actual node object'''
        self.X_obs_s = STRtree(static_obs)
        '''R-tree of static obstacles'''
        self.static_obs = static_obs    # collision
        self.X_obs_d: List[geom.Polygon] = []
        '''List of dynamic obstacles'''

        self.theta: Optional[float] = None
        '''Current heading of the robot'''
        self.v_bot:Optional[RRT_Node] = None
        self.pos_polygon:geom.Polygon = geom.Point(v_goal.x, v_goal.y).buffer(0.01)   #dummy init
        # self.v_bot:RRT_Node = RRT_Node(init_x,init_y)
        # self.pos_polygon = geom.Point(init_x,init_y).buffer(self.lf)
        '''Robot pose / state'''
        self.all_nodes = [self.goal_node] # for debugging

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.lanelet_ext = lanelet_poly_exterior.buffer(self.hw).exterior
        assert self.lanelet_ext.geom_type=="LinearRing"

        self.Q: List[RRT_Node] = []
        '''Priority queue to determine the order in which nodes become eps-consistent
        during the rewiring cascades.
        '''

        # For collision checking
        # Check if this is free.
        self.V_free:scipy.spatial.KDTree = None
        ''' Set of points that have been explicitly found to be collision-free from static obstacles '''
        self.V_free_l:List[KDState] = []
        '''List to periodically update the V_free KDTree (points free from static obs)'''
        self.V_obs:scipy.spatial.KDTree = None
        ''' Set of points that have been explicitly found to in collision with static obstacles '''
        self.V_obs_l:List[KDState] = []
        '''List to periodically update the V_obs KDTree (pts colliding w static obs)'''
        self.V_dist: Dict[KDState, float] = {}
        ''' Mapping a KDState to a min distance. (to obstacle, or free space) '''
        self.orphan_nodes: Set[RRT_Node] = set()
        '''List of orphaned nodes from obstacles appearing'''
        self.nodes_in_collision: Set[RRT_Node] = set()
        '''List of nodes for which their `is_in_collision` param is True'''

        self._get_obst_clearance(self.goal_node)
        # initialize goal_node's distance to nearest (static) obstacle

    def step(self, v_curr:Optional[Tuple[float,float,float]]=None,
            new_obs:Optional[List[geom.Polygon]]=None):

        self.curr_it += 1
        self.r = self._shrinking_ball_rad()

        if new_obs is not None:
            self._update_obstacles(new_obs)

        if v_curr is not None:
            self._update_robot(v_curr)

        v_coords = self._sample_node()
        # ? print(f"step(): sampled node at x:{v_coords[0]:.2f}, y:{v_coords[1]:.2f}")

        v_nearest_coords = self._get_nearest_coords(v_coords,self.V,self.V_l)
        if v_nearest_coords not in self.Vq:
            raise KeyError(f"{v_nearest_coords} not in Vq. {self.Vq}")
        v_nearest = self.Vq[v_nearest_coords]
        dist_to_nearest = self.dist(v_coords, v_nearest.kd())
        # ? print(f"step(): Nearest node: {v_nearest}, distance: {dist_to_nearest:.2f}")

        # if dist_to_nearest > self.delta:
        v_coords = self._saturate(v_coords, v_nearest.kd(), dist_to_nearest)
        v = RRT_Node(x=v_coords[0], y=v_coords[1])
        # ? print(f"step(): v after saturation: x:{v}, dist: {self.dist(v_coords, v_nearest.kd()):.2f}")

        #? print("step(): Checking for collision")
        if self._not_colliding(v):
            # ? print("step(): Extending")
            if self._extend(v):
                # ? print("step(): Rewiring neighbors")
                self._rewire_neighbors(v)
                # ? print("step(): Reducing inconsistency.")
                self._reduce_inconsistency()
                self.all_nodes.append(v)
                self.num_nodes += 1

    def get_trajectory(self) -> NDArray:
        ''' Sample trajectory from T (the shortest path subtree) to get the current
            control inputs.

            - hdg: Current heading of robot
            - vx: Current speed of robot
        '''
        if self.v_bot is None or self.theta is None:
            raise AttributeError("Position of v_bot has not been initialized.")

        traj = [self.v_bot]
        p = self.v_bot
        p_it = p.p_T_out
        its = 0

        while p_it is not None:
            path = self._get_path(p, p_it)
            if path is not None:
                traj.append(p_it)
            else:
                break

            assert p_it.p_T_out is not p, f"The parent of p_it:{p_it} is {p}."  # infinite loop

            if p_it.is_in_collision:
                print(f"{p_it} collides with an obstacle. This trajectory is invalid.")
                return None

            # There must be a loop somewhere if this happens
            if its > 100:
                print("Took > 100 iterations to find traj. We are likely to have loops in our graph.")
                return None

            p = p_it
            p_it = p_it.p_T_out
            its += 1

        # Under the full-traversal model, we only know if the subtree we are connected to
        # is invalid if it does not end at the goal.
        if traj[-1] is not self.goal_node:
            print("The last node is not the goal node. Invalid path.")
            print(traj[-5:])
            return None

        # Prune trajectory here
        p_id = 0
        p_prev_id = 0
        p_it_id = 1
        pruned_traj = [traj[0].kd()]
        while True:
            if p_prev_id==p_id:
                col_check = False
            else:
                col_check = True

            path = self._get_path(traj[p_id], traj[p_it_id], col_check)
            if path is not None:
                p_prev_id = p_it_id
                p_it_id += 1
            else:
                assert p_id != p_prev_id, f"{pruned_traj}"
                p_id = p_prev_id
                p_it_id = p_id + 1
                pruned_traj.append(traj[p_prev_id].kd())

                assert traj[p_id] is not self.v_bot, "Was unable to find a trajectory from v_bot"

            if p_it_id >= len(traj):
                pruned_traj.append(traj[-1].kd())
                break

        return np.array(pruned_traj)

    def _shrinking_ball_rad(self) -> float:
        ''' Returns the radius of a ball based on the cardinality of `self.V`
            We have a 4-dimensional space to search so d=4
        '''
        # Don't let this value become too small
        if self.r <= self.delta*1.2:
            return self.r

        c_v = max(self.num_nodes,1)

        # here we have max(c_v,2) because at init, log(1) = 0
        res = min(
            self.search_rad*np.power(np.log(max(c_v,2))/c_v, 1/2),
            self.max_search_radius
        )
        # print(f"Card(V): {c_v}, at iteration {self.curr_it}. new_r: {res:.2f}")
        return res

    def _update_obstacles(self, new_obs_l:List[geom.Polygon]):
        ''' Updates the value of `self.X_obs_d` (dynamic obstacles in environment)
        based on new observations. '''

        while len(self.X_obs_d)>0:
            obs = self.X_obs_d.pop()
            # ? print(f"_update_obstacles(): Removing obstacle at ({obs.centroid.x:.2f},{obs.centroid.y:.2f})")
            self._remove_obstacle(obs)
        # ? print("_update_obstacles(): Reducing inconsistency 1")
        self._reduce_inconsistency()

        # ? print(f"_update_obstacles(): {self.nodes_in_collision}")
        assert len(self.nodes_in_collision)==0, f"Not all collisions flags were cleared. {self.nodes_in_collision}"

        for obs in new_obs_l:
            # ? print(f"_update_obstacles(): Adding obstacle at ({obs.centroid.x:.2f},{obs.centroid.y:.2f})")
            self._add_obstacle(obs)
            self.X_obs_d.append(obs)

        assert len(self.X_obs_d) == len(new_obs_l), f"Not all elems of new_obs_l (len {len(new_obs_l)}) were copied to X_obs_d (len {len(self.X_obs_d)})"

        # ? print("_update_obstacles(): Propagating descendants")
        self._propagate_descendants()

        # ? print("_update_obstacles(): Verifying queue for v_bot")
        if self.v_bot is not None:
            self._verify_queue(self.v_bot)

        # ? print("_update_obstacles(): Reducing inconsistency 2")
        self._reduce_inconsistency()

    def _remove_obstacle(self, obs:geom.Polygon):
        # the edges that have path in collision with O
        E_o, c_l = self._get_all_colliding_edges(obs, coll_rad=self.EDGE_COLL_RAD)
        # ! Remove from E_o the edges that have path in collision with obs_near.
        # ! This is in the original algorithm,
        # ! but we skip it here because we assume all dynamic obstacle are cars,
        # ! which should not intersect with any static obstacles.
        for v in c_l:
            v.is_in_collision = False
            self.nodes_in_collision.discard(v)

        for v in E_o:
            v.is_in_collision = False
            self.nodes_in_collision.discard(v)
            for u, dic_name in E_o[v]:
                # Update the cost of d_pi(v,u)
                if dic_name=="N_o_inc":
                    dic = v.N_o_inc
                elif dic_name=="N_o_out":
                    dic = v.N_o_out
                elif dic_name=="N_r_inc":
                    dic = v.N_r_inc
                elif dic_name=="N_r_out":
                    dic = v.N_r_out
                assert u in dic, f"{u} was not found in {v}.{dic_name}"

                path = self._get_path(v,u)
                if path is not None:
                    dic[u] = path.cost
                else:
                    dic[u] = np.inf

            self._updateLMC(v)
            if not np.allclose(v.lmc, v.g):
                self._verify_queue(v)

    def _add_obstacle(self, obs: geom.Polygon):
        """Invalidate every graph edge intersecting a newly observed obstacle."""
        clearance_radius = self.lf + self.OBST_CLEARANCE
        E_o, c_l = self._get_all_colliding_edges(obs, coll_rad=clearance_radius)

        for v in c_l:
            v.is_in_collision = True
            self.nodes_in_collision.add(v)

        # Invalidate every directed graph-edge representation found by the
        # exhaustive geometric collision scan.
        for v, edges in E_o.items():
            for u, dic_name in edges:
                getattr(v, dic_name)[u] = np.inf

        # Explicitly inspect the current shortest-path-tree edges.  This is
        # the authoritative representation used by path extraction.
        obstacle_region = obs.buffer(clearance_radius)
        for v in list(self.Vq.values()):
            u = v.p_T_out
            if u is None:
                continue
            edge = geom.LineString([[v.x, v.y], [u.x, u.y]])
            if edge.intersects(obstacle_region):
                # Make every representation of this edge unusable.
                v.N_o_out[u] = np.inf
                v.N_r_out[u] = np.inf
                u.N_o_inc[v] = np.inf
                u.N_r_inc[v] = np.inf

                # Detach the invalid tree edge and orphan its downstream node.
                u.C_T_inc.discard(v)
                v.p_T_out = None
                self._verify_orphan(v)

    def _get_all_colliding_edges(self, obs: geom.Polygon, coll_rad) -> Tuple[
        Dict[RRT_Node, List[Tuple[RRT_Node, str]]],
        List[RRT_Node]
    ]:
        """Find every graph edge whose geometric path intersects the obstacle.

        The original implementation only inspected nodes within ``coll_rad`` of
        the obstacle centroid. That can miss a long edge which crosses the
        obstacle while both endpoints are farther away. We therefore inspect
        all currently stored graph edges and test their LineString geometry
        against the obstacle buffered by ``coll_rad``.
        """
        coll_set: Dict[Tuple[RRT_Node, RRT_Node], str] = {}
        coll_nodes: List[RRT_Node] = []

        # Collision test for graph vertices themselves.
        obstacle_region = obs.buffer(coll_rad)
        for x in self.Vq.values():
            if geom.Point(x.x, x.y).intersects(obstacle_region):
                coll_nodes.append(x)

        # Each neighbor dictionary represents directed graph edges.
        dict_names = ["N_o_inc", "N_o_out", "N_r_inc", "N_r_out"]

        for x in self.Vq.values():
            for dic_name in dict_names:
                dic = getattr(x, dic_name)
                for y in dic.keys():
                    edge = geom.LineString([[x.x, x.y], [y.x, y.y]])
                    if edge.intersects(obstacle_region):
                        coll_set[(x, y)] = dic_name

        ret: Dict[RRT_Node, List[Tuple[RRT_Node, str]]] = {}
        for (x, y), dic_name in coll_set.items():
            ret.setdefault(x, []).append((y, dic_name))

        return ret, coll_nodes

    def _verify_orphan(self, v: RRT_Node):
        # If v is in the Q, remove it
        if v in self.Q:
            self.Q.remove(v)
        heapq.heapify( self.Q ) # Remove from the queue and preserve invariant

        self.orphan_nodes.add(v)

    def _propagate_descendants(self) -> None:
        """Propagate orphan status through the complete shortest-path subtree."""
        # Use a queue so descendants at arbitrary depth are handled, not only
        # children one level below the initially orphaned vertices.
        pending = list(self.orphan_nodes)
        visited = set(self.orphan_nodes)

        while pending:
            v = pending.pop()
            for ch in v.C_T_inc:
                if ch not in visited:
                    visited.add(ch)
                    self.orphan_nodes.add(ch)
                    pending.append(ch)

        # Every node downstream of an orphan loses its valid g-value until
        # RRTX finds a new consistent parent.
        for v in self.orphan_nodes:
            outgoing = list(v.N_o_out.keys()) + list(v.N_r_out.keys())
            if v.p_T_out is not None:
                outgoing.append(v.p_T_out)

            for u in outgoing:
                if u in self.orphan_nodes:
                    continue
                u.g = np.inf
                self._verify_queue(u)

        while self.orphan_nodes:
            p = self.orphan_nodes.pop()
            p.g = np.inf
            p.lmc = np.inf
            if p.p_T_out is not None:
                p.p_T_out.C_T_inc.discard(p)
                p.p_T_out = None

    def _update_robot(self, v_curr: Tuple[float,float,float]):
        ''' Updates the value of `self.v_bot` based on new observations. '''
        x = v_curr[0]
        y = v_curr[1]
        self.theta = v_curr[2]

        self.v_bot = RRT_Node(x,y)
        self.v_bot.is_v_bot = True
        self._get_obst_clearance(self.v_bot)

        while True:
            # ? print("_update_robot(): Extending")
            if self._extend(self.v_bot):
                # ? print("_update_robot(): Rewiring neighbors")
                self._rewire_neighbors(self.v_bot)
                # self._verify_queue(self.v_bot)      # this is needed to ensure v_bot.g is updated
                # ? print("_update_robot(): Reducing inconsistency.")
                self._reduce_inconsistency()
                self.all_nodes.append(self.v_bot)
                self.num_nodes += 1
                break

            else:
                print(f"It {self.curr_it} | No nodes near v_bot {self.v_bot} yet. Sampling more pts.")
                self.step()

    def _sample_node(self) -> KDState:
        ''' Samples a node in the state space.

            The state space is:
            - x,y coordinates (within `self.x_bounds`, `self.y_bounds`)
            - heading (from 0 to 2pi)
        '''
        x = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
        y = np.random.uniform(self.y_bounds[0], self.y_bounds[1])

        return (x, y)

    def _get_nearest_coords(self, coords:KDState,
            query_KD:Optional[scipy.spatial.KDTree],
            query_list:Optional[List[KDState]]
        ) -> Optional[KDState]:

        ''' Queries `self.V` to find the nearest node in the list of vertices,
        if the query KD-tree + query_list are not empty.'''
        closest: KDState = None

        if query_KD is not None:
            # dist, idx = query_KD.query(coords)
            dist, idx = query_KD.query(coords, workers=1)
            closest = tuple( query_KD.data[idx,:] )

        # only iterates if there are elements inside.
        for vx in query_list:
            c_dist = self.dist(vx, coords)
            if closest is None or c_dist < dist:
                closest = vx
                dist = c_dist

        if closest is not None:
            return tuple(np.round(e, ROUND_DP) for e in closest)
            # Round off for accessing the dictionary
        else:
            return None

    def _saturate(self, coords:KDState, n_coords:KDState, dist:float) -> KDState:
        ''' Returns a `KDState` that is `self.delta` away from `n_coords`.

            If `use_L2`, then we just saturate with regards to the L2 norm.
        '''
        x_sat = n_coords[0] + (coords[0]-n_coords[0])*(self.delta/dist)
        y_sat = n_coords[1] + (coords[1]-n_coords[1])*(self.delta/dist)
        return (x_sat, y_sat)

    def _static_collision(self, v: RRT_Node) -> bool:
        ''' Static obstacle checking for the node at `v`.
        Returns `True` if there is no collision and `False` otherwise.'''
        curr_poly = geom.Point(v.x,v.y).buffer((self.lf + self.OBST_CLEARANCE))
        # ? print(f"_static_collision(): Checking highway collision with {v}")
        if self.lanelet_ext.intersects(curr_poly):
            return False    # Extension collides with the highway
        # ? print(f"_static_collision(): Checking static collision with {v}")
        obs = self.X_obs_s.query(curr_poly)
        for x in obs:
            if curr_poly.intersects(x):
                return False
        return True

    def _get_obst_clearance(self, v:RRT_Node)->None:
        '''Populates the "obst_clearance" attribute of RRT_node V, by finding the distance to the closest obstacle.'''
        curr_point = geom.Point(v.x, v.y)

        hw_dist = curr_point.distance(self.lanelet_ext)
        obs = self.X_obs_s.nearest(curr_point)
        obs_dist = curr_point.distance(obs)
        for o in self.X_obs_d:
            obs_dist = min(obs_dist, curr_point.distance(o))
        v.obst_clearance = min(obs_dist, hw_dist)

    def _not_colliding(self, v: RRT_Node) -> bool:
        """Return True only when the node is collision-free."""
        if not self._static_collision(v):
            return False

        vp = geom.Point(v.x, v.y).buffer(self.EDGE_COLL_RAD)
        for obst in self.X_obs_d:
            if vp.intersects(obst):
                v.is_in_collision = True
                self.nodes_in_collision.add(v)
                return False

        v.is_in_collision = False
        return True

    def _extend(self, v: RRT_Node) -> bool:
        ''' Attempts to insert v into
            G (the overall graph)
            T (the shortest-path subtree)

            Returns True if extension succeeded, else False.
        '''
        # Find nodes near to new graph v
        v_ind = self.V.query_ball_point(v.kd(), self.r, workers=-1)
        V_near = self.V.data[v_ind,:].tolist()
        for u in self.V_l:
            dist_to_node = self.dist(v.kd(), u)
            if dist_to_node <= self.r:
                V_near.append(u)

        if len(V_near)==0:
            print(f"_extend(): Node {v} has no neighbours within {self.r:.2f}")
            return False    # Early stop.
        # print("_extend(): V_near, nodes close to v:", V_near)

        # Find the parent of v in V_Near
        for x_coords in V_near:
            x_coords = tuple(np.round(x_coords, ROUND_DP))
            x = self.Vq[x_coords]
            # Compute a trajectory through the state space through points v and x.
            traj = self._get_path(v, x, explicit_col_check=False)
            if traj is not None:    # if traj is valid and obstacle free:
                if traj.path.length < self.r:
                    if v.lmc > traj.cost+x.lmc:
                        assert v is not x, f"Attempting to assign v:{v} as its own parent x:{x}"
                        v.p_T_out = x
                        x.C_T_inc.add(v)
                        v.lmc = traj.cost + x.lmc

        if v.p_T_out is None:
            print(f"_extend(): No feasible way of getting from {v} to other nodes")
            return False

        # Add v to the list of nodes
        if v.kd() not in self.Vq.keys():
            self.V_l.append(v.kd())
            self.Vq[v.kd()] = v
            print(f"It {self.curr_it:04d}, #Nodes: {self.num_nodes:04d} || Inserting {v.kd()} into Vq. r: {self.r:.2f}")
        else:
            print(f"It {self.curr_it:04d}, #Nodes: {self.num_nodes:04d} || {v.kd()} is already in self.Vq. Updating its key.")

        # ! Periodically rebuild the KD-Tree (tune how often this is done for performance)
        if len(self.V_l) > self.kd_tree_rebuild_interval:
            self.V = scipy.spatial.KDTree(
                np.vstack((self.V.data, np.array(self.V_l, ndmin=2))) )
            self.V_l = []

        for x_coords in V_near:
            x_coords = tuple(np.round(x_coords, ROUND_DP))
            x = self.Vq[x_coords]
            p_vx = self._get_path(v, x)
            if p_vx is not None:
                v.N_o_out[x] = p_vx.cost
                x.N_r_inc[v] = p_vx.cost
            p_xv = self._get_path(x, v)
            if p_xv is not None:
                v.N_o_inc[x] = p_xv.cost
                x.N_r_out[v] = p_xv.cost

        return True

    def _cull_neighbors(self, v:RRT_Node)->None:
        '''Removes all neighbors in the running outgoing list if they are more than `self.r` away.'''
        rem_list = []
        for u in v.N_r_out:
            if self.r < self.dist(v.kd(),u.kd()) and v.p_T_out is not u:
                rem_list.append(u)

        for u in rem_list:
            v.N_r_out.pop(u, None)      # Returns None if u is not found
            v.N_r_inc.pop(u, None)      # Returns None if u is not found

    def _rewire_neighbors(self, v: RRT_Node)->None:
        ''' Rewires v's in-neighbors to use v as their parent, if this
            results in a better cost-to-go
        '''
        if v.g-v.lmc > self.epsilon:
            # Cull neighbors
            self._cull_neighbors(v)

            for dic in [v.N_r_inc, v.N_o_inc]:
                for u in dic:
                    if v.p_T_out is u:
                        continue
                    assert u is not v, f"Somehow v:{v} got into its own incoming neighbor set."

                    dist = dic[u]
                    if u.lmc > dist + v.lmc:
                        u.lmc = dist + v.lmc
                        # makeParentOf(v,u)
                        if u.p_T_out is not None:
                            u.p_T_out.C_T_inc.discard(u)
                        u.p_T_out = v
                        v.C_T_inc.add(u)
                        if u.g - u.lmc > self.epsilon:
                            self._verify_queue(u)

    def _reduce_inconsistency(self):
        ''' Manages the rewiring cascade that propagates cost-to-goal info. '''

        if len(self.Q)>0 and self.v_bot is not None:
            while \
                not np.allclose(self.v_bot.lmc, self.v_bot.g) or \
                np.allclose(self.v_bot.g, np.inf) or \
                self.Q[0] < self.v_bot or \
                self.v_bot in self.Q:

                # ? print("Reducing Inconsistency. Q contents:", self.Q[:5])

                v = heapq.heappop(self.Q)
                if v.g-v.lmc > self.epsilon:
                    self._updateLMC(v)
                    if not v.is_in_collision:
                        self._rewire_neighbors(v)

                v.g = v.lmc

                if len(self.Q)==0:
                    break

        if self.v_bot is not None:
            # ? print(f"_reduce_inconsistency(): v_bot lmc:{self.v_bot.lmc:.2f}, g:{self.v_bot.g:.2f}")
            pass

    def _updateLMC(self, v: RRT_Node):
        # Recompute the one-step lookahead value from scratch.  Keeping a stale
        # finite lmc here can preserve a parent whose edge was just invalidated.
        self._cull_neighbors(v)
        old_parent = v.p_T_out
        v.lmc = np.inf
        p_prime = None
        for u in v.N_r_out:
            # Consider using set subtraction if self.orphan nodes is large.
            if u in self.orphan_nodes or u.p_T_out is v:
                continue
            dist = v.N_r_out[u]
            if v.lmc > dist + u.lmc:
                p_prime = u
                v.lmc = dist + u.lmc

        for u in v.N_o_out:
            # Consider using set subtraction if self.orphan nodes is large.
            if u in self.orphan_nodes or u.p_T_out is v:
                continue
            dist = v.N_o_out[u]
            if v.lmc > dist + u.lmc:
                p_prime = u
                v.lmc = dist + u.lmc

        if p_prime is not None:
            # makeParentOf(p_prime,v)
            if v.p_T_out is not None:
                v.p_T_out.C_T_inc.discard(v)
            v.p_T_out = p_prime
            p_prime.C_T_inc.add(v)

    def _verify_queue(self, v: RRT_Node):
        # If v is in Q, update it. If not, add v to Q.
        if v in self.Q:
            self.Q.remove(v)
            heapq.heapify(self.Q)

        heapq.heappush(self.Q, v)

    def dist(self, v1:KDState, v2:KDState):
        '''
        Returns a measure of distance between the two KDStates v1 and v2

        A KDState is a tuple (x, y).
        '''
        return  np.sqrt(
            (v1[0]-v2[0])**2 + \
            (v1[1]-v2[1])**2
        )

    def _get_path(self, v1:RRT_Node, v2:RRT_Node, explicit_col_check=False) -> "Path":
        ''' Compute path from one node to another.

            For now this is a straight-line path.

            If explicit_col_check is true, clearance of (self.lf + self.OBST_CLEARANCE)
            is checked for.
        '''
        line = geom.LineString([[v1.x, v1.y], [v2.x, v2.y]])
        # Populate this for trajectory cost information
        if v1.obst_clearance is None:
            self._get_obst_clearance(v1)
        if v2.obst_clearance is None:
            self._get_obst_clearance(v2)

        if explicit_col_check is False:
            if line.intersects(self.lanelet_ext):
                return None

            # Enforce the same vehicle + safety clearance for both static and
            # dynamic obstacles.  The previous implementation only rejected
            # geometric intersection, allowing a repaired edge to pass very
            # close to a dynamic obstacle.
            clearance_radius = self.lf + self.OBST_CLEARANCE
            search_region = line.buffer(clearance_radius)
            obsts = self.X_obs_s.query(search_region)
            for obs in list(obsts) + list(self.X_obs_d):
                if line.distance(obs) < clearance_radius:
                    return None

            min_d_obs = min(v1.obst_clearance, v2.obst_clearance)

        else:
            # Checks for clearance
            min_d_obs = line.distance(self.lanelet_ext)
            if min_d_obs < (self.lf + self.OBST_CLEARANCE):
                return None
            # Check static obstacles (and dynamic ones)
            for obs in [self.X_obs_s.nearest(line)]+self.X_obs_d:
                dist = line.distance(obs)
                if dist < (self.lf + self.OBST_CLEARANCE):
                    return None
                if dist < min_d_obs: min_d_obs = dist

        return Path(path=line, obs_clearance=min_d_obs)
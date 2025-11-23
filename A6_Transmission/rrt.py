import numpy as np


class Node:
    """RRT tree node with 6-DOF configuration."""

    def __init__(self, config):
        self.config = np.array(config, dtype=float)
        self.parent = None

    @property
    def position(self):
        return self.config[:3]

    @property
    def rotation(self):
        return self.config[3:6]


class RRT:
    """6-DOF RRT planner for position and rotation."""

    def __init__(
        self,
        transmission,
        start_config,
        goal_config,
        step_size,
        rotation_step_size,
        max_iter,
        goal_threshold,
        goal_sample_rate,
        seed=None,
    ):
        self.transmission = transmission
        self.start = Node(start_config)
        self.goal = Node(goal_config)
        self.step_size = step_size
        self.rotation_step_size = rotation_step_size
        self.max_iter = max_iter
        self.goal_threshold = goal_threshold
        self.goal_sample_rate = goal_sample_rate
        self.tree = [self.start]

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Tight position bounds focused around start/goal corridor
        start_pos = self.start.position
        goal_pos = self.goal.position
        margin = 50  # mm margin around start/goal TODO: move out of the scope into externally manageable parms

        self.pos_bounds = np.array(
            [
                [
                    min(start_pos[0], goal_pos[0]) - margin,
                    max(start_pos[0], goal_pos[0]) + margin,
                ],
                [
                    min(start_pos[1], goal_pos[1]) - margin,
                    max(start_pos[1], goal_pos[1]) + margin,
                ],
                [
                    min(start_pos[2], goal_pos[2]) - margin,
                    max(start_pos[2], goal_pos[2]) + margin,
                ],
            ]
        )

        # Dynamic rotation bounds based on start and goal
        # This allows RRT to explore from start orientation to goal orientation
        start_rot = self.start.rotation
        goal_rot = self.goal.rotation
        rot_margin = np.radians(10)  # 10° margin beyond goal

        self.rot_bounds = np.array(
            [
                [
                    min(start_rot[0], goal_rot[0]) - rot_margin,
                    max(start_rot[0], goal_rot[0]) + rot_margin,
                ],  # Roll
                [
                    min(start_rot[1], goal_rot[1]) - rot_margin,
                    max(start_rot[1], goal_rot[1]) + rot_margin,
                ],  # Pitch
                [
                    min(start_rot[2], goal_rot[2]) - rot_margin,
                    max(start_rot[2], goal_rot[2]) + rot_margin,
                ],  # Yaw
            ]
        )

        # Track best node seen so far
        self.best_node = self.start
        self.best_dist = self._distance_to_goal(self.start)

    def _format_node(self, node):
        """Format node position and rotation for display (DRY)."""
        pos = node.position
        rot_deg = np.degrees(node.rotation)
        return f"pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]mm, rot=[{rot_deg[0]:.1f}, {rot_deg[1]:.1f}, {rot_deg[2]:.1f}]°"

    def _distance_to_goal(self, node):
        """Calculate distance from node to goal."""
        return self.distance(node.config, self.goal.config)

    def sample_random(self):
        """Sample random configuration with goal biasing."""

        # Sample goal directly with some probability
        if np.random.random() < self.goal_sample_rate:
            return Node(self.goal.config.copy())

        # Otherwise sample uniformly in focused bounds
        pos = np.array(
            [
                np.random.uniform(self.pos_bounds[i][0], self.pos_bounds[i][1])
                for i in range(3)
            ]
        )
        rot = np.array(
            [
                np.random.uniform(self.rot_bounds[i][0], self.rot_bounds[i][1])
                for i in range(3)
            ]
        )

        return Node(np.concatenate([pos, rot]))

    def distance(self, config1, config2):
        """Weighted distance metric."""
        pos_dist = np.linalg.norm(config1[:3] - config2[:3])
        rot_dist = np.linalg.norm(config1[3:6] - config2[3:6])
        return pos_dist + 20.0 * rot_dist

    def nearest_node(self, random_node):
        """Find nearest tree node."""
        distances = [
            self.distance(node.config, random_node.config) for node in self.tree
        ]
        return self.tree[np.argmin(distances)]

    def steer(self, from_node, to_node):
        """Steer from one node toward another."""
        pos_dir = to_node.position - from_node.position
        pos_dist = np.linalg.norm(pos_dir)

        rot_dir = to_node.rotation - from_node.rotation
        rot_dist = np.linalg.norm(rot_dir)

        if pos_dist > 0:
            new_pos = from_node.position + (pos_dir / pos_dist) * min(
                self.step_size, pos_dist
            )
        else:
            new_pos = from_node.position.copy()

        if rot_dist > 0:
            new_rot = from_node.rotation + (rot_dir / rot_dist) * min(
                self.rotation_step_size, rot_dist
            )
        else:
            new_rot = from_node.rotation.copy()

        new_node = Node(np.concatenate([new_pos, new_rot]))
        new_node.parent = from_node
        return new_node

    def is_collision_free(self, node):
        """Check collision status."""
        try:
            return not self.transmission.check_collision(
                node.position.tolist(), node.rotation.tolist()
            )
        except Exception as e:
            print(f"Collision check error: {e}")
            return False

    def is_goal_reached(self, node):
        """Check if node is within threshold of goal."""
        return self.distance(node.config, self.goal.config) < self.goal_threshold

    def plan(self):
        """Execute RRT planning algorithm."""
        collision_count = 0
        first_free_iter = None

        # Print start and goal info
        print(f"Start: {self._format_node(self.start)}")
        print(f"Goal:  {self._format_node(self.goal)}")
        print(
            f"Position bounds: X=[{self.pos_bounds[0][0]:.0f}, {self.pos_bounds[0][1]:.0f}], "
            f"Y=[{self.pos_bounds[1][0]:.0f}, {self.pos_bounds[1][1]:.0f}], "
            f"Z=[{self.pos_bounds[2][0]:.0f}, {self.pos_bounds[2][1]:.0f}]"
        )
        print(
            f"Rotation bounds: Roll=[{np.degrees(self.rot_bounds[0][0]):.0f}°, {np.degrees(self.rot_bounds[0][1]):.0f}°], "
            f"Pitch=[{np.degrees(self.rot_bounds[1][0]):.0f}°, {np.degrees(self.rot_bounds[1][1]):.0f}°], "
            f"Yaw=[{np.degrees(self.rot_bounds[2][0]):.0f}°, {np.degrees(self.rot_bounds[2][1]):.0f}°]"
        )
        print()

        for i in range(self.max_iter):
            random_node = self.sample_random()
            nearest = self.nearest_node(random_node)
            new_node = self.steer(nearest, random_node)

            if self.is_collision_free(new_node):
                self.tree.append(new_node)

                # Track best node
                dist = self._distance_to_goal(new_node)
                if dist < self.best_dist:
                    self.best_node = new_node
                    self.best_dist = dist

                if first_free_iter is None:
                    first_free_iter = i + 1
                    print(f"First free node at iter {first_free_iter}:")
                    print(f"  {self._format_node(new_node)}, dist to goal: {dist:.1f}")

                if self.is_goal_reached(new_node):
                    final_dist = self._distance_to_goal(new_node)
                    print(f"\nGoal reached at iter {i+1}:")
                    print(f"  {self._format_node(new_node)}")
                    print(
                        f"  Tree: {len(self.tree)} nodes, dist to goal: {final_dist:.1f}"
                    )
                    return self._extract_path(new_node)
            else:
                collision_count += 1

            if (i + 1) % 100 == 0:
                # Show both last node and best node
                if len(self.tree) > 1:
                    last_node = self.tree[-1]
                    last_dist = self._distance_to_goal(last_node)
                    print(
                        f"[{i+1}/{self.max_iter}] nodes: {len(self.tree)}, coll: {collision_count}"
                    )
                    print(
                        f"  Last: {self._format_node(last_node)}, dist: {last_dist:.1f}"
                    )
                    print(
                        f"  Best: {self._format_node(self.best_node)}, dist: {self.best_dist:.1f}"
                    )
                else:
                    print(
                        f"[{i+1}/{self.max_iter}] nodes: {len(self.tree)}, coll: {collision_count}"
                    )

        print(f"\nMax iterations reached:")
        print(f"  Tree: {len(self.tree)} nodes, collisions: {collision_count}")
        print(
            f"  Best node achieved: {self._format_node(self.best_node)}, dist: {self.best_dist:.1f}"
        )
        if len(self.tree) > 1:
            last_node = self.tree[-1]
            last_dist = self._distance_to_goal(last_node)
            print(f"  Last node: {self._format_node(last_node)}, dist: {last_dist:.1f}")
        return None

    def _extract_path(self, goal_node):
        """Extract path by backtracking from goal."""
        path = []
        current = goal_node
        while current is not None:
            path.append(current.config.tolist())
            current = current.parent
        path.reverse()
        return path

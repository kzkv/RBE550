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
        pos_margin,
        rot_margin_deg,
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
        self.pos_margin = pos_margin
        self.rot_margin = np.radians(rot_margin_deg)
        self.tree = [self.start]

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Tight position bounds focused around a start/goal corridor
        start_pos = self.start.position
        goal_pos = self.goal.position

        print(
            f"Planning: step {self.step_size} mm, rotation step {np.degrees(self.rotation_step_size):.0f} deg, {self.max_iter} iterations, goal sampling {self.goal_sample_rate * 100:.0f}%"
        )

        self.pos_bounds = np.array(
            [
                [
                    min(start_pos[0], goal_pos[0]) - self.pos_margin,
                    max(start_pos[0], goal_pos[0]) + self.pos_margin,
                ],
                [
                    min(start_pos[1], goal_pos[1]) - self.pos_margin,
                    max(start_pos[1], goal_pos[1]) + self.pos_margin,
                ],
                [
                    min(start_pos[2], goal_pos[2]) - self.pos_margin,
                    max(start_pos[2], goal_pos[2]) + self.pos_margin,
                ],
            ]
        )

        # Dynamic rotation bounds based on start and goal
        # This allows RRT to explore from start orientation to goal orientation
        start_rot = self.start.rotation
        goal_rot = self.goal.rotation

        self.rot_bounds = np.array(
            [
                [
                    min(start_rot[0], goal_rot[0]) - self.rot_margin,
                    max(start_rot[0], goal_rot[0]) + self.rot_margin,
                ],  # Roll
                [
                    min(start_rot[1], goal_rot[1]) - self.rot_margin,
                    max(start_rot[1], goal_rot[1]) + self.rot_margin,
                ],  # Pitch
                [
                    min(start_rot[2], goal_rot[2]) - self.rot_margin,
                    max(start_rot[2], goal_rot[2]) + self.rot_margin,
                ],  # Yaw
            ]
        )

        # Track the best node seen so far
        self.best_node = self.start
        self.best_dist = self._distance_to_goal(self.start)

    @staticmethod
    def _format_node(node):
        """Format node position and rotation for display (DRY)."""
        pos = node.position
        rot_deg = np.degrees(node.rotation)
        return f"[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] mm, [{rot_deg[0]:.1f}, {rot_deg[1]:.1f}, {rot_deg[2]:.1f}] deg"

    def _distance_to_goal(self, node):
        """Calculate the distance from node to goal."""
        return self.distance(node.config, self.goal.config)

    def sample_random(self):
        """Sample a random configuration with goal biasing."""

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

    @staticmethod
    def distance(config1, config2):
        """Weighted distance metric."""
        pos_dist = np.linalg.norm(config1[:3] - config2[:3])
        rot_dist = np.linalg.norm(config1[3:6] - config2[3:6])
        return pos_dist + 20.0 * rot_dist

    def nearest_node(self, random_node):
        """Find the nearest tree node."""
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
        """Check if a node is within a threshold of goal."""
        return self.distance(node.config, self.goal.config) < self.goal_threshold

    def plan(self):
        """Execute RRT planning algorithm."""
        collision_count = 0

        for i in range(self.max_iter):
            random_node = self.sample_random()
            nearest = self.nearest_node(random_node)
            new_node = self.steer(nearest, random_node)

            if self.is_collision_free(new_node):
                self.tree.append(new_node)

                # Track the best node
                dist = self._distance_to_goal(new_node)
                if dist < self.best_dist:
                    self.best_node = new_node
                    self.best_dist = dist

                if self.is_goal_reached(new_node):
                    final_dist = self._distance_to_goal(new_node)
                    print(f"\nGoal reached at iter {i+1}:")
                    print(f"  {self._format_node(new_node)}")
                    print(
                        f"  Tree: {len(self.tree)} nodes, distance to goal: {final_dist:.1f} mm"
                    )
                    return self._extract_path(new_node)
            else:
                collision_count += 1

            if (i + 1) % 25 == 0:
                # Show both the last node and the best node
                last_node = self.tree[-1]
                last_dist = self._distance_to_goal(last_node)
                print(
                    f"[{i+1}/{self.max_iter}] nodes: {len(self.tree)}, collisions: {collision_count}"
                )
                print(f"  Last: {self._format_node(last_node)}, dist: {last_dist:.1f}")
                print(
                    f"  Best: {self._format_node(self.best_node)}, dist: {self.best_dist:.1f}"
                )

        print(f"\nMax iterations reached:")
        print(f"  Tree: {len(self.tree)} nodes, collisions: {collision_count}")
        print(
            f"  Best node: {self._format_node(self.best_node)}, dist: {self.best_dist:.1f}"
        )
        if len(self.tree) > 1:
            last_node = self.tree[-1]
            last_dist = self._distance_to_goal(last_node)
            print(f"  Last node: {self._format_node(last_node)}, dist: {last_dist:.1f}")
        return None

    @staticmethod
    def _extract_path(goal_node):
        """Extract the path by backtracking from goal."""
        path = []
        current = goal_node
        while current is not None:
            path.append(current.config.tolist())
            current = current.parent
        path.reverse()
        return path

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
        step_size=10.0,
        rotation_step_size=np.radians(5.0),
        max_iter=5000,
    ):
        self.transmission = transmission
        self.start = Node(start_config)
        self.goal = Node(goal_config)
        self.step_size = step_size
        self.rotation_step_size = rotation_step_size
        self.max_iter = max_iter
        self.tree = [self.start]

        self.pos_bounds = np.array([[-100, 400], [-100, 300], [-100, 400]])
        self.rot_bounds = np.array(
            [[-np.pi, np.pi], [-np.pi / 2, np.pi / 2], [-np.pi, np.pi]]
        )

    def sample_random(self):
        """Sample random configuration with optional goal biasing."""
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

    def is_goal_reached(self, node, threshold=20.0):
        """Check if node is within threshold of goal."""
        return self.distance(node.config, self.goal.config) < threshold

    def plan(self):
        """Execute RRT planning algorithm."""
        collision_count = 0
        first_free_iter = None

        for i in range(self.max_iter):
            random_node = self.sample_random()
            nearest = self.nearest_node(random_node)
            new_node = self.steer(nearest, random_node)

            if self.is_collision_free(new_node):
                self.tree.append(new_node)

                if first_free_iter is None:
                    first_free_iter = i + 1
                    print(
                        f"First free node: iter {first_free_iter}, pos {new_node.position}"
                    )

                if self.is_goal_reached(new_node):
                    print(f"Goal reached: iter {i+1}, tree {len(self.tree)} nodes")
                    return self._extract_path(new_node)
            else:
                collision_count += 1

            if (i + 1) % 500 == 0:
                print(
                    f"[{i+1}/{self.max_iter}] tree={len(self.tree)} coll={collision_count}"
                )

        print(f"Max iterations reached: tree={len(self.tree)} coll={collision_count}")
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

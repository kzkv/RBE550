"""
Forked from https://github.com/nathanlct/reeds-shepp-curves/blob/master/reeds_shepp.py
Optimized for performance after extensive profiling: ~550s vs. original ~1080s (about 50% improvement)

Key Optimizations Applied:

1. Eliminated dataclasses.replace() (→ 832s, -23%)
   Problem: replace() creates unnecessary overhead with validation and copying (124M+ calls)
   Solution: Direct PathElement() construction instead of replace(self, field=value)

2. Pre-computed Enum Negations (→ 640s, -21%)
   Problem: Enum arithmetic (Steering(-value), Gear(-value)) involves method calls and
            validation, executed 24M+ times during path transformations
   Solution: Pre-computed dictionary lookups (_STEERING_NEG, _GEAR_NEG) for O(1) access

3. Combined Path Filtering (→ 811s, -2.5%)
   Problem: Multiple iterations over path lists for filtering and validation
   Solution: Single-pass filtering with early rejection of invalid paths in get_all_paths()

4. Pre-negated Coordinates (included in #3)
   Problem: Repeated negation operations (-x, -y, -theta) for each of 12 path functions
   Solution: Pre-compute negations once: neg_x, neg_y, neg_theta = -x, -y, -theta

5. Direct Construction in PathElement.create()
   Problem: Calling reverse_gear() added unnecessary method call overhead
   Solution: Inline the negation using _GEAR_NEG dictionary lookup

Additional optimizations in firetruck.py's _integrate_path_step() (called 26M times):
- Hardcoded mathematical constants (pi/2, 2*pi) to avoid repeated calculations
- Fast angle normalization using direct comparison instead of modulo operator
- Cached attribute values to eliminate repeated lookups in tight loops

All optimizations were identified through profiler analysis, focusing on the top bottlenecks.
The "Own Time" metric was crucial in identifying where actual computation was happening vs.
where time was spent in child function calls.

=============

Implementation of the optimal path formulas given in the following paper:

OPTIMAL PATHS FOR A CAR THAT GOES BOTH FORWARDS AND BACKWARDS
J. A. REEDS AND L. A. SHEPP

notes: there are some typos in the formulas given in the paper;
some formulas have been adapted (cf http://msl.cs.uiuc.edu/~lavalle/cs326a/rs.c)

Each of the 12 functions (each representing 4 of the 48 possible words)
have 3 arguments x, y and phi, the goal position and angle (in degrees) of the
object given it starts at position (0, 0) and angle 0, and returns the
corresponding path (if it exists) as a list of PathElements (or an empty list).

(actually there are less than 48 possible words but this code is not optimized)
"""

import math
from enum import Enum
from dataclasses import dataclass


def M(theta):
    """
    Return the angle phi = theta mod (2 pi) such that -pi <= theta < pi.
    """
    theta = theta % (2 * math.pi)
    if theta < -math.pi:
        return theta + 2 * math.pi
    if theta >= math.pi:
        return theta - 2 * math.pi
    return theta


def R(x, y):
    """
    Return the polar coordinates (r, theta) of the point (x, y).
    """
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return r, theta


def change_of_basis(p1, p2):
    """
    Given p1 = (x1, y1, theta1) and p2 = (x2, y2, theta2) represented in a
    coordinate system with origin (0, 0) and rotation 0 (in degrees), return
    the position and rotation of p2 in the coordinate system which origin
    (x1, y1) and rotation theta1.
    """
    theta1 = deg2rad(p1[2])
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    new_x = dx * math.cos(theta1) + dy * math.sin(theta1)
    new_y = -dx * math.sin(theta1) + dy * math.cos(theta1)
    new_theta = p2[2] - p1[2]
    return new_x, new_y, new_theta


def rad2deg(rad):
    return 180 * rad / math.pi


def deg2rad(deg):
    return math.pi * deg / 180


def sign(x):
    return 1 if x >= 0 else -1


class Steering(Enum):
    LEFT = -1
    RIGHT = 1
    STRAIGHT = 0


class Gear(Enum):
    FORWARD = 1
    BACKWARD = -1


# Pre-compute negated enum values to avoid repeated lookups
_STEERING_NEG = {
    Steering.LEFT: Steering.RIGHT,
    Steering.RIGHT: Steering.LEFT,
    Steering.STRAIGHT: Steering.STRAIGHT,
}

_GEAR_NEG = {
    Gear.FORWARD: Gear.BACKWARD,
    Gear.BACKWARD: Gear.FORWARD,
}


@dataclass(eq=True, frozen=True)
class PathElement:
    param: float
    steering: Steering
    gear: Gear

    @classmethod
    def create(cls, param: float, steering: Steering, gear: Gear):
        if param >= 0:
            return cls(param, steering, gear)
        else:
            # Use pre-computed negation
            return cls(-param, steering, _GEAR_NEG[gear])

    def __repr__(self):
        s = (
            "{ Steering: "
            + self.steering.name
            + "\tGear: "
            + self.gear.name
            + "\tdistance: "
            + str(round(self.param, 2))
            + " }"
        )
        return s

    def reverse_steering(self):
        # Use pre-computed negation
        return PathElement(self.param, _STEERING_NEG[self.steering], self.gear)

    def reverse_gear(self):
        # Use pre-computed negation
        return PathElement(self.param, self.steering, _GEAR_NEG[self.gear])


def path_length(path):
    """
    this one's obvious
    """
    return sum([e.param for e in path])


def get_optimal_path(start, end):
    """
    Return the shortest path from start to end among those that exist
    """
    paths = get_all_paths(start, end)
    return min(paths, key=path_length)


def get_all_paths(start, end):
    """
    Return a list of all the paths from start to end generated by the
    12 functions and their variants
    """
    path_fns = [
        path1,
        path2,
        path3,
        path4,
        path5,
        path6,
        path7,
        path8,
        path9,
        path10,
        path11,
        path12,
    ]

    # get coordinates of end in the set of axis where start is (0,0,0)
    x, y, theta = change_of_basis(start, end)

    # Pre-negate coordinates to avoid repeated operations
    neg_x, neg_y, neg_theta = -x, -y, -theta

    paths = []
    for get_path in path_fns:
        # Generate base paths once
        p1 = get_path(x, y, theta)
        p2 = get_path(neg_x, y, neg_theta)
        p3 = get_path(x, neg_y, neg_theta)
        p4 = get_path(neg_x, neg_y, theta)

        # Apply transformations and filter in one pass
        for path in [p1, timeflip(p2), reflect(p3), reflect(timeflip(p4))]:
            if path:  # Skip None/empty paths
                # Filter out zero-param elements
                filtered = [e for e in path if e.param != 0]
                if filtered:  # Only add if still non-empty after filtering
                    paths.append(filtered)

    return paths


def timeflip(path):
    """
    timeflip transform described around the end of the article
    Optimized with pre-computed enum negations
    """
    if not path:
        return path
    return [PathElement(e.param, e.steering, _GEAR_NEG[e.gear]) for e in path]


def reflect(path):
    """
    reflect transform described around the end of the article
    Optimized with pre-computed enum negations
    """
    if not path:
        return path
    return [PathElement(e.param, _STEERING_NEG[e.steering], e.gear) for e in path]


def path1(x, y, phi):
    """
    Formula 8.1: CSC (same turns)
    """
    phi = deg2rad(phi)
    path = []

    u, t = R(x - math.sin(phi), y - 1 + math.cos(phi))
    v = M(phi - t)

    path.append(PathElement.create(t, Steering.LEFT, Gear.FORWARD))
    path.append(PathElement.create(u, Steering.STRAIGHT, Gear.FORWARD))
    path.append(PathElement.create(v, Steering.LEFT, Gear.FORWARD))

    return path


def path2(x, y, phi):
    """
    Formula 8.2: CSC (opposite turns)
    """
    phi = M(deg2rad(phi))
    path = []

    rho, t1 = R(x + math.sin(phi), y - 1 - math.cos(phi))

    if rho * rho >= 4:
        u = math.sqrt(rho * rho - 4)
        t = M(t1 + math.atan2(2, u))
        v = M(t - phi)

        path.append(PathElement.create(t, Steering.LEFT, Gear.FORWARD))
        path.append(PathElement.create(u, Steering.STRAIGHT, Gear.FORWARD))
        path.append(PathElement.create(v, Steering.RIGHT, Gear.FORWARD))

    return path


def path3(x, y, phi):
    """
    Formula 8.3: C|C|C
    """
    phi = deg2rad(phi)
    path = []

    xi = x - math.sin(phi)
    eta = y - 1 + math.cos(phi)
    rho, theta = R(xi, eta)

    if rho <= 4:
        A = math.acos(rho / 4)
        t = M(theta + math.pi / 2 + A)
        u = M(math.pi - 2 * A)
        v = M(phi - t - u)

        path.append(PathElement.create(t, Steering.LEFT, Gear.FORWARD))
        path.append(PathElement.create(u, Steering.RIGHT, Gear.BACKWARD))
        path.append(PathElement.create(v, Steering.LEFT, Gear.FORWARD))

    return path


def path4(x, y, phi):
    """
    Formula 8.4 (1): C|CC
    """
    phi = deg2rad(phi)
    path = []

    xi = x - math.sin(phi)
    eta = y - 1 + math.cos(phi)
    rho, theta = R(xi, eta)

    if rho <= 4:
        A = math.acos(rho / 4)
        t = M(theta + math.pi / 2 + A)
        u = M(math.pi - 2 * A)
        v = M(t + u - phi)

        path.append(PathElement.create(t, Steering.LEFT, Gear.FORWARD))
        path.append(PathElement.create(u, Steering.RIGHT, Gear.BACKWARD))
        path.append(PathElement.create(v, Steering.LEFT, Gear.BACKWARD))

    return path


def path5(x, y, phi):
    """
    Formula 8.4 (2): CC|C
    """
    phi = deg2rad(phi)
    path = []

    xi = x - math.sin(phi)
    eta = y - 1 + math.cos(phi)
    rho, theta = R(xi, eta)

    if rho <= 4:
        u = math.acos(1 - rho * rho / 8)
        A = math.asin(2 * math.sin(u) / rho)
        t = M(theta + math.pi / 2 - A)
        v = M(t - u - phi)

        path.append(PathElement.create(t, Steering.LEFT, Gear.FORWARD))
        path.append(PathElement.create(u, Steering.RIGHT, Gear.FORWARD))
        path.append(PathElement.create(v, Steering.LEFT, Gear.BACKWARD))

    return path


def path6(x, y, phi):
    """
    Formula 8.7: CCu|CuC
    """
    phi = deg2rad(phi)
    path = []

    xi = x + math.sin(phi)
    eta = y - 1 - math.cos(phi)
    rho, theta = R(xi, eta)

    if rho <= 4:
        if rho <= 2:
            A = math.acos((rho + 2) / 4)
            t = M(theta + math.pi / 2 + A)
            u = M(A)
            v = M(phi - t + 2 * u)
        else:
            A = math.acos((rho - 2) / 4)
            t = M(theta + math.pi / 2 - A)
            u = M(math.pi - A)
            v = M(phi - t + 2 * u)

        path.append(PathElement.create(t, Steering.LEFT, Gear.FORWARD))
        path.append(PathElement.create(u, Steering.RIGHT, Gear.FORWARD))
        path.append(PathElement.create(u, Steering.LEFT, Gear.BACKWARD))
        path.append(PathElement.create(v, Steering.RIGHT, Gear.BACKWARD))

    return path


def path7(x, y, phi):
    """
    Formula 8.8: C|CuCu|C
    """
    phi = deg2rad(phi)
    path = []

    xi = x + math.sin(phi)
    eta = y - 1 - math.cos(phi)
    rho, theta = R(xi, eta)
    u1 = (20 - rho * rho) / 16

    if rho <= 6 and 0 <= u1 <= 1:
        u = math.acos(u1)
        A = math.asin(2 * math.sin(u) / rho)
        t = M(theta + math.pi / 2 + A)
        v = M(t - phi)

        path.append(PathElement.create(t, Steering.LEFT, Gear.FORWARD))
        path.append(PathElement.create(u, Steering.RIGHT, Gear.BACKWARD))
        path.append(PathElement.create(u, Steering.LEFT, Gear.BACKWARD))
        path.append(PathElement.create(v, Steering.RIGHT, Gear.FORWARD))

    return path


def path8(x, y, phi):
    """
    Formula 8.9 (1): C|C[pi/2]SC
    """
    phi = deg2rad(phi)
    path = []

    xi = x - math.sin(phi)
    eta = y - 1 + math.cos(phi)
    rho, theta = R(xi, eta)

    if rho >= 2:
        u = math.sqrt(rho * rho - 4) - 2
        A = math.atan2(2, u + 2)
        t = M(theta + math.pi / 2 + A)
        v = M(t - phi + math.pi / 2)

        path.append(PathElement.create(t, Steering.LEFT, Gear.FORWARD))
        path.append(PathElement.create(math.pi / 2, Steering.RIGHT, Gear.BACKWARD))
        path.append(PathElement.create(u, Steering.STRAIGHT, Gear.BACKWARD))
        path.append(PathElement.create(v, Steering.LEFT, Gear.BACKWARD))

    return path


def path9(x, y, phi):
    """
    Formula 8.9 (2): CSC[pi/2]|C
    """
    phi = deg2rad(phi)
    path = []

    xi = x - math.sin(phi)
    eta = y - 1 + math.cos(phi)
    rho, theta = R(xi, eta)

    if rho >= 2:
        u = math.sqrt(rho * rho - 4) - 2
        A = math.atan2(u + 2, 2)
        t = M(theta + math.pi / 2 - A)
        v = M(t - phi - math.pi / 2)

        path.append(PathElement.create(t, Steering.LEFT, Gear.FORWARD))
        path.append(PathElement.create(u, Steering.STRAIGHT, Gear.FORWARD))
        path.append(PathElement.create(math.pi / 2, Steering.RIGHT, Gear.FORWARD))
        path.append(PathElement.create(v, Steering.LEFT, Gear.BACKWARD))

    return path


def path10(x, y, phi):
    """
    Formula 8.10 (1): C|C[pi/2]SC
    """
    phi = deg2rad(phi)
    path = []

    xi = x + math.sin(phi)
    eta = y - 1 - math.cos(phi)
    rho, theta = R(xi, eta)

    if rho >= 2:
        t = M(theta + math.pi / 2)
        u = rho - 2
        v = M(phi - t - math.pi / 2)

        path.append(PathElement.create(t, Steering.LEFT, Gear.FORWARD))
        path.append(PathElement.create(math.pi / 2, Steering.RIGHT, Gear.BACKWARD))
        path.append(PathElement.create(u, Steering.STRAIGHT, Gear.BACKWARD))
        path.append(PathElement.create(v, Steering.RIGHT, Gear.BACKWARD))

    return path


def path11(x, y, phi):
    """
    Formula 8.10 (2): CSC[pi/2]|C
    """
    phi = deg2rad(phi)
    path = []

    xi = x + math.sin(phi)
    eta = y - 1 - math.cos(phi)
    rho, theta = R(xi, eta)

    if rho >= 2:
        t = M(theta)
        u = rho - 2
        v = M(phi - t - math.pi / 2)

        path.append(PathElement.create(t, Steering.LEFT, Gear.FORWARD))
        path.append(PathElement.create(u, Steering.STRAIGHT, Gear.FORWARD))
        path.append(PathElement.create(math.pi / 2, Steering.LEFT, Gear.FORWARD))
        path.append(PathElement.create(v, Steering.RIGHT, Gear.BACKWARD))

    return path


def path12(x, y, phi):
    """
    Formula 8.11: C|C[pi/2]SC[pi/2]|C
    """
    phi = deg2rad(phi)
    path = []

    xi = x + math.sin(phi)
    eta = y - 1 - math.cos(phi)
    rho, theta = R(xi, eta)

    if rho >= 4:
        u = math.sqrt(rho * rho - 4) - 4
        A = math.atan2(2, u + 4)
        t = M(theta + math.pi / 2 + A)
        v = M(t - phi)

        path.append(PathElement.create(t, Steering.LEFT, Gear.FORWARD))
        path.append(PathElement.create(math.pi / 2, Steering.RIGHT, Gear.BACKWARD))
        path.append(PathElement.create(u, Steering.STRAIGHT, Gear.BACKWARD))
        path.append(PathElement.create(math.pi / 2, Steering.LEFT, Gear.BACKWARD))
        path.append(PathElement.create(v, Steering.RIGHT, Gear.FORWARD))

    return path

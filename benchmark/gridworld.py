"""
gridworld.py -- an embodied navigation task with deliberately PARTIAL senses.

This is the task the whole architecture argument rests on, so its design is
adversarial to the hypothesis on purpose: it is built so that no single
modality can solve it, and so that a monolithic network gets exactly the same
information the modular one does.

THE ENVIRONMENT. A size x size occupancy grid with random walls, one agent,
one goal. Walls are resampled until a path exists. Supervision is a BFS
distance field from the goal, so the optimal action is exact -- no RL, no
reward shaping, no variance from exploration. The label is simply the move
that most decreases true distance-to-goal.

THE THREE SENSES, AND WHY EACH IS CRIPPLED.

  image   (-> occipital)  An agent-centric VIEW x VIEW window, rendered. The
                          goal is usually outside it, so vision alone tells you
                          what is nearby but not which way to go.

  sensor  (-> parietal via VPL)  Eight rangefinder beams: normalised distance
                          to the first wall, plus a hit flag. Tells you what
                          blocks you. Carries no goal information whatsoever.

  text    (-> temporal)   A coarse instruction, "the goal lies north-east, far
                          away". Tells you roughly where to head. Carries no
                          obstacle information whatsoever.

Direction without obstacles is not enough (you walk into walls); obstacles
without direction are not enough (you have nowhere to go). A model must
combine at least two streams to beat the trivial baseline. That is the point:
if modular fusion is worth anything, this is where it should show.

FAIRNESS TO THE BASELINE. `flat_features()` returns the concatenation of the
same three observations, so a monolithic network sees identical information
and any difference is attributable to architecture, not to input.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config

#  north, south, west, east -- (dy, dx)
ACTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES = ("north", "south", "west", "east")
NUM_ACTIONS = len(ACTIONS)

#  Eight compass bearings for the rangefinder ring, matching SENSOR_BEAMS.
BEARINGS: tuple[tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
)

COMPASS = (
    "north", "north-east", "east", "south-east",
    "south", "south-west", "west", "north-west",
)


@dataclass
class Observation:
    """One navigation step, as the four things the brain can be shown."""

    image: np.ndarray            # (H, W, 3) uint8, agent-centric window
    sensor: np.ndarray           # (SENSOR_RAW_DIM,) float32
    text: str                    # coarse goal instruction
    action: int                  # ONE optimal move (ties broken at random)
    #  Every move that is equally optimal. Several states have more than one
    #  correct answer, so scoring against a single label would penalise a model
    #  for a choice that is not wrong. Accuracy is measured as "is the chosen
    #  action IN this set", which is the honest metric for an oracle with ties.
    optimal: tuple[int, ...] = ()
    view_cells: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    #  diagnostics, never fed to the model
    meta: dict = field(default_factory=dict)


class GridWorld:
    """Random walled grids with an exact BFS oracle."""

    def __init__(
        self,
        size: int = 12,
        wall_prob: float = 0.22,
        view: int = 5,
        cell_px: int = 12,
        beams: int = config.SENSOR_BEAMS,
        seed: int = 0,
    ):
        if beams != len(BEARINGS):
            raise ValueError(
                f"SENSOR_BEAMS={beams} but this world defines {len(BEARINGS)} "
                f"bearings; they must agree"
            )
        self.size = size
        self.wall_prob = wall_prob
        self.view = view
        self.cell_px = cell_px
        self.beams = beams
        self.rng = np.random.default_rng(seed)

    # -- world generation ------------------------------------------------
    def _distance_field(self, walls: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
        """BFS distance from every free cell to the goal. inf where unreachable."""
        dist = np.full(walls.shape, np.inf, dtype=np.float32)
        dist[goal] = 0.0
        queue = deque([goal])
        while queue:
            y, x = queue.popleft()
            for dy, dx in ACTIONS:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < self.size and 0 <= nx < self.size):
                    continue
                if walls[ny, nx] or np.isfinite(dist[ny, nx]):
                    continue
                dist[ny, nx] = dist[y, x] + 1.0
                queue.append((ny, nx))
        return dist

    def _new_world(self):
        """A grid, an agent and a goal, guaranteed connected and non-adjacent."""
        for _ in range(200):
            walls = self.rng.random((self.size, self.size)) < self.wall_prob
            walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True
            free = np.argwhere(~walls)
            if len(free) < 4:
                continue

            goal = tuple(free[self.rng.integers(len(free))])
            dist = self._distance_field(walls, goal)

            # Require a non-trivial path: at least 3 moves, and reachable.
            candidates = np.argwhere(np.isfinite(dist) & (dist >= 3))
            if len(candidates) == 0:
                continue
            agent = tuple(candidates[self.rng.integers(len(candidates))])
            return walls, np.array(agent), np.array(goal), dist
        raise RuntimeError("could not generate a connected grid; lower wall_prob")

    # -- the three senses ------------------------------------------------
    def _render_view(self, walls, agent, goal) -> np.ndarray:
        """Agent-centric window. The goal is usually OUTSIDE it, by design."""
        half = self.view // 2
        px = self.cell_px
        canvas = np.zeros((self.view * px, self.view * px, 3), dtype=np.uint8)

        for i in range(self.view):
            for j in range(self.view):
                y, x = agent[0] - half + i, agent[1] - half + j
                inside = 0 <= y < self.size and 0 <= x < self.size
                blocked = (not inside) or walls[y, x]

                if blocked:
                    colour = (28, 28, 34)              # wall / out of bounds
                elif inside and (y, x) == tuple(goal):
                    colour = (250, 240, 120)           # goal, when visible
                else:
                    colour = (150, 158, 170)           # free floor
                canvas[i * px:(i + 1) * px, j * px:(j + 1) * px] = colour

        # The agent itself, centred.
        c0, c1 = half * px, (half + 1) * px
        canvas[c0 + 3:c1 - 3, c0 + 3:c1 - 3] = (245, 90, 60)
        return canvas

    def _view_cells(self, walls, agent, goal) -> np.ndarray:
        """The same window as `_render_view`, as one value per cell channel.

        The rendered image paints each cell a flat colour, so this carries
        exactly the same information at the grid's own resolution. It exists so
        the monolithic baseline can be given identical content without having
        to reverse-engineer it out of pixels -- if the baseline saw less, any
        win for the modular system would be an artefact of unfair input.
        """
        half = self.view // 2
        cells = np.zeros((self.view, self.view, 3), dtype=np.float32)
        for i in range(self.view):
            for j in range(self.view):
                y, x = agent[0] - half + i, agent[1] - half + j
                inside = 0 <= y < self.size and 0 <= x < self.size
                blocked = (not inside) or walls[y, x]
                cells[i, j, 0] = 1.0 if blocked else 0.0                    # wall
                cells[i, j, 1] = 1.0 if (inside and (y, x) == tuple(goal)) else 0.0
                cells[i, j, 2] = 1.0 if (i == half and j == half) else 0.0  # self
        return cells

    def _rangefinder(self, walls, agent) -> np.ndarray:
        """Eight beams: normalised distance to the first wall, plus a hit flag.

        Carries NO goal information. This is a proximity sensor, exactly as a
        mobile robot has one, and it is the only sense that is cheap enough to
        be plausible on real hardware.
        """
        readings = np.zeros(self.beams * 2, dtype=np.float32)
        reach = float(self.size)
        for b, (dy, dx) in enumerate(BEARINGS):
            step = 1
            while step < self.size:
                y, x = agent[0] + dy * step, agent[1] + dx * step
                if not (0 <= y < self.size and 0 <= x < self.size) or walls[y, x]:
                    break
                step += 1
            readings[2 * b] = min(step, reach) / reach       # distance
            readings[2 * b + 1] = 1.0 if step < self.size else 0.0   # hit
        return readings

    def _instruction(self, agent, goal, dist_to_goal: float) -> str:
        """Coarse bearing and range. NO obstacle information."""
        dy, dx = float(goal[0] - agent[0]), float(goal[1] - agent[1])
        # Screen coords: +y is south. Bearing measured clockwise from north.
        angle = math.degrees(math.atan2(dx, -dy)) % 360.0
        bearing = COMPASS[int((angle + 22.5) // 45) % 8]

        span = math.hypot(dy, dx)
        if span <= 2.5:
            how_far = "close by"
        elif span <= 6.0:
            how_far = "some distance away"
        else:
            how_far = "far away"
        return f"the goal lies {bearing}, {how_far}"

    # -- sampling --------------------------------------------------------
    def sample(self) -> Observation:
        walls, agent, goal, dist = self._new_world()

        # Optimal action = any legal move that most reduces true distance.
        # Ties are collected, not silently resolved by action order: the first
        # version of this preferred whichever action came first in ACTIONS,
        # which skewed labels to 37% north / 34% south purely from tie-breaking.
        here = dist[tuple(agent)]
        costs: dict[int, float] = {}
        for index, (dy, dx) in enumerate(ACTIONS):
            y, x = agent[0] + dy, agent[1] + dx
            if not (0 <= y < self.size and 0 <= x < self.size) or walls[y, x]:
                continue
            costs[index] = float(dist[y, x])
        if not costs:
            return self.sample()                       # boxed in; resample
        best_cost = min(costs.values())
        optimal = tuple(i for i, c in costs.items() if c == best_cost)
        best = int(self.rng.choice(optimal))

        return Observation(
            image=self._render_view(walls, agent, goal),
            sensor=self._rangefinder(walls, agent),
            text=self._instruction(agent, goal, here),
            action=best,
            optimal=optimal,
            # Exact cell colours of the rendered window. Storing them avoids
            # the baseline having to re-derive the grid from pixels, so the
            # monolithic model provably sees the same information.
            view_cells=self._view_cells(walls, agent, goal),
            meta={
                "distance_to_goal": float(here),
                "goal_visible": bool(
                    abs(goal[0] - agent[0]) <= self.view // 2
                    and abs(goal[1] - agent[1]) <= self.view // 2
                ),
            },
        )

    def generate(self, n: int) -> list[Observation]:
        return [self.sample() for _ in range(n)]


def flat_features(obs: Observation) -> np.ndarray:
    """The same observation as one vector, for the monolithic baseline.

    The baseline must not be handicapped by receiving less than the modular
    system does, or the comparison proves nothing. Image is downsampled to the
    grid resolution it was rendered from (no information is added or removed --
    each cell was a flat colour block), text becomes a one-hot bearing plus a
    scalar range, and the sensor vector passes through unchanged.
    """
    grid = obs.view_cells.reshape(-1)

    bearing = np.zeros(len(COMPASS), dtype=np.float32)
    for i, name in enumerate(COMPASS):
        if f" {name}," in obs.text:
            bearing[i] = 1.0
            break
    span = np.float32(
        0.2 if "close by" in obs.text else (0.6 if "some distance" in obs.text else 1.0)
    )

    return np.concatenate(
        [grid.astype(np.float32), obs.sensor, bearing, [span]]
    ).astype(np.float32)


def modality_sufficiency(samples: list[Observation]) -> dict[str, float]:
    """How far each sense gets ALONE, as a majority-class oracle.

    Reported in the paper as evidence that the task genuinely requires fusion
    rather than asserting it. If any single modality scores near 1.0 the task
    is broken and the architecture claim would be unearned.
    """
    from collections import Counter

    actions = [o.action for o in samples]
    most_common = Counter(actions).most_common(1)[0][1] / len(actions)

    # Best achievable by mapping each distinct text instruction to one action.
    by_text: dict[str, Counter] = {}
    for o in samples:
        by_text.setdefault(o.text, Counter())[o.action] += 1
    text_only = sum(c.most_common(1)[0][1] for c in by_text.values()) / len(samples)

    # Same for the discretised sensor pattern (hit flags only).
    by_sensor: dict[tuple, Counter] = {}
    for o in samples:
        key = tuple(o.sensor[1::2].astype(int))
        by_sensor.setdefault(key, Counter())[o.action] += 1
    sensor_only = sum(c.most_common(1)[0][1] for c in by_sensor.values()) / len(samples)

    return {
        "majority_class": most_common,
        "text_only_ceiling": text_only,
        "sensor_only_ceiling": sensor_only,
        "goal_visible_fraction": float(
            np.mean([o.meta["goal_visible"] for o in samples])
        ),
    }


__all__ = [
    "GridWorld",
    "Observation",
    "flat_features",
    "modality_sufficiency",
    "ACTIONS",
    "ACTION_NAMES",
    "NUM_ACTIONS",
    "COMPASS",
]

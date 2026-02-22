import gymnasium as gym

from shared.achievements import KEYCORRIDOR_ACHIEVEMENTS, compute_eps, count_achievements


class KeyCorridorAchievementWrapper(gym.Wrapper):
    """Wraps MiniGrid-KeyCorridor-* to inject info["achievements"] and info["eps"]."""

    def __init__(self, env):
        super().__init__(env)
        self._ach = {a: False for a in KEYCORRIDOR_ACHIEVEMENTS}
        self._prev_open_doors = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ach = {a: False for a in KEYCORRIDOR_ACHIEVEMENTS}
        self._prev_open_doors = set()
        info["achievements"] = dict(self._ach)
        info["eps"] = compute_eps(count_achievements(self._ach), 0.0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        inner = self.unwrapped

        # Detect what the agent is carrying
        if inner.carrying is not None:
            if inner.carrying.type == 'key':
                self._ach["found_key"] = True
                self._ach["picked_up_key"] = True
            elif inner.carrying.type == 'ball':
                self._ach["found_target"] = True
                self._ach["picked_up_target"] = True

        # Scan grid for doors and balls
        ax, ay = inner.agent_pos
        curr_open_doors = set()
        for j in range(inner.grid.height):
            for i in range(inner.grid.width):
                cell = inner.grid.get(i, j)
                if cell is None:
                    continue
                if cell.type == 'door':
                    if abs(ax - i) + abs(ay - j) <= 1:
                        self._ach["reached_door"] = True
                    if cell.is_open:
                        curr_open_doors.add((i, j))
                elif cell.type == 'ball' and not self._ach["found_target"]:
                    # Ball visible when within 2 tiles (partial obs env)
                    if abs(ax - i) + abs(ay - j) <= 2:
                        self._ach["found_target"] = True

        # Detect newly opened doors
        if curr_open_doors - self._prev_open_doors:
            self._ach["opened_door"] = True
        self._prev_open_doors = curr_open_doors

        info["achievements"] = dict(self._ach)
        info["eps"] = compute_eps(count_achievements(self._ach), 0.0)
        return obs, reward, terminated, truncated, info

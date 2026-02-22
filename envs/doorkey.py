import gymnasium as gym

from shared.achievements import DOORKEY_ACHIEVEMENTS, compute_eps, count_achievements


class DoorKeyAchievementWrapper(gym.Wrapper):
    """Wraps MiniGrid-DoorKey-* to inject info["achievements"] and info["eps"]."""

    def __init__(self, env):
        super().__init__(env)
        self._ach = {a: False for a in DOORKEY_ACHIEVEMENTS}
        self._prev_door_open = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ach = {a: False for a in DOORKEY_ACHIEVEMENTS}
        self._prev_door_open = False
        info["achievements"] = dict(self._ach)
        info["eps"] = compute_eps(count_achievements(self._ach), 0.0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        inner = self.unwrapped

        # Detect key pickup
        if inner.carrying is not None and inner.carrying.type == 'key':
            self._ach["found_key"] = True
            self._ach["picked_up_key"] = True

        # Scan grid for door
        door_pos = None
        door_is_open = False
        ax, ay = inner.agent_pos
        for j in range(inner.grid.height):
            for i in range(inner.grid.width):
                cell = inner.grid.get(i, j)
                if cell is not None and cell.type == 'door':
                    door_pos = (i, j)
                    door_is_open = cell.is_open
                    # reached_door: agent adjacent to door
                    if abs(ax - i) + abs(ay - j) <= 1:
                        self._ach["reached_door"] = True
                    # opened_door: door just became open
                    if door_is_open and not self._prev_door_open:
                        self._ach["opened_door"] = True
                    # crossed_door: agent is past the door's column
                    if door_is_open and ax > i:
                        self._ach["crossed_door"] = True
                    break  # only one door in DoorKey
            else:
                continue
            break

        self._prev_door_open = door_is_open

        # reached_goal: episode terminates successfully
        if terminated and reward > 0:
            self._ach["reached_goal"] = True

        info["achievements"] = dict(self._ach)
        info["eps"] = compute_eps(count_achievements(self._ach), 0.0)
        return obs, reward, terminated, truncated, info

DOORKEY_ACHIEVEMENTS = [
    "found_key",
    "picked_up_key",
    "reached_door",
    "opened_door",
    "crossed_door",
    "reached_goal",
]

KEYCORRIDOR_ACHIEVEMENTS = [
    "found_key",
    "picked_up_key",
    "reached_door",
    "opened_door",
    "found_target",
    "picked_up_target",
]


def get_achievements_for_env(env_id):
    if "KeyCorridor" in env_id:
        return KEYCORRIDOR_ACHIEVEMENTS
    elif "DoorKey" in env_id:
        return DOORKEY_ACHIEVEMENTS
    raise ValueError(f"No achievements defined for env_id: {env_id}")


# EPS = achievements + 0.9 * (materials_acquired / materials_needed)
def compute_eps(achievements_completed, materials_progress):
    return achievements_completed + 0.9 * materials_progress


# count the number of completed achievements from a dict of bools
def count_achievements(achievements_dict):
    return sum(1 for v in achievements_dict.values() if v)

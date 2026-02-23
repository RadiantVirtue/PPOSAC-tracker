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

DOORKEY_ACHIEVEMENT_REWARDS = {
    "found_key":     0.05,
    "picked_up_key": 0.05,
    "reached_door":  0.1,
    "opened_door":   0.1,
    "crossed_door":  0.1,
    "reached_goal":  1.0,   # strongly biased final
}

KEYCORRIDOR_ACHIEVEMENT_REWARDS = {
    "found_key":        0.05,
    "picked_up_key":    0.05,
    "reached_door":     0.1,
    "opened_door":      0.1,
    "found_target":     0.1,
    "picked_up_target": 1.0,   # strongly biased final
}


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

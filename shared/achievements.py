"""Crafter achievement definitions and EPS scoring.

Crafter provides 22 achievements tracking a tech-tree progression.
info["achievements"] from crafter.Env returns integer counts (0 or 1 per episode).
"""

CRAFTER_ACHIEVEMENTS = [
    # Tier 1 — basic survival
    "collect_wood",
    "collect_sapling",
    "collect_drink",
    "eat_plant",
    "wake_up",
    # Tier 2 — basic crafting / combat
    "place_table",
    "collect_stone",
    "defeat_zombie",
    "eat_cow",
    "defeat_skeleton",
    "make_wood_pickaxe",
    "make_wood_sword",
    # Tier 3 — advanced crafting
    "collect_coal",
    "place_stone",
    "place_furnace",
    "make_stone_pickaxe",
    "make_stone_sword",
    # Tier 4 — endgame
    "collect_iron",
    "collect_diamond",
    "make_iron_pickaxe",
    "make_iron_sword",
    "place_plant",
]

CRAFTER_ACHIEVEMENT_REWARDS = {
    # Tier 1
    "collect_wood":    0.05,
    "collect_sapling": 0.05,
    "collect_drink":   0.05,
    "eat_plant":       0.05,
    "wake_up":         0.02,
    # Tier 2
    "place_table":       0.1,
    "collect_stone":     0.1,
    "defeat_zombie":     0.1,
    "eat_cow":           0.1,
    "defeat_skeleton":   0.15,
    "make_wood_pickaxe": 0.1,
    "make_wood_sword":   0.1,
    # Tier 3
    "collect_coal":      0.2,
    "place_stone":       0.1,
    "place_furnace":     0.2,
    "make_stone_pickaxe": 0.2,
    "make_stone_sword":  0.2,
    # Tier 4
    "collect_iron":    0.3,
    "collect_diamond": 1.0,
    "make_iron_pickaxe": 0.4,
    "make_iron_sword": 0.4,
    "place_plant":     0.3,
}


ACHIEVEMENT_MATERIALS = {
    "place_table":         {"wood": 2},
    "make_wood_pickaxe":   {"wood": 1},
    "make_wood_sword":     {"wood": 1},
    "place_stone":         {"stone": 1},
    "place_furnace":       {"stone": 4},
    "make_stone_pickaxe":  {"stone": 3},
    "make_stone_sword":    {"stone": 2},
    "make_iron_pickaxe":   {"iron": 3},
    "make_iron_sword":     {"iron": 2},
    "place_plant":         {"sapling": 1},
}


def compute_materials_fraction(achievements_dict: dict, inventory: dict) -> float:
    """Fraction of materials held toward the closest unachieved achievement.

    For each unachieved achievement that has material prerequisites, computes
    the ratio of materials held vs required (capped per material). Returns the
    maximum ratio across all such achievements — i.e. how prepared the agent
    is for the achievement it is closest to completing materially.
    """
    best = 0.0
    for ach, required in ACHIEVEMENT_MATERIALS.items():
        if achievements_dict.get(ach, False):
            continue
        total_needed = sum(required.values())
        if total_needed == 0:
            continue
        held = sum(min(int(inventory.get(mat, 0)), need) for mat, need in required.items())
        ratio = held / total_needed
        if ratio > best:
            best = ratio
    return best


def get_achievements_for_env(env_id: str) -> list:
    """Return achievement list for the given env_id (always Crafter now)."""
    return CRAFTER_ACHIEVEMENTS


def compute_eps(achievements_completed: float, materials_progress: float) -> float:
    """EPS = achievements unlocked + 0.9 * materials_progress."""
    return achievements_completed + 0.9 * materials_progress


def count_achievements(achievements_dict: dict) -> int:
    """Count number of achievements with a truthy value."""
    return sum(1 for v in achievements_dict.values() if v)

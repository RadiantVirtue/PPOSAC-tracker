"""Crafter achievement definitions.

Shared by all entities that use the Crafter environment. Contains no logic.
"""

ACHIEVEMENT_NAMES = [
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

ACHIEVEMENT_LABEL_MAP = {
    "collect_wood":       "Wood",
    "collect_stone":      "Stone",
    "collect_iron":       "Iron",
    "collect_coal":       "Coal",
    "collect_diamond":    "Diamond",
    "collect_sapling":    "Sapling",
    "collect_drink":      "Drink",
    "defeat_zombie":      "Zombie",
    "defeat_skeleton":    "Skeleton",
    "eat_plant":          "Eat Plant",
    "eat_cow":            "Eat Cow",
    "wake_up":            "Wake Up",
    "place_table":        "Table",
    "place_stone":        "Place Stone",
    "place_furnace":      "Furnace",
    "place_plant":        "Place Plant",
    "make_wood_pickaxe":  "Wood Pickaxe",
    "make_stone_pickaxe": "Stone Pickaxe",
    "make_iron_pickaxe":  "Iron Pickaxe",
    "make_wood_sword":    "Wood Sword",
    "make_stone_sword":   "Stone Sword",
    "make_iron_sword":    "Iron Sword",
}

ACHIEVEMENT_GROUPS = {
    "fighting": frozenset({
        "Zombie", "Skeleton",
        "Wood Sword", "Stone Sword", "Iron Sword",
    }),
    "resource": frozenset({
        "Wood", "Stone", "Iron", "Coal",
        "Wood Pickaxe", "Stone Pickaxe", "Iron Pickaxe",
    }),
    "crafting": frozenset({
        "Wood Pickaxe", "Stone Pickaxe", "Iron Pickaxe",
        "Wood Sword", "Stone Sword", "Iron Sword",
        "Furnace",
    }),
    "housing": frozenset({
        "Furnace", "Table", "Place Stone", "Wake Up",
    }),
}

# Materials required to craft each achievement — used by compute_eps() only
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

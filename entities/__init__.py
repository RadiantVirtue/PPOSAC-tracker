"""entities — SOFT-CODED entity wrappers and environment definitions.

Each entity = one algorithm + one environment. It implements the Entity
protocol from core/entity.py.

To add a new entity: see ADDING_ENTITIES.md at the project root.

Subdirectories:
    definitions/   — environment-specific achievement definitions; shared by
                     all entities that use the same environment

Current entities:
    ppo_crafter.py — PPO (Stable-Baselines3) + Crafter

Current definitions:
    definitions/crafter.py — 22 Crafter achievements: names, labels, groups, materials
"""

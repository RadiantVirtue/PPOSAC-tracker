"""entities/definitions - SOFT-CODED environment achievement definitions.

One file per environment. Each file defines:
    ACHIEVEMENT_NAMES     list[str]              - ordered achievement IDs
    ACHIEVEMENT_LABEL_MAP dict[str, str]         - ID -> display label
    ACHIEVEMENT_GROUPS    dict[str, frozenset]   - group name -> set of labels
    ACHIEVEMENT_MATERIALS dict[str, dict]        - (optional) materials per craftable

Imported by entity files and by training/achievement_tracker.py.
No logic lives here - only data definitions.

To add a new environment's definitions: create definitions/{environment}.py.
See ADDING_ENTITIES.md for the required structure.
"""

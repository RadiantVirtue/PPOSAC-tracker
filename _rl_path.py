"""Single place for the sys.path injection needed to import from rl-starter-files.

Usage:
    import _rl_path  # noqa: F401
    from model import ACModel
    from utils.format import get_obss_preprocessor
"""
import os
import sys

_p = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rl-starter-files')
)
if _p not in sys.path:
    sys.path.insert(0, _p)

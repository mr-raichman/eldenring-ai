"""
config - split by concern into submodules, re-exported here so that
`from eldenring_ai import config; config.LEARNING_RATE` (and friends) keep working.

`paths` is also exposed as a submodule: `config.paths.MODELS_DIR`.
"""

from .training import *  # noqa: F401,F403
from .vision import *  # noqa: F401,F403
from .offsets import *  # noqa: F401,F403
from .runtime import *  # noqa: F401,F403

from . import paths  # noqa: F401

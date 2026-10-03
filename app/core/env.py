"""Environment loading, independent of import order.

The .env file used to be loaded as a side effect of importing
``app.database.session``. Any module that read its configuration at import time
and was imported first therefore saw an empty environment. That is how the
token-verification settings ended up blank at runtime, which silently dropped
the service into its unverified development path.

Loading happens here, in a module with no other responsibilities, so every
consumer can import it first and be sure the environment is populated. This is
deliberately a separate module from ``app.core.config``, which holds domain
constants rather than process configuration.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT_DIR = Path(__file__).resolve().parents[2]
_ENV_PATH = _ROOT_DIR / ".env"
if not _ENV_PATH.exists():
    _ENV_PATH = _ROOT_DIR / ".env.example"

if _ENV_PATH.exists():
    # override=False: a real environment variable wins over the file, so a
    # deployment is not silently redefined by a checked-out .env.
    load_dotenv(_ENV_PATH)


def get_env(name: str, default: str = "") -> str:
    """A trimmed environment value, read at call time.

    Read lazily so a value injected after import (by a test, or by a process
    manager that sets it late) is still honoured.
    """
    return os.getenv(name, default).strip()

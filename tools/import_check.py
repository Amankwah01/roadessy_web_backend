"""Runtime import checker for the `app` package.

Walks the `app` package and attempts to import every discovered module,
reporting any exceptions with tracebacks and exiting non-zero if errors
are found.
"""

import importlib
import os
import pkgutil
import sys
import traceback


def main() -> int:
    # ensure the repository root is on sys.path so local packages can be imported
    root = os.getcwd()
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        import app
    except Exception:
        print("ERROR: failed to import top-level package 'app'")
        traceback.print_exc()
        return 2

    errors = []
    for module_info in pkgutil.walk_packages(app.__path__, app.__name__ + "."):
        name = module_info.name
        try:
            importlib.import_module(name)
            print(f"OK:    {name}")
        except Exception:
            print(f"ERROR: {name}")
            traceback.print_exc()
            errors.append(name)

    if errors:
        print(f"\nFound {len(errors)} import errors.")
        return 2

    print("All imports successful")
    return 0


if __name__ == "__main__":
    sys.exit(main())

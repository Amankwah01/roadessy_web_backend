import pkgutil
import importlib
import sys
import os

root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root)

errors = []
for finder, name, ispkg in pkgutil.walk_packages(path=[os.path.join(root, "app")], prefix="app."):
    try:
        importlib.import_module(name)
    except Exception as exc:
        errors.append((name, exc))

if not errors:
    print("Import check: OK — no import-time errors detected")
    sys.exit(0)

print(f"Import check: {len(errors)} module(s) failed to import")
for name, exc in errors:
    print(name, type(exc).__name__, exc)
sys.exit(1)

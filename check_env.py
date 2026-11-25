"""Check installed package versions against requirements.txt.

The results depend on the numerics library versions, so running in the wrong
environment should be noticed rather than silently producing different
numbers.  Exits non-zero on any mismatch, and prints what to do about it.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version

REQUIREMENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "requirements.txt")


def pinned():
    """Yield ``(package, required_version)`` from requirements.txt."""
    with open(REQUIREMENTS) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "==" not in line:
                continue
            name, _, required = line.partition("==")
            yield name.strip(), required.strip()


def main() -> int:
    print(f"python   {sys.version.split()[0]}  |  {sys.executable}")
    problems = []
    for name, required in pinned():
        try:
            installed = version(name)
        except PackageNotFoundError:
            print(f"{name:10s} MISSING (need {required})")
            problems.append(name)
            continue
        mark = "" if installed == required else f"  != {required}"
        print(f"{name:10s} {installed}{mark}")
        if installed != required:
            problems.append(name)

    if problems:
        print()
        print("Version mismatch: " + ", ".join(sorted(problems)))
        print("The pinned versions are the ones these results were produced")
        print("with.  To match them:")
        print()
        print("    python -m venv .venv")
        print("    source .venv/bin/activate")
        print("    pip install -r requirements.txt")
        print()
        print("To run anyway, call make_results.py directly instead of")
        print("run_all.sh.  Expect small numerical differences.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

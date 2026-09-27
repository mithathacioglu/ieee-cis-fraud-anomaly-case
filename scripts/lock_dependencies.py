"""pyproject'teki bağımlılıklardan ulaşılan kurulu sürümleri kaydeder."""

import tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

root = Path(__file__).resolve().parents[1]
project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
pending = [Requirement(item) for item in project["dependencies"]]
pending += [Requirement(item) for group in project["optional-dependencies"].values() for item in group]
visited, pinned = set(), {}
while pending:
    requirement = pending.pop()
    name = canonicalize_name(requirement.name)
    key = (name, tuple(sorted(requirement.extras)))
    if key in visited:
        continue
    visited.add(key)
    distribution = metadata.distribution(name)
    if requirement.specifier and distribution.version not in requirement.specifier:
        raise RuntimeError(f"Installed {name} does not satisfy {requirement}")
    pinned[name] = distribution.version
    for dependency in distribution.requires or []:
        child = Requirement(dependency)
        if child.marker is None or any(child.marker.evaluate({"extra": extra}) for extra in {"", *requirement.extras}):
            pending.append(child)
(root / "requirements.lock").write_text(
    "# Validated on Python 3.14.6 / Windows. Install before pip install -e . --no-deps\n" +
    "\n".join(f"{name}=={version}" for name, version in sorted(pinned.items())) + "\n")
print(f"Locked {len(pinned)} installed dependencies.")

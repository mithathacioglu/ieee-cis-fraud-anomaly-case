"""Kaynağı ve raporları açık bir allowlist'e göre paketler."""

import hashlib
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = [root / name for name in ("README.md", "pyproject.toml", "requirements.lock", ".gitignore")]
for folder in ("src/fraud_case", "tests", "scripts", "config", "knowledge_base", "docs", "reports"):
    paths.extend(p for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
paths.extend(p for p in (root / ".github").rglob("*") if p.is_file())
paths = sorted(set(paths))
output = root / "dist"
output.mkdir(exist_ok=True)
archive = output / "logo-fraud-case.zip"
manifest = {}
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in paths:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        manifest[relative] = hashlib.sha256(content).hexdigest()
        bundle.writestr(relative, content)
    bundle.writestr("submission_manifest.json", json.dumps(manifest, indent=2))
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    assert not any(name.startswith(("models/", ".tools/", ".venv/", "artifacts/", "data/raw/", "data/processed/")) for name in bundle.namelist())
print(f"Packaged {len(paths)} source/report files: {archive} ({archive.stat().st_size:,} bytes)")

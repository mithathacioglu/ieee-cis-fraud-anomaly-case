"""Açılmış arşivden manifest'i, linkleri ve testleri doğrular."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive = ROOT / "dist/logo-fraud-case.zip"
    (ROOT / "tmp").mkdir(exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="verified_submission_", dir=ROOT / "tmp"))
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        manifest = json.loads(bundle.read("submission_manifest.json"))
        assert set(bundle.namelist()) == set(manifest) | {"submission_manifest.json"}
        for name, digest in manifest.items():
            assert (target / name).resolve().is_relative_to(target.resolve()), name
            content = bundle.read(name)
            assert hashlib.sha256(content).hexdigest() == digest, name
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        bundle.extractall(target)
    broken = []
    for path in [target / "README.md", *(target / "docs").glob("*.md"), *(target / "reports").glob("*.md")]:
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in link or link.startswith("#"):
                continue
            if not (path.parent / link.split("#", 1)[0]).exists():
                broken.append((path.relative_to(target).as_posix(), link))
    assert not broken, broken
    env = {**os.environ, "PYTHONPATH": str(target / "src"), "PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8"}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    subprocess.run([sys.executable, "-c", "from pathlib import Path; import fraud_case; assert Path(fraud_case.__file__).resolve().is_relative_to(Path.cwd())"],
                   cwd=target, env=env, check=True, creationflags=flags)
    report = ROOT / "artifacts/platform/packaged_pytest.xml"
    # Test geçici dosyaları bu dizinde kalsın; kullanıcının global pytest
    # temp klasörü başka bir oturuma ait olabilir
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                             "--basetemp", str(target / ".pytest-work"), "--junitxml", str(report)],
                            cwd=target, env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", creationflags=flags)
    log = result.stdout + result.stderr
    (report.parent / "packaged_tests.log").write_text(log, encoding="utf-8")
    print(log[-4000:], flush=True)
    result.check_returncode()
    suites = ET.parse(report).getroot().findall(".//testsuite")
    counts = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    assert counts["tests"] > 0 and counts["failures"] == counts["errors"] == 0
    result = {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files_verified": len(manifest),
              "crc_passed": True, "markdown_links_valid": True, "import_from_extracted_source": True,
              "tests": counts, "python": sys.version.split()[0],
              "scope": "Extracted source and installed dependencies; no Kaggle/model download or full pipeline rebuild."}
    (ROOT / "artifacts/platform/packaged_tests.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

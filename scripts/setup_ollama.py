"""Sürümü sabitlenmiş taşınabilir Windows kurulumu. İndirme ile çevrimdışı çalışma ayrı."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
VERSION = "v0.34.4"
ZIP_SHA256 = "535193f38f3344e5b08f5d1c171c31ce11aa17f0124ff69ae26d8ec7fe06fa62"
URL = f"https://github.com/ollama/ollama/releases/download/{VERSION}/ollama-windows-amd64.zip"


def install():
    folder = ROOT / ".tools/ollama"
    binary = folder / "ollama.exe"
    if binary.exists():
        return binary
    folder.mkdir(parents=True, exist_ok=True)
    archive = ROOT / ".tools/ollama-windows-amd64.zip"
    if not archive.exists():
        print(f"Downloading portable Ollama {VERSION} (1.46 GB) ...", flush=True)
        partial = archive.with_suffix(".partial")
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        partial.replace(archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != ZIP_SHA256:
        raise RuntimeError("Ollama release archive checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not (folder / name).resolve().is_relative_to(folder.resolve()):
                raise RuntimeError("Unsafe archive entry")
        bundle.extractall(folder)
    if not binary.exists():
        raise RuntimeError("Portable binary missing after extraction")
    return binary


def start(binary, base_url):
    client = httpx.Client(base_url=base_url, timeout=5, trust_env=False)
    try:
        response = client.get("/api/version")
        response.raise_for_status()
        print(f"Local Ollama already available: {response.json()}", flush=True)
        return
    except httpx.HTTPError:
        pass
    env = os.environ.copy()
    env.update(OLLAMA_HOST="127.0.0.1:11435", OLLAMA_MODELS=str(ROOT / "models/ollama"),
               OLLAMA_NO_CLOUD="1", OLLAMA_NUM_PARALLEL="1", OLLAMA_MAX_LOADED_MODELS="1",
               OLLAMA_CONTEXT_LENGTH="8192", OLLAMA_KEEP_ALIVE="30s")
    logs = ROOT / "artifacts/runtime"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "ollama.stdout.log").open("ab") as stdout, (logs / "ollama.stderr.log").open("ab") as stderr:
        process = subprocess.Popen([str(binary), "serve"], cwd=binary.parent, env=env,
                                   stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    (logs / "ollama.pid").write_text(str(process.pid))
    for _ in range(30):
        try:
            response = client.get("/api/version")
            response.raise_for_status()
            print(f"Hidden local Ollama started: {response.json()}", flush=True)
            return
        except httpx.HTTPError:
            if process.poll() is not None:
                # Baglanti hatasi beklenen sey; asil sorun surecin olmesi
                raise RuntimeError("Ollama stopped; inspect artifacts/runtime/ollama.stderr.log") from None
            time.sleep(1)
    raise RuntimeError("Ollama startup timed out")


def pull(config):
    # Ağ üzerinden bağlı Windows diskinde büyük dosya digest'i yavaş olabiliyor
    with httpx.Client(base_url=config["base_url"], timeout=1800, trust_env=False) as client:
        for model in (config["embedding_model"], config["generation_model"]):
            print(f"Pulling {model} ...", flush=True)
            previous = None
            with client.stream("POST", "/api/pull", json={"model": model, "stream": True}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    event = json.loads(line)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    percent = int(100 * event.get("completed", 0) / max(1, event.get("total", 1)))
                    status = (event.get("status"), percent // 10)
                    if status != previous:
                        print(f"{model}: {event.get('status')} {percent}%", flush=True)
                        previous = status
        tags = client.get("/api/tags").json()
    out = ROOT / "artifacts/runtime/models.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ollama_version": VERSION, "archive_sha256": ZIP_SHA256, "models": tags}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-only", action="store_true", help="Use installed files; do not download")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/rag.json").read_text())
    binary = ROOT / ".tools/ollama/ollama.exe" if args.start_only else install()
    if not binary.exists():
        raise RuntimeError("Run setup once before offline start")
    start(binary, config["base_url"])
    if not args.start_only:
        pull(config)


if __name__ == "__main__":
    main()

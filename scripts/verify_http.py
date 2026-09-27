"""Penceresiz açılan gerçek bir loopback Uvicorn sunucusunu doğrular."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from fraud_case.service import CONTEXT_PROFILES

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-profile", choices=list(CONTEXT_PROFILES), default="raw")
    profile = parser.parse_args().context_profile
    folder = ROOT / "artifacts/platform"
    folder.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "FRAUD_CASE_ROOT": str(ROOT), "FRAUD_CONTEXT_PROFILE": profile}
    suffix = "" if profile == "raw" else f"_{profile}"
    with (folder / f"http_server{suffix}.log").open("wb") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "fraud_case.api:app", "--host", "127.0.0.1", "--port", "18081"],
                                   cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        with httpx.Client(base_url="http://127.0.0.1:18081", trust_env=False, timeout=660) as client:
            for _ in range(40):
                if process.poll() is not None:
                    raise RuntimeError("Test API failed to start; inspect http_server.log")
                try:
                    health = client.get("/health")
                    if health.status_code == 200:
                        assert health.json()["context_profile"] == profile
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.25)
            else:
                raise RuntimeError("Test API did not become ready")
            results = {}
            for endpoint, payload in [
                ("/score", {"transaction_id": 3400481}),
                ("/rules/evaluate", {"transaction_id": 3400481}),
                ("/explain", {"transaction_id": 3400481, "include_rag": True}),
                ("/rag/query", {"question": "Anomali skoru dolandırıcılık olasılığı mıdır?"}),
            ]:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                results[endpoint] = {"status": response.status_code, "body": response.json()}
                print(f"HTTP {profile} {endpoint}: {response.status_code}", flush=True)
            assert results["/explain"]["body"]["rag"]["model_used"]
            explanation = results["/explain"]["body"]
            assert explanation["rag"]["answer_mode"] == "source_selection_with_engine_facts"
            facts = explanation["rag"]["authoritative_evidence"]
            assert facts["decision"] == explanation["rule_decision"]
            assert facts["winning_rule"] == explanation["winning_rule"]
            assert results["/score"]["body"]["winning_rule"] == "R03_daily_velocity"
            assert results["/score"]["body"]["context_profile"] == profile
            (folder / f"http_verification{suffix}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    finally:
        # Sadece bu çağrının başlattığı süreci kapat
        process.terminate()
        process.wait(timeout=20)


if __name__ == "__main__":
    main()

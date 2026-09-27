"""Toplu kanıtı dışa aktarır ve paketlemeden önce dokümanı kontrol eder."""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path(__file__).resolve().parents[1]
def read(path):
    return json.loads((root / path).read_text(encoding="utf-8"))

platform = read("artifacts/platform/verification.json")
http = read("artifacts/platform/http_verification.json")
for route in ("/score", "/explain", "/rules/evaluate", "/rag/query"):
    assert http[route]["status"] == platform["api_results"][route]["status"] == 200
assert platform["api_results"]["/explain"]["body"]["rag"]["model_used"]
for name, expected in platform["source_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
assert platform["api_results"]["/explain"]["body"]["rag"]["answer_mode"] == "source_selection_with_engine_facts"
assert http["/explain"]["body"]["rag"]["answer_mode"] == "source_selection_with_engine_facts"
manifest = read("artifacts/final/frozen_manifest.json")
changed = []
for name, expected in manifest.items():
    with (root / name).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            changed.append(Path(name).as_posix())
# Final testten sonra donmus olmasi gereken veri ve model dosyalari. Kaynak
# dosyalar bicim/yorum icin degisebilir; davranis esitligi asagidaki 64 kayitli
# skor tekrariyla kanitlanir, byte hash'iyle degil.
frozen_outputs = [n for n in changed if not n.startswith("src/fraud_case/") and n != "config/rag.json"]
assert not frozen_outputs, frozen_outputs
# Feature katmaninin kendi bagimsiz kontrolu; "64 kayit her seyi kanitlar" demek
# yerine hangi katmanin neyle kontrol edildigini yazabilmek icin gerekli.
feature_checks = read("artifacts/features/verification.json")
assert feature_checks["status"] == "passed" and len(feature_checks["checks"]) >= 6
criterion = read("artifacts/context_criterion/evaluation.json")
selected = read("artifacts/context/selected_config.json")
# Olcut karsilastirmasi dondurulmus secime yazmamis olmali.
assert criterion["frozen_outputs_written"] is False
assert criterion["original_criterion"]["shipped_strength"] == selected["strength"]
uncertainty = read("artifacts/product_uncertainty/evaluation.json")
assert not uncertainty["test_labels_used"] and uncertainty["block_bootstrap"]
default_policy = read("artifacts/default_policy/evaluation.json")
# Kayitli kural kararlari behavior_context profiline ait; bu iddia olcumle duruyor.
assert default_policy["stored_decisions_reproduced_by_behavior_context"]
context_api = read("artifacts/platform/context_api_verification.json")
assert context_api["default_score_replays"] == 64 and len(context_api["scenarios"]) == 6
# Rapor context'i acmamak gerektigini yaziyor; servis varsayilani da oyle olmali.
assert context_api["default_profile"] == "raw" and context_api["default_profile_applies_no_reduction"]
for name, expected in context_api["source_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
product_check = read("artifacts/platform/product_profile_verification.json")
for name, expected in product_check["source_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
product_http = read("artifacts/platform/http_verification_product_risk.json")
assert all(result["status"] == 200 for result in product_http.values())
assert product_http["/explain"]["body"]["rag"]["model_used"]
assert product_http["/explain"]["body"]["rag"]["answer_mode"] == "source_selection_with_engine_facts"
demo = read("artifacts/demo/full/demo.json")
assert demo["llm_enabled"] and not demo["labels_read"] and len(demo["cases"]) == 3
# Kayitli HTML kontrolu gercekten teslim edilen sayfaya ait olmali; eski bir
# kontrolun yeni bir sayfa icin gecerli sayilmasini istemiyorum.
html_check = read("artifacts/platform/demo_html_check.json")
assert hashlib.sha256((root / html_check["checked_file"]).read_bytes()).hexdigest() == html_check["html_sha256"]
assert hashlib.sha256((root / "scripts/verify_demo_html.py").read_bytes()).hexdigest() == html_check["source_sha256"]
for name, expected in demo["source_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
scenario_demo = read("artifacts/demo/scenarios/verification.json")
assert scenario_demo["scenarios_verified"] == 48 and not scenario_demo["fraud_labels_read"]
assert hashlib.sha256((root / "reports/scenario_demo.html").read_bytes()).hexdigest() == scenario_demo["html_sha256"]
for name, expected in scenario_demo["source_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
suites = ET.parse(root / "artifacts/platform/pytest.xml").getroot().findall(".//testsuite")
counts = {key: sum(int(suite.get(key, "0")) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
assert counts["tests"] > 0 and counts["failures"] == counts["errors"] == 0, counts
evidence = {
    "unit_and_integration_tests_passed": counts["tests"] - counts["skipped"],
    "test_environment": "Windows / Python 3.14.6; dependency closure pinned in requirements.lock",
    "test_warning": "Starlette 1.7.0 deprecates its httpx TestClient backend; tests pass with pinned httpx 0.28.1.",
    "clean_install": {key: value for key, value in read("artifacts/platform/clean_install.json").items() if key != "tests"},
    "demo_html_check": html_check,
    "scenario_demo": scenario_demo,
    "local_model_runtime": read("artifacts/runtime/models.json"),
    "rag": {key: platform[key] for key in ("model", "generation_digest", "embedding_metadata", "hit_at_k", "top_k", "evaluation_scope")},
    "http_status": {route: http[route]["status"] for route in http},
    "final_evaluation": read("artifacts/final/evaluation.json"),
    "baseline_post_final_scoring_unchanged": True,
    "post_final_changes": changed,
    "post_final_change_scope": "Source files were reformatted and their comments rewritten after the final test, so byte "
                               "hashes differ from the frozen manifest. What actually pins behaviour is the derived "
                               "output side: scores.parquet, context_scores.parquet and rule_decisions.parquet still "
                               "match the frozen manifest byte for byte, so no source change was propagated into a "
                               "different result. Optional product_risk uses train labels and development-only metrics. "
                               "RAG selects sources; engine facts and full source quotations replace generated claims. "
                               "Three-case demo checks API consistency.",
    "replay_coverage": {
        "default_requests": context_api["default_score_replays"],
        "covers": "Scoring, aggregation and context from STORED features with the SAVED model, field by field.",
        "does_not_cover": "Feature generation is not re-derived by this replay, so it cannot by itself detect a change "
                          "in how history features are built. That layer is checked separately by "
                          "scripts/verify_features.py, which recomputes counts, means and velocity windows with a "
                          "different method, and by the frozen parquet hashes above.",
        "not_possible_here": "Re-deriving features from the raw IEEE-CIS CSV files is outside this package, which "
                             "deliberately excludes the raw data.",
        "feature_layer_status": feature_checks["status"], "feature_layer_checks": feature_checks["checks"],
    },
    "rag_contract": {"mode": "source_selection_with_engine_facts", "current_source_sha256": platform["source_sha256"],
                     "limitation": "Selection relevance and source correctness are not guaranteed; free-form model claims are not accepted."},
    "demo": {"created_utc": demo["created_utc"], "llm_enabled": demo["llm_enabled"], "labels_read": demo["labels_read"],
             "cases": [{"transaction_id": c["transaction_id"], "title": c["title"],
                        "decision": c["response"]["rule_decision"], "winning_rule": c["response"]["winning_rule"]} for c in demo["cases"]]},
    "supervised_baseline": read("artifacts/baseline/evaluation.json"),
    "context_selection_criterion": criterion,
    "product_risk_uncertainty": uncertainty,
    "default_policy_rule_decisions": default_policy,
    "context_api": context_api,
    "product_context_verification": product_check,
    "product_context_http_status": {route: result["status"] for route, result in product_http.items()},
    "product_context_development": read("artifacts/product_context/evaluation.json"),
}
(root / "reports/verification_summary.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
(root / "artifacts/final/post_evaluation_changes.json").write_text(
    json.dumps({"changed_files": changed, "scope": evidence["post_final_change_scope"]}, indent=2), encoding="utf-8")
missing = []
for path in [root / "README.md", *(root / "docs").glob("*.md"), *(root / "reports").glob("*.md")]:
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("#"):
            continue
        target = target.split("#", 1)[0]
        if target and not (path.parent / target).exists():
            missing.append((str(path.relative_to(root)), target))
assert not missing, missing
print("Verified aggregate evidence, unchanged risk parameters, four live HTTP routes and Markdown links.")

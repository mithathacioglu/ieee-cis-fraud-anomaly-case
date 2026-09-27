"""Demo HTML'inin yapısını kendi içinde doğrular.

Bu kontrolü ayrı bir dosyaya taşımamın nedeni provenance: `reports/verification_summary.json`
içindeki her alanın bu depodaki bir betikten çıkması gerekiyor. Görsel bir tarayıcı
kontrolü yapmıyorum ve yaptığımı da yazmıyorum; burada ölçülen şey işaretlemenin
kendisi. Sayfanın gerçekten "iyi göründüğü" iddiası bu kontrolün kapsamı değil.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "artifacts/demo/full/index.html"


def check(html):
    anchors = set(re.findall(r'href="#([^"]+)"', html))
    identifiers = set(re.findall(r'id="([^"]+)"', html))
    return {
        "three_cards": len(re.findall(r'<article id="case-', html)) == 3,
        "anchor_links_valid": bool(anchors) and anchors <= identifiers,
        # Sayfa tek dosya olarak açılabilmeli: uzak bir yere bağlanmamalı.
        "external_assets": bool(re.search(r'(?:src|href)\s*=\s*"https?:', html)),
        "scripts_or_frames": bool(re.search(r"<(?:script|iframe)\b", html)),
        "utf8_replacement_characters": "�" in html,
        "doctype_and_charset": html.startswith("<!doctype html") and 'charset="utf-8"' in html,
        "visual_browser_check": False,
    }


def main():
    # Hash ham byte uzerinden: metni okuyup yeniden kodlamak satir sonlarini
    # degistirir ve teslim kontrolu ile ayni sayiyi vermez.
    raw = TARGET.read_bytes()
    result = check(raw.decode("utf-8"))
    for name in ("three_cards", "anchor_links_valid", "doctype_and_charset"):
        assert result[name], name
    for name in ("external_assets", "scripts_or_frames", "utf8_replacement_characters"):
        assert not result[name], name
    result.update(
        reason="Görsel tarayıcı kontrolü çalıştırılmadı. Kontrol edilen şey işaretlemenin yapısı: üç kart, "
               "kendi içinde çözülen bağlantılar, uzak varlık ve script yokluğu, bozulmamış UTF-8.",
        checked_file=TARGET.relative_to(ROOT).as_posix(),
        html_sha256=hashlib.sha256(raw).hexdigest(),
        checked_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    folder = ROOT / "artifacts/platform"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "demo_html_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

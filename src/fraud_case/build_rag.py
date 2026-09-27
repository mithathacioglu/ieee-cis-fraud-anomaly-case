"""Vektör index'ini kurar. policies.json veya kurallar değişince tekrar çalıştır."""

import argparse
import json
from pathlib import Path

from fraud_case.rag import OllamaClient, RagConfig, VectorIndex, load_documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    root = parser.parse_args().root.resolve()
    config = RagConfig(**json.loads((root / "config/rag.json").read_text()))
    client = OllamaClient(config)
    try:
        index = VectorIndex.build(load_documents(root), client)
        index.save(root / "artifacts/rag")
        print(json.dumps({"documents": len(index.documents), **index.metadata}, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()

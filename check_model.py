"""Load the assignment's model on CPU and record actual installed versions."""

import os
from pathlib import Path
import platform
from importlib.metadata import version

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    model = SentenceTransformer(MODEL, device="cpu")
    assert model.get_sentence_embedding_dimension() == 384
    assert callable(model.encode_document) and callable(model.encode_query)
    lines = [f"Python: {platform.python_version()}", f"Model: {MODEL}",
             "Device: cpu", "Embedding dimensions: 384", "Model loaded successfully."]
    lines += [f"{name}: {version(name)}" for name in ("sentence-transformers", "numpy", "torch", "transformers")]
    report = "\n".join(lines) + "\n"
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence" / "environment.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()

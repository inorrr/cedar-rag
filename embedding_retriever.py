"""Inspect real normalized embeddings and retrieve complete Cedar Lab passages."""

import argparse
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import platform
from importlib.metadata import version

# check_model configures the project-local model cache before importing the model.
from check_model import MODEL, SentenceTransformer
import numpy as np
from keyword_retriever import load_chunks, positive_k


class EmbeddingRetriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.model = SentenceTransformer(MODEL, device="cpu")
        # Encode documents once, then reuse this matrix for all questions.
        self.documents = self.model.encode_document(
            [chunk.text for chunk in self.chunks],
            normalize_embeddings=True, convert_to_numpy=True,
        )
        assert self.documents.shape == (6, 384)
        assert np.allclose(np.linalg.norm(self.documents, axis=1), 1, atol=1e-6)

    def retrieve(self, question, k=2):
        query = self.model.encode_query(
            question, normalize_embeddings=True, convert_to_numpy=True,
        )
        assert query.shape == (384,)
        assert np.isclose(np.linalg.norm(query), 1, atol=1e-6)
        # No rounding before ranking; every coordinate contributes.
        scores = self.documents @ query
        order = sorted(range(len(self.chunks)),
                       key=lambda i: (-float(scores[i]), self.chunks[i].id))
        return query, scores, order, order[:k]

    def inspect(self, question, k=2):
        query, scores, order, selected = self.retrieve(question, k)
        print(f"Model: {MODEL}\nDevice: cpu\nPython: {platform.python_version()}")
        for package in ("sentence-transformers", "numpy", "torch", "transformers"):
            print(f"{package}: {version(package)}")
        print(f"Document matrix shape: {self.documents.shape}")
        print(f"Question vector shape: {query.shape}")
        print("All 384 coordinates are used; only the first 12 are displayed.")
        print("Individual coordinates do not have reliable semantic labels.")
        for chunk, vector in zip(self.chunks, self.documents):
            print(f"{chunk.id}: " + " ".join(f"{x:.4f}" for x in vector[:12]))
            print(f"  vector length: {np.linalg.norm(vector):.8f}")
        print(f"\nQuestion: {question}")
        print("Query: " + " ".join(f"{x:.4f}" for x in query[:12]))
        print(f"  vector length: {np.linalg.norm(query):.8f}")
        print(f"\nAll scores (k={k}):")
        for i in order:
            print(f"{self.chunks[i].id}: {scores[i]:.8f}")
        print("\nSelected passages (no relevance threshold):")
        for i in selected:
            chunk = self.chunks[i]
            print(f"[{chunk.id}] {chunk.filename}\n{chunk.text}\n")

        top = order[0]
        products = query * self.documents[top]
        dot = np.dot(query, self.documents[top])
        cosine = dot / (np.linalg.norm(query) * np.linalg.norm(self.documents[top]))
        assert np.isclose(dot, cosine, rtol=1e-6, atol=1e-7)
        assert np.isclose(products[:12].sum() + products[12:].sum(), dot,
                          rtol=1e-6, atol=1e-7)
        print(f"Top passage calculation: {self.chunks[top].id}")
        print("First 12 coordinate products: " + " ".join(f"{x:.8f}" for x in products[:12]))
        print(f"First 12 subtotal: {products[:12].sum():.8f}")
        print(f"Remaining 372 subtotal: {products[12:].sum():.8f}")
        print(f"Full dot product: {dot:.8f}\nDirect cosine: {cosine:.8f}")
        print("Dot product and direct cosine agree within numerical tolerance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-k", "--k", type=positive_k, default=2)
    parser.add_argument("--output", type=Path, help="Save actual successful inspection output")
    args = parser.parse_args()
    retriever = EmbeddingRetriever()
    buffer = StringIO()
    with redirect_stdout(buffer):
        retriever.inspect(args.question, args.k)
    report = buffer.getvalue()
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

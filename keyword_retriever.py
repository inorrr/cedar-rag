"""Inspectable word-overlap retrieval using only the Python standard library."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


DOCS = Path(__file__).resolve().parent / "docs"
STOPWORDS = set("a an the is are to of in at for and or may can i it be".split())


@dataclass(frozen=True)
class Chunk:
    id: str
    filename: str
    text: str


def load_chunks(directory=DOCS):
    """Load paragraphs, keeping labels and filenames out of searchable text."""
    chunks = []
    for filename in ("access.txt", "training.txt", "booking.txt"):
        for paragraph in (directory / filename).read_text(encoding="utf-8").strip().split("\n\n"):
            match = re.fullmatch(r"\[([ATB][12])\] (.+)", paragraph.strip())
            if match is None:
                raise ValueError(f"Malformed paragraph in {filename}: {paragraph!r}")
            chunks.append(Chunk(match[1], filename, match[2]))
    if len(chunks) != 6 or {chunk.id for chunk in chunks} != {"A1", "A2", "T1", "T2", "B1", "B2"}:
        raise ValueError("Expected exactly six uniquely labeled Cedar Lab chunks.")
    return chunks


def words(text):
    """A set counts repeated words only once. No stemming or synonyms."""
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def retrieve(question, chunks, k=2):
    query_words = words(question)
    scored = []
    for chunk in chunks:
        matched = sorted(query_words & words(chunk.text))
        scored.append((chunk, len(matched), matched))
    ranked = sorted(scored, key=lambda row: (-row[1], row[0].id))
    selected = [row for row in ranked if row[1] > 0][:k]
    return ranked, selected


def positive_k(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("k must be at least 1")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-k", "--k", type=positive_k, default=2)
    args = parser.parse_args()
    ranked, selected = retrieve(args.question, load_chunks(), args.k)
    print(f"Question: {args.question}\nk: {args.k}\nAll scores:")
    for chunk, score, matched in ranked:
        print(f"{chunk.id}: score={score}; matched words={', '.join(matched) or '(none)'}")
    if not selected:
        print("No matching passages.")
    else:
        print("\nSelected passages:")
        for chunk, score, _ in selected:
            print(f"[{chunk.id}] {chunk.filename}; score={score}\n{chunk.text}\n")


if __name__ == "__main__":
    main()

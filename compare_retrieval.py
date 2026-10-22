"""Run the assigned comparisons; preserve student judgments on subsequent runs."""

import argparse
import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from embedding_retriever import EmbeddingRetriever
from keyword_retriever import retrieve

ROOT = Path(__file__).resolve().parent
QUESTIONS = [
    ("Q1", "May a trained graduate student operate the laser alone at 19:00?", "A1;A2"),
    ("Q2", "Can I use the laser without another person present?", "A2"),
    ("Q3", "How frequently must I refresh my laser qualification?", "T1"),
    ("Q4", "What is the laser wavelength?", "absent"),
]
FIELDS = ["question_id", "keyword_top_ids", "keyword_scores", "embedding_top_ids",
          "embedding_scores", "required_ids", "keyword_evidence_complete",
          "embedding_evidence_complete", "observation"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="Write results and evidence here without changing the source corpus")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    results_path = output_dir / "results.csv"
    if results_path.exists():
        with results_path.open(newline="", encoding="utf-8") as handle:
            if any(any(row.get(field) for field in FIELDS[-3:])
                   for row in csv.DictReader(handle)):
                raise SystemExit("results.csv contains judgments; move it aside before rerunning.")
    retriever = EmbeddingRetriever()
    output = StringIO()
    rows = []
    with redirect_stdout(output):
        for qid, question, required in QUESTIONS:
            ranked, selected = retrieve(question, retriever.chunks, 2)
            print(f"=== {qid}: {question} ===\nKeyword scores (k=2):")
            for chunk, score, matched in ranked:
                print(f"{chunk.id}: {score}; matched words={', '.join(matched) or '(none)'}")
            print("Selected keyword passages:")
            for chunk, score, _ in selected:
                print(f"[{chunk.id}] {chunk.filename}; score={score}\n{chunk.text}")
            if not selected:
                print("No matching passages.")
            print("\nEmbedding inspection:")
            retriever.inspect(question, 2)
            _, scores, _, indices = retriever.retrieve(question, 2)
            rows.append(dict(zip(FIELDS, [
                qid, ";".join(row[0].id for row in selected),
                ";".join(str(row[1]) for row in selected),
                ";".join(retriever.chunks[i].id for i in indices),
                ";".join(format(float(scores[i]), ".9g") for i in indices),
                required, "", "", "",
            ])))
    evidence = output_dir / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "retrieval-runs.txt").write_text(output.getvalue(), encoding="utf-8")
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    experiment = StringIO()
    with redirect_stdout(experiment):
        print("Q1 embedding experiment. Required evidence: A1 and A2.")
        for k in (1, 2, 3):
            _, scores, _, selected = retriever.retrieve(QUESTIONS[0][1], k)
            print(f"\nk={k}")
            for i in selected:
                chunk = retriever.chunks[i]
                print(f"[{chunk.id}] {chunk.filename}; score={scores[i]:.8f}\n{chunk.text}")
        print("\nStudent audit pending: evidence completeness, coverage, and what extra passages add.")
    (evidence / "k-experiment.txt").write_text(experiment.getvalue(), encoding="utf-8")
    for row in rows:
        print(f"{row['question_id']}: keyword={row['keyword_top_ids']} ({row['keyword_scores']}); "
              f"embedding={row['embedding_top_ids']} ({row['embedding_scores']})")
    print(experiment.getvalue())


if __name__ == "__main__":
    main()

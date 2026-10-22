# Cedar Lab RAG assignment

Student: Yinuo Zhao (working individually).

This teaching project compares word overlap with pretrained embeddings on six
fictional Cedar Lab passages. Retrieval runs, embedding understanding checks,
student comparison judgments, and generated answers with audit notes are saved.
The reflection is saved in `reflection.txt`; final repository checks and submission remain.

## Setup

Run from this directory. Python 3.13 is used for the local virtual environment.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python check_model.py
```

The first install and model download require internet access. The model check
uses CPU, requires no API key, and writes actual versions to
`evidence/environment.txt` only after successfully loading
`sentence-transformers/all-MiniLM-L6-v2`. Model files are cached in `.cache/`
and excluded from Git. The model was downloaded successfully in the student's
Terminal and verified on CPU. Installed versions: Python 3.13.9,
sentence-transformers 6.2.0, numpy 2.5.4, torch 2.14.1, transformers 5.20.2.
See `requirements-lock.txt` for the installed dependency snapshot. Once the
model is cached, prefix commands with `HF_HUB_OFFLINE=1` to run without network.

## Task 1

Before running Q1, record your prediction of the passages needed for a complete
answer. Then run:

```bash
.venv/bin/python keyword_retriever.py "May a trained graduate student operate the laser alone at 19:00?" -k 2
```

The keyword retriever needs only the standard library, so it works even while
the embedding dependencies are unavailable.

`load_chunks()` reads each paragraph and separates its ID and filename from its
text. `words()` lowercases text, extracts alphanumeric words, removes the exact
assignment stopwords, and uses a set to count repeated words only once.
`retrieve()` counts the intersection of each passage's words with the question's
words. It sorts by decreasing score, then alphabetically by ID, and selects up
to k passages with positive scores. The command prints all scores and matched
words, followed by selected complete passages. It does not generate answers.

## Task 2

After the model check succeeds, run:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python embedding_retriever.py "May a trained graduate student operate the laser alone at 19:00?" -k 2 --output evidence/embedding-inspection.txt
```

`EmbeddingRetriever.__init__()` encodes the six paragraph texts once into a
matrix. `retrieve()` encodes the question in the same space, multiplies the
matrix by that vector to compute six dot products, sorts the scores, and takes
the first k passages. No score threshold is used. `inspect()` displays actual
vectors, lengths, rankings, selected passages, and coordinate-product subtotals,
and checks the dot product against cosine computed directly using vector lengths.
Numbers are rounded only when printed. The actual Q1 output is saved in
`evidence/embedding-inspection.txt`: A2 and T1 were selected. The document
matrix is (6, 384), the question vector is (384,), and normalization and direct
cosine checks passed. Separate floating-point dot-product implementations can
differ slightly in their final digits because of accumulation order.

## Task 3

```bash
HF_HUB_OFFLINE=1 .venv/bin/python compare_retrieval.py --output-dir rerun
```

This reuses one document embedding matrix for all four questions, saves actual
output to `evidence/retrieval-runs.txt`, and records the Q1 k=1, 2, 3 experiment
in `evidence/k-experiment.txt`. `results.csv` holds selected IDs and scores in
rank order. Its judgment columns contain the student's recorded assessments.
This command writes fresh results with blank judgments under `rerun/`, preserving
the submitted results and audits. Omitting `--output-dir` targets the repository
root; the script refuses to overwrite results.csv once judgments have been entered.

## Task 4

The prompts in `evidence/q1-answer-prompt.txt` and
`evidence/q4-answer-prompt.txt` contain only the question, instructions, and
actual top-two embedding passages. The student confirmed running them in
separate fresh chats outside this project, using "5.6 medium".
Responses are preserved in `evidence/q1-answer.txt` and `evidence/q4-answer.txt`.
`evidence/task4-audit.txt` separates the student's overall judgment from the
assistant's claim-by-claim checks. Q1's no-alone conclusion is supported, but
the response does not explicitly acknowledge the missing access-hours rule.
Q4 correctly states that wavelength information is absent.

## Remaining work

- Rerun Q1 once with the revised prompt in `evidence/q1-answer-prompt-revised.txt`
  in a fresh chat with the same passages, then audit whether it acknowledges the
  missing access-hours information. The handout requires one revised-instruction
  run when an answer fails an audit check; the original answer is preserved.
- Finalize reproducibility instructions, push to GitHub, and submit the repo link.

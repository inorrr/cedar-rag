# Task 2 understanding checks

Student explanation of similarity versus answerability:

"Similarity might mean that this evidence is very similar to the question itself, but does not mean it contains the answer."

Actual vectors and similarity calculations are in embedding-inspection.txt.

Student explanation of dimensions (recorded from conversation):
- 6 is the number of passages in the corpus.
- 384 is the embedding dimensionality; each passage and question is represented
  by 384 numerical features capturing semantic information. Matching dimensions
  allow the vectors to be mathematically compared.

Assistant clarification: matching dimensions are necessary but not sufficient;
the vectors must also be in a compatible embedding space. This project uses
the same model for both roles. Individual features do not have reliable names.

Student explanation of selection (recorded from conversation):
Sort similarity scores in descending order and select the two passages with
the highest scores. Higher cosine similarity indicates stronger semantic alignment.
Implementation detail: exact score ties are broken by ascending chunk ID.

Student coordinate check:
"largest: 0.1357, smallest: -0.1167. I don't think we can tell what T1 means by any of these number alone."

Assistant clarification: meaning is represented by the pattern across the vector;
individual coordinates have no reliable standalone semantic label.

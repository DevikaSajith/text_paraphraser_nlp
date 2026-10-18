# Automatic Text Paraphraser

NLP course project. Rewrites text so it keeps its meaning but uses different words and structure.
Two generators, an automatic reranker, evaluation metrics, a CLI and a Streamlit demo.

## How it works

```
input text
   │ sentence split (NLTK punkt)
   ├──► Rule-based  : connective swap → contraction toggle → clause reordering
   │                  → POS tag → Lesk WSD → WordNet synonym → re-inflect (lemminflect)
   ├──► Neural      : T5 seq2seq paraphrase model (beam / diverse beam / nucleus sampling)
   ▼
candidate pool ──► reranker: score = α·semantic_similarity + (1−α)·PINC
                   (candidates under the similarity threshold are flagged / ranked last)
   ▼
top-n paraphrases + metrics
```

| Module | Purpose |
|---|---|
| `paraphraser/rule_based.py` | WordNet + Lesk synonym substitution and syntactic rewrites (no GPU, no model download) |
| `paraphraser/neural.py` | Lazy-loaded HuggingFace T5 paraphraser, 3 decoding strategies |
| `paraphraser/pipeline.py` | `ParaphraseEngine`: candidate generation, scoring, reranking |
| `paraphraser/evaluate.py` | PINC, self-BLEU, Jaccard, edit ratio, ROUGE-L, BLEU, semantic similarity |
| `cli.py` | `paraphrase` and `eval` commands |
| `app.py` | Streamlit web demo |
| `tests/` | pytest suite (neural model is mocked, so tests need no download) |
| `data/sample.csv` | 12 source/reference sentence pairs for a quick evaluation run |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng wordnet omw-1.4 stopwords
```
(The NLTK data is also downloaded automatically on first use.)
For rule-based only, `pip install nltk lemminflect` is enough.
The first neural run downloads the model (~900 MB for T5-base).

## Usage

```bash
# rule-based, 3 variants
python cli.py paraphrase --text "Because the weather was terrible, we cancelled the picnic." --method rule -n 3

# neural with diverse beam search
python cli.py paraphrase --file essay.txt --method neural --strategy diverse -n 3

# hybrid: pool neural + rule candidates and rerank
python cli.py paraphrase --text "The company announced a significant increase in profits." --method hybrid

# JSON output / reproducible rule-based output
python cli.py paraphrase --text "..." --method rule --seed 42 --json

# evaluate on a CSV (columns: source[,reference])
python cli.py eval --input data/sample.csv --method rule --output results.csv

# web demo
streamlit run app.py

# tests
pytest -q
```

Python API:

```python
from paraphraser import ParaphraseEngine
engine = ParaphraseEngine(seed=42)
for c in engine.paraphrase("The students were happy with the exam results.", method="rule", n=3):
    print(c.method, round(c.similarity, 2), round(c.pinc, 2), c.text)
```

## Evaluation methodology (for your report)

A paraphrase should be **faithful** and **different**, so no single metric is enough.

* **Semantic similarity** (adequacy): cosine similarity of `all-MiniLM-L6-v2` sentence embeddings.
  If `sentence-transformers` or the model is unavailable it falls back to bag-of-words cosine, which
  cannot see synonyms and so under-scores rule-based output. Check `semantic_backend` in the summary.
* **PINC** (Chen & Dolan, 2011): share of candidate n-grams (n=1..4) absent from the source. 0 = copy.
* **Self-BLEU** (vs. source), **Jaccard**, **edit ratio**: lower = more different from the source.
* **BLEU / ROUGE-L vs. references**: only when the CSV has a `reference` column. Note that these reward
  overlap with one particular gold paraphrase, so treat them as secondary for this task.

`data/sample.csv` is a tiny smoke-test set. For results worth reporting, evaluate on a standard dataset
such as MRPC, PAWS or Quora Question Pairs (use the paraphrase-labelled pairs) by exporting them to the same CSV format.

Suggested experiments: rule vs. neural vs. hybrid; the three decoding strategies; the effect of `replace_prob`
and `α`; an ablation with and without the syntactic rewrites; a small human rating of fluency and adequacy.

## Known limitations

* Rule-based output can be unnatural: Lesk word-sense disambiguation is weak, and synonyms differ in register.
* Clause reordering and connective swaps are regex heuristics; they only fire on simple patterns.
* The neural model can drift in meaning or drop details on long or technical sentences, which is what the reranker guards against.
* Sentences are paraphrased independently, so cross-sentence coherence (pronouns, discourse flow) is not modelled.
* Diverse beam search depends on your `transformers` version; the code falls back to sampling automatically if it is unavailable.

## Ideas for extension

Fine-tune T5 on PAWS/MRPC, add back-translation (en→de→en) as a third generator, use BERTScore, add a
fluency score from a language model to the reranker, or expose a REST API with FastAPI.

# EPL499 — shared starter kit (group agreement v1)

Read this together with the original assignment. This is shared infrastructure, not a completed submission. The model implementations are intentionally left to their assigned members. Do not put the provided datasets in the submission ZIP or Git repository.

## 1. Fixed assignment requirements

- Use only the supplied datasets. Each challenge has its own training and labeled validation (`test.csv`) split. Never fit a vectorizer, Word2Vec, scaler, feature selector or classifier on validation data. Do not combine training sets across challenges for the required models.
- Run five core approaches on each of A, B, C. Report classification_report and macro-F1 for all 15. Compare approaches and engineered feature contributions, analyze errors, and answer critical-thinking questions using actual experiments.
- Every model receives raw texts through `predict(texts)` and returns labels 0/1, with preprocessing/extraction inside the model. Reproduction must be automatic. Custom classes belong in separate `.py` files.
- Every member analyzes at least five wrong predictions (five total minimum, not five per challenge as an assignment requirement), reviews another member's code, and contributes to the report and presentation.

### Label meanings and expected counts (assignment page 1)

| Challenge | 0 | 1 | Train 0 / 1 | Validation 0 / 1 |
|---|---|---|---|---|
| A | Fully human-written | Human-written, AI-polished | 5250 / 5250 | 1125 / 1125 |
| B | Fully human-written | AI-written, machine-humanized | 5250 / 10500 | 1125 / 2250 |
| C | Human-written, AI-polished | AI-written, machine-humanized | 5250 / 10500 | 1125 / 2250 |

Never reinterpret label 0 as fully human for C. B and C have class imbalance; explain per-class performance, accuracy and macro-F1. Verify counts when real files arrive.

## 2. Group decisions (not professor-mandated)

Use Python 3.13.5 (starter tested on this version); use the pinned shared requirements. Use seed 42; for reproducible Gensim use `workers=1`, `seed=42` and start Python with `PYTHONHASHSEED=42`. Fix preprocessing and parameters in code and record them in the report. Run on the same dependency versions; exact numerical reproducibility across hardware is not guaranteed.

For the initial LR comparison, everyone calls `config.make_logistic_regression()` without overrides: `C=1.0`, `solver='lbfgs'`, `max_iter=1000`, `class_weight=None`, `random_state=42`, other options default. The mandatory NB baseline uses its exact prescribed settings, not this factory. The core representations may use different preprocessing where needed; report those differences and do not overclaim that the comparison isolates representations if other factors differ. Use train-fitted scaling if needed (especially engineered features). Additional variants (e.g., balanced class weights) are separately named and reported; do not silently replace the core run. If convergence warnings occur, investigate and agree on a common change or document the difference.

Input: `list[str]` raw texts and a one-dimensional NumPy integer label array for fit; `predict` returns a one-dimensional integer array (or list of integers) of the same length and order as the inputs. `fit` returns `self`. A no-argument constructor creates an unfitted model; optional constructor arguments are allowed for extra experiments. Never mutate inputs; handle empty or unknown-token texts rather than dropping them. `predict` must not refit. No global fitted state or data loading at import time. Use paths relative to the project or supplied directory, never a personal absolute path.

The CSV column names `text`, `label`, `id` (test ID required; training ID optional) are an assumption because actual CSVs were not supplied to the planner. The loader preserves raw text and IDs as strings, including leading zeros. If headers differ, change **only the shared loader by agreement**, not each model. No cleaning or shuffling in the shared loader. Model-specific tokenization/preprocessing belongs inside each model. Use the original rows and their ordering.

## 3. Ownership and filenames (do not rename)

| Member | File | Class | Runner name |
|---|---|---|---|
| 1 | models/baseline.py | BaselineNB | baseline |
| 1 | models/bow_lr.py | BoWLR | bow_lr |
| 1 | models/tfidf_lr.py | TfidfLR | tfidf_lr |
| 2 | models/word2vec_lr.py | Word2VecLR | word2vec_lr |
| 3 | models/features_lr.py | FeaturesLR | features_lr |

- Member 1: raw text baseline `CountVectorizer(max_features=10000, min_df=5)` + `MultinomialNB(alpha=1.0)`, all other parameters defaults. Expected validation macro-F1 approximately A 75%, B 79.8%, C 78.6% (not measured results). BoW + LR and TF-IDF + LR use `max_features=10000, min_df=5`; compare all three. No extra preprocessing for baseline.
- Member 2: train a fresh Word2Vec on each challenge's training texts, form document vectors, then fit LR. No pretrained embeddings. Choose and document tokenization, embedding parameters, pooling and out-of-vocabulary behavior. Suggested starting point (not mandatory): mean of known word vectors, 100 dimensions, window 5, min_count 2, skip-gram (`sg=1`), 10 epochs, zero vector if no known words. Run a controlled comparison (e.g., change only pooling or one Word2Vec setting) and compare with Member 1's actual outputs. Analyze at least five errors.
- Member 3: at least 10 original engineered features; define, justify, and compare contributions using controlled experiments (e.g., leave-one-feature-out, with all other settings unchanged). Fit scaling on training only if used. Analyze correlated features, e.g., text length vs word count.

Everyone owns only their assigned modules and member experiment scripts; change shared files by agreement. Suggested Git branches `member1`, `member2`, `member3` off the same starter commit. Do not commit datasets, venv, or generated caches. Share experiment results separately and include required results in final ZIP despite `.gitignore`.

## 4. Setup and execution

From the root folder, place the supplied CSVs in `data/A/`, `data/B/`, `data/C/` (each contains `train.csv`, `test.csv`). Install Python and create a virtual environment:

```sh
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For reproducibility set `export PYTHONHASHSEED=42` (macOS/Linux) or `$env:PYTHONHASHSEED='42'` (PowerShell) before running Python. Verify environment with `python -m unittest discover -s tests -v`.

Individual work: once your model module exists, the other members' modules are not needed for the selected run.

```sh
python main.py --model word2vec_lr --challenge A
python main.py --model word2vec_lr
python main.py --model features_lr
python main.py --model baseline
python main.py --model bow_lr
python main.py --model tfidf_lr
```

For each challenge, the runner instantiates a fresh model, passes only the training inputs to `fit`, and passes raw validation texts to `predict`. It never passes validation labels into the model.

Member 2 and Member 3 implement `experiments/member2.py` and `experiments/member3.py` with `run(data_dir, output_dir, challenges)` for controlled experiments and feature contribution analysis. Member 1 may add `experiments/member1.py` if additional runs are needed. See `experiments/_template.py`. Each hook should run its experiments without prompts, save metrics with `evaluate_and_save`, and save a comparison table. Do not use validation texts for fitting; use fresh models for variants. Run own extras with `python main.py --model word2vec_lr --extras` or `python main.py --model features_lr --extras`.

After integration, run `python main.py --extras` to reproduce the 15 core evaluations and member experiment hooks. Without `--extras` only the core approaches run. Use `--data-dir PATH` and `--output PATH` for alternate directories if necessary.

## 5. Outputs

`results/core/<model>/<challenge>/` contains `metrics.json`, `classification_report.txt`, `predictions.csv` (exactly id,label), `errors.csv` (id,true_label,predicted_label,text). `results/summary_<model>_<challenge>.csv` records each selected core run. Full run writes `results/comparison.csv` (A/B/C/average), `results/selected_models.csv`, and `results/predictions_A.csv`, `predictions_B.csv`, `predictions_C.csv`.

Group decision: final prediction files default to the best **core** model per challenge by validation macro-F1 (unrounded); ties use registry order. The assignment doesn't specify which model supplies the files. Agree on the policy before submission; if you select an extra model or one common model instead, update the runner and document it. The labeled test split is validation used for model selection, not an untouched final test; do not present selection scores as independent generalization results. Mean macro-F1 across A/B/C uses equal challenge weights. `errors.csv` contains actual texts, so keep it local or within the authorized group; do not include raw error dumps in the submission if that would redistribute datasets. Include selected, necessary examples in the report as appropriate.

## 6. Final handoff checklist

- All five model files; automated core + extras; requirements.txt; README updated with real member names/student IDs and instructions.
- All required results/comparison tables, final three id,label prediction files; report up to 8 pages, individual contribution table, and presentation (10 min plus 5 min total individual questions).
- Each member records at least five actual error analyses with challenge, ID, actual/predicted label, observed text cues, and plausible explanation (not proof). Review another member's code and record that contribution.
- Source discrepancy: section 5 lists four numbered critical-thinking questions, while section 7 says answer six. Ask the instructor whether two are missing; do not invent them.
- Remove datasets from final `EPL499_<groupname>.zip`; do not submit this starter unmodified. Confirm deadlines via Blackboard/instructor.

Status: shared code tested with synthetic data only; actual CSVs, real model scores and member modules are not included. The original assignment is the authority if any group convention conflicts with it.

# Copy-paste this prompt into each member's LLM

Attach the original assignment PDF and the extracted shared starter kit (or ZIP if supported) along with this prompt. Replace [MEMBER NUMBER] below. Work from the same starter version; do not regenerate incompatible common infrastructure.

---

You are helping me implement my part of our EPL499 group assignment, Human-AI Text Classification: 3 Binary Challenges. I am **member [MEMBER NUMBER]** (1, 2, or 3). Read the original assignment and the shared starter kit README, config.py, data_loader.py, evaluation.py, model_contract.py, main.py, and templates before writing code. The assignment takes precedence; the starter specifies our agreed engineering conventions. We are developing independently and merging later. Help me understand and explain the code, not just produce it.

## Non-negotiable integration contract

- Only supplied `data/A/train.csv`, `data/A/test.csv`, and analogous B/C files. Train each challenge separately. `test.csv` is labeled validation. Never fit representations, embeddings, scalers, feature selectors or classifiers on validation; no external data or pretrained embeddings for mandatory Word2Vec. Do not create a different split for core evaluation.
- Use the original raw text order and numeric labels. A: 0 human, 1 human AI-polished; B: 0 human, 1 AI-written machine-humanized; C: 0 human AI-polished, 1 AI-written machine-humanized. Do not rebalance or drop rows in core runs.
- All models must have a no-argument constructor, `fit(self, texts, labels)` returning self, and `predict(self, texts)` accepting raw text list and returning 1D integer labels 0/1 in input order. No model-level loading of CSVs, no fit in predict, no code executing on import. Fit only on arguments. Preprocessing/feature extraction stays inside the model. Do not mutate input data; handle empty/OOV documents. Extra constructor options are allowed.
- Import `make_logistic_regression` and `SEED` from `config.py`. Core LR uses the factory without overrides (C=1, lbfgs, max_iter=1000, class_weight=None, random_state=42). NB baseline is the prescribed exact configuration. Seed 42; Word2Vec workers=1 and seed=42; set PYTHONHASHSEED=42 before running. Extra variants named separately; document differences.
- Use `load_data(challenge, data_dir)` from data_loader.py and `evaluate_and_save(data, predictions, output_dir)` from evaluation.py for experiments. Do not recreate these or change shared files, registry, dependency versions, file names, or class names without group agreement. Run core model through main.py. Ask me for actual CSV headers/data or inspect local files if needed; do not fabricate results. Assume the shared loader's text/label/id headers until verified. Use pinned requirements in the shared environment.

## Implement only the files for my member number

| Member | File | Class | CLI model name |
|---|---|---|---|
| 1 | models/baseline.py | BaselineNB | baseline |
| 1 | models/bow_lr.py | BoWLR | bow_lr |
| 1 | models/tfidf_lr.py | TfidfLR | tfidf_lr |
| 2 | models/word2vec_lr.py | Word2VecLR | word2vec_lr |
| 3 | models/features_lr.py | FeaturesLR | features_lr |

Member 1: mandatory exact baseline `CountVectorizer(max_features=10000, min_df=5)` and `MultinomialNB(alpha=1.0)` with remaining parameters default, no additional preprocessing. Build CountVectorizer + LR and TfidfVectorizer + LR using max_features=10000, min_df=5. Compare results on A/B/C. Expected baseline scores are reference values, not actual results.

Member 2: implement Gensim Word2Vec trained exclusively on the training texts of each challenge, document embeddings and LR. Suggested core starting point: mean pooling, vector_size=100, window=5, min_count=2, sg=1, epochs=10, workers=1, seed=42, zero vector for documents with no in-vocabulary tokens. Explain tokenization and choices. These Word2Vec settings are recommendations, not professor requirements. Implement `experiments/member2.py` with `run(data_dir, output_dir, challenges)` for at least one controlled comparison (e.g. pooling or a single parameter), write comparison tables, and compare to actual BoW/TF-IDF outputs when available.

Member 3: implement at least ten original engineered features plus LR; document definitions and rationale for each. If scaling, fit scaler on training only. Implement `experiments/member3.py` with `run(data_dir, output_dir, challenges)` to evaluate feature contributions (e.g. leave-one-feature-out), save results and discuss correlations. No artificial performance claims.

Member 1 can optionally add `experiments/member1.py` using the same hook for any additional experiments. Required member2 and member3 hooks should run on supplied challenges without interactive input, save results under supplied output_dir using shared evaluation and produce comparison CSVs. Keep core model settings fixed; each variant starts unfitted. Do not implement other members' models unless we explicitly agree.

## Deliverables for my part

1. Explain architecture and give complete working Python file(s) for my assigned module(s), and experiment hook where applicable. Follow the shared registry exactly. Include small interface tests, e.g. fit then predict raw texts; no other members required. If datasets are unavailable, use synthetic examples only for smoke tests and label them as such.
2. Give commands for running my models on A/B/C. Core runner saves classification_report, macro-F1, accuracy, per-class metrics, predictions and errors. Explain actual metrics only when outputs are available; discuss imbalance in B/C and what changes across challenges. No invented scores.
3. Help me analyze at least five actual misclassified examples, review another member's code, and prepare my individual report/presentation explanation. Keep notes of hyperparameters, model assumptions, tradeoffs, and contribution. Do not invent examples or claim review that hasn't happened.
4. Tell me explicitly about any conflicts with the assignment, assumptions, missing files/dependencies or changes needed to the common contract; propose changes for group approval instead of silently altering shared code. The assignment lists four critical-thinking questions but refers to six elsewhere; flag this for instructor clarification.

Start by inspecting the attached starter and assignment; then work on my assigned files only. If the actual dataset is not accessible, produce code and clearly say that real scores cannot be validated until it is supplied.

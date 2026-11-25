# EPL499 course coding reference

**Scope:** University of Cyprus, Introduction to Natural Language Processing — group project, Human-AI Text Classification: 3 Binary Challenges. This is a source-grounded reference for coding assistants, not an assignment implementation. No external tutorials or documentation consulted; no notebook execution or new experimental results. Companion: [LAB_CODE_EXAMPLES.md](LAB_CODE_EXAMPLES.md).

## 1. Evidence rules and source inventory

Evidence categories (can overlap): **A** = demonstrated in supplied course code; **B** = described/theoretical, not demonstrated as code for that technique; **C** = explicit assignment requirement; **D** = reasonable assignment adaptation or group convention, not a professor requirement; **E** = not found in supplied teaching material (not necessarily forbidden). `L8 c60` means Lab 8 solution notebook, cell 60, counting all Markdown/code cells from 1 in physical notebook order, not execution counts. `L7 p12` means PDF page 12 (1-based). `P` means assignment PDF, `R` README.md, `H` LLM_HANDOFF.md. All filenames below are exact.

| File | Extent | Topics / role |
|---|---:|---|
| `LAB1_Brief_Introduction_to_Python_and_Jupyter_Notebooks.pdf` | 70 pages | Jupyter, Python basics, functions, loops, comprehensions, text operations |
| `LAB1_solutions.ipynb` | 110 cells; 61 nonempty code cells | Jupyter, Python basics, functions, loops, comprehensions, text operations |
| `LAB2_Regular Expressions_and_Pre-processing_with_NLTK.pdf` | 55 pages | Regex, cleaning, tokenization, normalization, vocabulary |
| `LAB2_solutions.ipynb` | 126 cells; 72 nonempty code cells | Regex, cleaning, tokenization, normalization, vocabulary |
| `LAB3_Application_of_Fundamental_NLP_Tasks.pdf` | 16 pages | POS, NER, sentiment, constituency and dependency parsing |
| `LAB3_solutions.ipynb` | 76 cells; 34 nonempty code cells | POS, NER, sentiment, constituency and dependency parsing |
| `LAB4_Language_Modeling_with_NLTK.pdf` | 31 pages | N-gram language modeling, smoothing and generation |
| `LAB4_solutions.ipynb` | 80 cells; 50 nonempty code cells | N-gram language modeling, smoothing and generation |
| `LAB5_Text_Classification.pdf` | 40 pages | Pandas, BoW, MultinomialNB, evaluation, CV |
| `LAB5_solutions.ipynb` | 116 cells; 67 nonempty code cells | Pandas, BoW, MultinomialNB, evaluation, CV |
| `LAB6_Text_Classification_2.pdf` | 21 pages | Lexical/linguistic features, feature merging, RF selection, NB |
| `LAB6_solutions.ipynb` | 114 cells; 68 nonempty code cells | Lexical/linguistic features, feature merging, RF selection, NB |
| `LAB7_Text_Classification_and_Dimensionality_Reduction_with_SVD.pdf` | 19 pages | TF-IDF, SGD logistic regression, SVD |
| `LAB7_solutions.ipynb` | 88 cells; 53 nonempty code cells | TF-IDF, SGD logistic regression, SVD |
| `LAB8_solutions.ipynb` | 103 cells; 62 nonempty code cells | Word2Vec, pretrained and own embeddings, mean pooling, RF |
| `LAB9_Feed_Forward_Neural_Networks_with_Keras.pdf` | 24 pages | BoW, SGD logistic regression, Keras, overfitting |
| `LAB9_solutions.ipynb` | 71 cells; 34 nonempty code cells | BoW, SGD logistic regression, Keras, overfitting |
| `Text_Classification_extra_notes.pdf` | 4 pages | Abstract split/fit/transform pipeline |
| `ΕΠΛ499 Group Project - Human, AI, and AI-assisted Text Generation Detection.pdf` | 4 pages | Assignment authority: tasks, rules and deliverables |
| `README.md` | Markdown | Group starter documentation; not course teaching evidence |
| `LLM_HANDOFF.md` | Markdown | Group starter documentation; not course teaching evidence |

**Coverage:** See §12 for the second-pass per-document coverage and correction log. 21 supplied source files: 9 solution notebooks, 8 lab PDFs, one extra-notes PDF, one assignment PDF, two starter documents. Lab 8 slides were not supplied; Lab 8 notebook is accessible and is the primary embedding evidence. No actual starter `.py` files, requirements.txt, tests, CSVs or separate lecture slides were supplied. All source files were opened locally. PDF text was extracted and relevant visual diagrams/screenshots examined, but not every slide or notebook output image was independently transcribed at full resolution; use the source if an exact visual detail is needed. Notebook cell source is authoritative for verbatim code. No code was executed.

## 2. Assignment requirements (C)

P p1: data adapted from PAN 2025 Subtask 2 (the superscript 1 is a footnote, not part of the subtask number). Use only the supplied datasets, separately per challenge. Machine-humanized means machine processing, not human editing. `test.csv` contains labels and is explicitly validation for evaluation/model selection, not an independent blind test. Text shared across challenges stays in the same split. Do not pool training sets for mandatory models.

| Challenge | Label 0 | Label 1 | Train 0 / 1 | Validation 0 / 1 |
|---|---|---|---:|---:|
| A | Fully human-written | Human-written → AI-polished | 5250 / 5250 | 1125 / 1125 |
| B | Fully human-written | AI-written → machine-humanized | 5250 / 10500 | 1125 / 2250 |
| C | Human-written → AI-polished | AI-written → machine-humanized | 5250 / 10500 | 1125 / 2250 |

P p2: all five approaches below for all three challenges, 15 core runs. Fit classifiers and representations on training only. Report `classification_report` and macro-F1 for each. Expected baseline macro-F1 A ~75%, B ~79.8%, C ~78.6%, mean ~77.8% are **assignment reference values**, not measured results. Beating them is not necessary for full credit (P p4).

| Mandatory approach | Course mapping and evidence | Assignment-specific differences |
|---|---|---|
| Count BoW + MultinomialNB | A: L5 p29–33, c47–76; L6 c40–45, 90–96 | C: exact `CountVectorizer(max_features=10000, min_df=5)`, `MultinomialNB(alpha=1.0)`, other defaults, no additional preprocessing. |
| Count BoW + LR | A: BoW L5; SGD logistic regression directly on BoW L9 c33–46 | C: max_features=10000, min_df=5; D: common sklearn LR factory, not SGD/Keras. |
| TF-IDF + LR | A: L7 p9–13, c36–44, 65–75 (actual SGD calls c68 and c73) | C: max_features=10000, min_df=5; no SVD needed; D: common LR factory. |
| Word2Vec + LR | A: L8 c23–28, 59–78, 90–100 (Random Forest); LR L7 c65–70 (c68 classifier) and L9 c42–46 (c45 classifier) | C: train own embeddings on each challenge's training texts, document representations, LR, compare; D: compose these taught steps with shared LR and OOV fallback. No complete Word2Vec+LR lab pipeline found. |
| ≥10 engineered features + LR | A: L6 c49–86 feature engineering; L3 POS/NER/sentiment; L2 text stats; L7/L9 LR | C: ≥10 own features with definition/rationale, compare contributions; D: feature-only matrix, shared LR, optional train-fitted scaling and ablation. |

P pp2–3: suggested member allocation matches group; every member contributes across A/B/C, analyzes ≥5 actual wrong predictions total, reviews another member's code, participates in report and presentation. Member 2 contribution includes a controlled comparison. Critical questions: best model/difficulty across challenges (at least two evidence-backed explanations), imbalance and per-class metrics, engineered feature contributions/correlations, count vs TF-IDF vs Word2Vec tradeoffs. P p3 optional competition uses equal-weight average of three macro-F1 scores; optional tuning/extra representations/classifiers do not replace core; pretrained transformers or LLMs as classifiers are prohibited. P p4: five equal 20-point criteria: pipeline/reproducibility, experiment design/comparisons, feature engineering/contributions, error analysis/critical documentation, presentation/individual understanding; final 20 points assigned individually. Automatic reproducibility, code in notebooks or `.py`, custom classes in separate `.py`, raw-text `predict(texts)` -> 0/1, requirements.txt and README including names/student IDs, results and `predictions_A.csv`, `predictions_B.csv`, `predictions_C.csv` with exactly `id,label`. Report ≤8 pages; presentation 10 minutes + 5 minutes individual questions total; no supplied datasets in ZIP. Deadline announced separately. P p3 lists four numbered critical questions while p4 says six; ask instructor, do not invent two.

## 3. Python level and text preprocessing

### Python foundations (A)
L1 p3–9 (Jupyter code vs Markdown, restart/run all, Anaconda/Colab), p12 (NLTK tokenization), p14–31 (types, collections, indexing/slicing, methods), p33–48 (indentation, comments, branches, loops, break/continue), p50–63 (built-ins, simple functions, comprehensions, import forms); c8–63 and c67–109. Use plain lists/dicts/sets, `len`, `sum`, `range`, `print`, `type`, `int`, `float`, membership, `.split`, `.replace`, `.join`, `.append`, `.count`, `.items`, simple return values. `math.sqrt`, `math.pi`, `random.randint`, `random.seed(123)` c61–63, 104–109. Lab1 c29 deliberately raises KeyError for missing dictionary key. Prefer named functions and readable intermediate variables over framework-heavy implementations. The course examples are procedural notebooks with a few helpers, not a large class architecture. Minimal stateful class wrappers are D/C (raw-text predict and separate custom classes required; fit interface group-defined).

### Regex and cleaning (A/B)
L2 p3–11, c4–45: `re.findall`, `match`, `fullmatch`, match `.group`, `sub`, `split(maxsplit=1)`, `re.MULTILINE` demonstrated; `IGNORECASE` is demonstrated in slide code p11; `DOTALL` is described on p11 (B). Character classes, alternation, negation, anchors, `? * + .`, `\b \d \s \w` and complements. L2 p4 `a^b` example is misleading for literal caret in Python; escape literal caret rather than silently changing course source. Unicode behavior of shorthand classes is broader than the simplified ASCII examples. L2 p14–20/c53–55: expand selected contractions via `uncontract` with `re.sub` and backreferences; `contractions.fix` shown on p19 (not needed for group). URL/email removal or conversion advised p20 (B; no implementation located there). Regex uncontract helper is not exhaustive and primarily handles ASCII apostrophes. Do not automatically use all cleaning in every model.

### Tokenization and normalization (A)
L2 p21–32/c59–66: `str.split`, `nltk.tokenize.word_tokenize`, `sent_tokenize`, downloads `punkt_tab`/`punkt`; sentence splitting by punctuation has limitations. L2 p33–53/c69–118: lowercase comprehension, `string.punctuation`, `filter(lambda ...)`, `isdigit`, `isalpha`, two punctuation/digit removal alternatives (only punctuation/digit tokens vs all nonalphabetic tokens), NLTK stopwords, `PorterStemmer().stem`, `WordNetLemmatizer().lemmatize`, `wordnet`/`omw-1.4`, `len(set(tokens))`, `collections.Counter(...).most_common`. WordCloud and matplotlib visualization c103–104 are optional exploration, not core feature requirements. L2 c122–125 applies preprocessing across tweets and compares vocabulary.

L4 p9–15/c11–12: language modeling uses tokenization, lowercase, lemmatization; retains punctuation and stopwords for natural language. L5–7 and L9 classification examples expand contractions, tokenize, lowercase, keep alphabetic tokens, remove stopwords and lemmatize; L8 intentionally does not stem/lemmatize in its embedding workflow. All apply same chosen steps to train and unseen inputs. Exact baseline overrides lab preprocessing entirely (C). Authorship cues such as punctuation, digits, contractions, stopwords, casing and spacing may matter; preserving them in raw-text engineered features or controlled Word2Vec variants is D, not an established performance claim. Keep raw text available.

## 4. Data handling and classical classification

### Data, splits, fit/transform (A/C)
L5 p3–14: `pd.Series`, `pd.DataFrame`, indexing `loc`/`iloc`, `head`, `tail`, `shape`, `columns`, `dtypes`, `describe`, `mean/sum/min/max`, `plot.hist`, `read_csv`, `to_csv(index=False)`, `copy`, `concat`, `merge`, `join`, `astype`, `apply`, `map`, `replace`. L6 p3–5: merge by index or explicit id, DataFrame from dictionary-of-lists or list-of-dictionaries. L5–9 notebooks use NLTK `movie_reviews` dataset, `fileids`, `categories`, `raw`, labels based on pos prefix. These are lab data only, forbidden as extra training data for mandatory project models. L5 p28/L8 c17–20: `train_test_split(..., test_size=0.20, random_state=123)` examples; L8 uses parallel text/label lists directly. Assignment provides split, so do not reproduce lab split for core. Extra notes p3–4 diagram explicitly splits first, applies same preprocessing, fits vectorizer only on training, transforms both, fits classifier on X_train/y_train, predicts X_test, evaluates against true test labels. Diagram's prediction box appears to label prediction `y_test` while final evaluation uses `y_pred`: treat as an illustrative naming inconsistency, not an instruction to overwrite true labels.

### Count-based BoW (A/C)
L5 p29–32/c47–59; L6 c40–45; L9 c33–40. `CountVectorizer` turns documents into token counts, rows documents, columns vocabulary; unigram `(1,1)` and bigram `(2,2)` with `max_features=1000`, `lowercase=False`, `tokenizer=None`, `preprocessor=None`, `stop_words=None`, `min_df=10`, `max_df=0.80` in lab; preprocessed tokens joined with `' '.join(tokens)`. `fit_transform` on training and `transform` on held-out; `.toarray()`, `pd.DataFrame`, `.get_feature_names_out()` for inspection. L5 p31 screenshot shows 1600 training rows × 1000 columns. `get_top_ngrams` sums columns and sorts descending (c57). P p2 explicitly changes parameters; do not copy lab `max_df=0.80` into baseline. Keeping sparse matrices for LR/NB instead of densifying is a simple D adaptation to avoid memory overhead at 10,000 columns; not a new modeling technique. Baseline no custom tokenization/lowercasing overrides; other defaults remain default.

### TF-IDF (A/C)
L7 p9–11/c36–44 uses `TfidfVectorizer`, same lab parameters as BoW, unigram and bigram runs, `fit_transform` then `transform`, dense DataFrame for inspection. TF-IDF reduces the influence of common terms across documents; compare with raw counts. Assignment fixes max_features and min_df, not all lab parameters; record actual preprocessing, other options. Do not fit IDF or vocabulary on validation.

### MultinomialNB (A/C)
L5 p33/c62–76: `MultinomialNB()` → `.fit(X_train,y_train)` → `.predict(X_test)`. L6 c90–94 demonstrates failure with negative `sentiment_difference`, drops it for NB. Counts must be nonnegative; never apply centering that creates negative values to NB. Project exact baseline alpha=1.0, no extras; lab leaves defaults implicit.

### Logistic Regression (A/C/D)
L7 p12–13/c65–85, L9 c42–46: `from sklearn.linear_model import SGDClassifier`; `SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, random_state=123)`. This is the **explicit sklearn logistic regression training approach** shown in supplied labs. L9 p5–12/c49 also demonstrates sigmoid single-neuron Keras logistic regression, `Sequential`, `Dense(1, input_dim=X_train.shape[1], activation='sigmoid')`, `SGD(learning_rate=0.01)`, `binary_crossentropy`, `epochs=50`, `batch_size=16`, threshold `>0.5`. L9 p6 diagram shows weighted sum, sigmoid, threshold, class. The slide describes equivalence in model form, not identical optimizer/regularization. No direct executable `sklearn.linear_model.LogisticRegression(...)` example was found; group uses it via `config.make_logistic_regression()` (D), valid interpretation of assignment LR, not professor's exact implementation. Do not switch to SGDClassifier in core without group approval.

### SVD and feature selection (A, optional)
L7 p16–17/c47–62: `TruncatedSVD(n_components=50, random_state=42)`, train fit_transform, test transform; `explained_variance_ratio_`, sum, `singular_values_`, compare shapes, new features less interpretable. Not required for core. L6 p14–15/c99–112: `RandomForestClassifier(n_estimators=100, random_state=123)`, `feature_importances_`, DataFrame and sort/head top100, `get_top_features`, `barh` top20, compare NB using all nonnegative features vs top100; excludes sentiment_difference for NB. RF importances are not the same as causal contribution or LR feature weights. Feature selection must be train-only; do not let full-train selection leak within CV folds.

## 5. Word2Vec — Member 2 primary reference

**Direct evidence:** `LAB8_solutions.ipynb`, c1–103. This is the most relevant course source; no Lab8 PDF supplied. It explicitly demonstrates both loading pretrained GoogleNews vectors and training own Word2Vec. Only the latter belongs in mandatory project implementation. The original notebook uses RF, not LR; c2 explicitly states this, c80 requests all three Random Forest runs, and c85/91/97 instantiate RF; substituting group LR is the required project adaptation.

| Stage | Evidence | Exact lab pattern / parameters |
|---|---|---|
| Split | A c17–20 | `train_test_split(reviews, labels, test_size=0.20, random_state=123)`; use existing splits instead for project |
| Resources | A c23–24 | NLTK `word_tokenize`, `stopwords`, `re`, `string`; also imports stemmers and Gutenberg (not necessary to active embedding path) |
| Preprocess | A c25–28 | `uncontract`, `tokenize_text`, `lowercase_tokens`, `remove_punctuation_digits` (`isalpha`), `remove_stopwords`; returns tokens. No stemming/lemmatization. c28 explicitly gives the absence of stemming/lemmatization as a choice for the pretrained model; own training reuses these tokens. This is not a statement that stemming or lemmatization is inherently incompatible with custom Word2Vec. |
| Pretrained demo | A c32–55 | `gensim.downloader as api`, `api.load('word2vec-google-news-300')`, `key_to_index`, vector lookup, `most_similar(positive=...,negative=...)`, 300 dims. **Not permitted for mandatory project.** |
| Train own | A c59–60 | `from gensim.models import Word2Vec`; `Word2Vec(sentences=clean_tokens_train, vector_size=100, window=5, min_count=1, sg=1, epochs=10, seed=123)` |
| CBOW vs skip-gram | B c60/74 comments; A skip-gram | `sg=1` target predicts surrounding context; `sg=0` context predicts target (CBOW). CBOW is explained but no run with `sg=0` found. |
| Inspect vocabulary | A c61–65 | `model.wv.key_to_index.keys()`, `model.wv['character']`, `model.wv.most_similar('character')`, `cosine_similarity([X,Y])` |
| Save/load | A c67–68 | `model.save('my_word2vec.model')`, `Word2Vec.load(...)` (not required for project) |
| OOV filtering | A c70 | `if t in model.wv:` then append `model.wv[t]`; repeated known tokens contribute repeatedly. c49 explicitly lists missing pretrained tokens. |
| Document mean | A c43–55,70–71,75–76 | `import numpy`, `numpy.mean(document_vectors, axis=0)`; same trained model for train/test. Mean fixed-width vector if at least one known token. |
| Compare dimensions | A c73–78 | retrain `model300` with 300 dimensions, same other explicit params; new mean helper and compare similarities. |
| Classify/evaluate | A c81–102 | `X_train = train_vectors_our_model`, `X_test = test_vectors_our_model`, RF 100 trees seed123, fit/predict, classification report, confusion matrix; repeat pretrained/100/300; compare to previous labs. No pandas conversion needed (c82). |

### Saved Lab 8 evaluation evidence (not project results)

The notebook contains saved output, so the width comparison is more than an exercise prompt. L8 c87/88 (pretrained vectors), c93/94 (own 100-dimensional vectors), and c99/100 (own 300-dimensional vectors) show Random Forest classification reports and confusion matrices on 400 movie reviews. The original rounded macro-F1 rows are 0.73, 0.74 and 0.76 respectively. Matrix rows (actual 0, 1) and columns (predicted 0, 1) are `[[151,56],[52,141]]`, `[[161,46],[57,136]]`, and `[[162,45],[52,141]]` respectively. These are transcribed saved lab outputs, not rerun or independently reproduced results. They demonstrate the comparison design and per-class evaluation, not that 300 dimensions will outperform 100 on the EPL499 human/AI dataset. Pretrained vectors remain disallowed for the mandatory Word2Vec approach.

### Lab vs group parameters — never conflate

| Parameter | Lab8 c60 | Lab8 c74 | Group R §2–3 / H Member2 |
|---|---:|---:|---|
| vector_size | 100 | 300 | 100 suggested |
| window | 5 | 5 | 5 suggested |
| min_count | 1 | 1 | 2 suggested |
| sg | 1 | 1 | 1 suggested |
| epochs | 10 | 10 | 10 suggested |
| seed | 123 | 123 | 42 convention |
| workers | not specified | not specified | 1 convention for reproducibility |
| pooling | mean | mean | mean suggested |
| empty/all-OOV fallback | absent | absent | zero vector suggested; handling empty/OOV inputs required by group contract |

P p2 fixes none of these numerical Word2Vec settings. The lab's `min_count=1` includes every observed token; `2` excludes singleton tokens (possible noise reduction, but can worsen coverage). `vector_size` controls document vector width, `window` context, `epochs` training passes, `sg` objective. Unspecified defaults are not confirmed from installed package; no external API/version lookup performed. No custom negative sampling, hierarchical softmax, learning-rate schedule, sentence chunking, subsampling tuning or CBOW experiment demonstrated. Word2Vec/GloVe are named on L5 p21, L6 p10 and L7 p5 (B for GloVe; no GloVe code). Gensim's constructor trains on token sequences; no one-hot implementation required. The lab imports `numpy` under its full name, not `np`; either alias is a coding choice, not a mandated convention. The notebook treats each entire document as a token sequence, not sentence-level splits for own model.

### Second-pass Word2Vec decision register

| Technique / recommendation | Evidence class | Verification and instruction for Codex |
|---|---|---|
| Train Gensim on token lists; use document-level lists | A/C | L8 c27,59–60,74; P p2 train only on each challenge. No sentence-level split in L8. |
| Expand contractions, NLTK word tokenize, lowercase, alphabetic filter, stopword removal | A; choice for project D | L8 c24–27. This is the actual active pipeline; no lemmatization or stemming. Lab2/4 provide other taught preprocessing, not automatic Word2Vec defaults. |
| Use this exact preprocessing for the mandatory task | D | Assignment leaves Word2Vec tokenization unspecified. Start close to L8 or deliberately simplify; retaining function words/case/punctuation as authorship cues is a testable hypothesis, not course-mandated. |
| 100/300 dimensions, window 5, min_count 1, skip-gram, 10 epochs, seed 123 | A | L8 c60/74. These are exact explicit lab arguments, not required project values. |
| min_count=2, seed=42, workers=1, PYTHONHASHSEED=42 | D | R §2–3, H. No lab run with min_count=2 or workers=1; lab seed is 123. |
| CBOW | B / D if experimented | c60/74 comments describe sg=0; no supplied run sets sg=0. Do not claim it was trained. |
| Average vectors for known tokens only | A | c54,70,75,71,76. c42 expressly explains averaging; repeated words contribute repeatedly. |
| Empty/OOV document gets zeros | D/E | c54/70/75 have no guard; zero vector suggested by R/H, while handling empties is group contract. No proof lab ran into this condition. |
| OOV coverage measurement | A for inspecting missing words/counts; D for corpus-wide coverage metric | c46–49 compares first document token count and known vectors, lists missing words. Aggregated OOV rate is a proposed diagnostic, not a demonstrated metric. |
| Fixed-shape arrays passed directly to classifier | A | c82,84–100, RF fit/predict. No DataFrame needed. |
| Shared sklearn LR factory | C/D/E | LR required (P p2); factory and `lbfgs` are group conventions. L7 c68/c73 and L9 c45 show SGD logistic loss, not direct LogisticRegression; no end-to-end Word2Vec+LR in supplied labs. |
| 100 versus 300 dimension controlled comparison | A pattern / C contribution / D implementation | L8 c73–78 and c90–100 demonstrate changing width and comparing RF evaluations; repeat with project LR, preserve other settings. Width changes stochastic initialization too, so control does not imply bitwise determinism. |
| Compare min_count, CBOW vs skip-gram, pooling alternatives | D/E as completed experiments | Not shown as comparative runs. Mean pooling is A; alternative pooling is not demonstrated. Prefer the already demonstrated dimension comparison first. |
| Optional scaling, sparse-vs-dense choice, model wrapper, per-challenge fresh state | D (training-only is C) | No standalone scaler demonstrated; no requirement for W2V scaling. Wrapper and fit return follow group contract. |
| Report and confusion matrix | A/C | L8 c87–100; assignment requires classification_report and macro-F1. Direct macro-F1 function call is D, not a demonstrated lab call; report macro row is A/B. |

**Additional source cautions:** L8 c40 gives a verbal analogy/formula whose ordering is potentially confusing. The actual c41 call is `most_similar(positive=['France', 'Berlin'], negative=['Paris'])`. Preserve the source, use the actual call when explaining the example, and do not introduce analogy calculations into project classification. L8 c82 refers to `y_train`/`y_test` as vectors, but actual document matrices are assigned to `X_train`/`X_test` in c84,90,96; labels remain separate. Gensim vocabulary lookup and cosine similarity are inspection examples, not required project features. No additional package documentation used to verify unspecified defaults.

### Recommended implementation design (D, not extracted code)
1. In `Word2VecLR.fit(texts, labels)`, apply one explainable tokenization function to every training raw string; train new Gensim Word2Vec only on those sequences. Follow group seed/workers, start with group suggested parameters and document differences from c60. Do not load GoogleNews or combine challenges. Never build/update vocabulary from validation. If entire training corpus yields no usable vocabulary, fail clearly or agree on a fallback; zero vectors solve individual documents, not training an empty embedding vocabulary.
2. Keep trained model as instance state. For each document, collect known vectors using c70's membership loop and compute mean along axis 0. **Lab limitation:** `numpy.mean([] , axis=0)` can produce a scalar NaN/warnings, not the desired vector. Guard the empty list and return `numpy.zeros(vector_size)` (D). Use a consistently shaped numeric matrix and ensure one row per raw input; no dropped documents. Unseen tokens ignored, not encoded with newly learned vectors. Do not count unknown words in mean denominator. Check empty and all-OOV input explicitly; zero vector predicts according to learned LR intercept rather than conveying lexical information.
3. Fit common LR factory on document vectors and labels. `predict(texts)` repeats same tokenization and vectorization with frozen `model.wv`, invokes fitted classifier `.predict`, returns 1D labels. No new training or file I/O. No scaling required by assignment/lab; if tested as variant, use train-only scaler and explain. Preserve original rows/labels.
4. Controlled comparison: 100 vs 300 dims is directly motivated by c73–78; hold other settings, split, LR, tokenization and pooling fixed, create fresh models per variant/challenge. Alternatively compare `min_count=1` vs 2 or skip-gram vs CBOW, marked D; change one factor. Report all macro-F1/class reports, optionally OOV coverage and run-time; record actual values, no fabricated comparisons. Compare Member1's actual BoW/TF-IDF results across A/B/C. Because preprocessing may differ, don't claim representation alone caused differences. Avoid choosing a variant from repeated validation experiments and presenting its validation score as independent test performance.
5. Discuss mean pooling's loss of word order and stylistic information, dense semantic vs sparse lexical features, context learning and coverage tradeoffs. These are explanatory D interpretations, not measured claims. For oral defense explain why training-only Word2Vec matters even without labels, what sg/min_count/mean mean, how unknown words are handled, and how to change one parameter.

## 6. Engineered features and NLP tools

L6 c49–86 (A): `opinion_lexicon.positive()/negative()` converted to sets; `count_lexical_features(tokens)` sums membership matches, returns `positive_tokens`, `negative_tokens`; `sentiment_difference` subtraction; `existence_no` on raw tokenized text, `existence_!` directly on raw text, `log_number_tokens` via `log(len(tokens))` (`from math import log`, c77–78). c82–87 merges lexical and BoW on row index, then adds columns. Lab uses six engineered features with BoW, not ≥10 engineered features alone with LR. `log(0)` fails for empty token lists; D guard (e.g. log1p or defined zero convention); do not silently alter original. Ensure index/row alignment. P p2 names length, word count, mean word length as examples (C requires at least ten, not these exact three). Avoid external training data; if considering NLTK lexicons or pretrained language tools for features, verify instructor interpretation of supplied-data-only and group approval, since taught tools are not automatically permitted external resources.

L3 p3–4/c12–24 POS with NLTK `pos_tag`, Penn tagset, spaCy `nlp`, `token.pos_`, `token.tag_`; p6/c27–29 NER `doc.ents`, `ent.text`, `ent.label_`; p8–9/c34–37 VADER `SentimentIntensityAnalyzer.polarity_scores`, TextBlob `.sentiment.polarity` and `.subjectivity`; p11–14/c41–47 Stanza `Pipeline(lang='en', processors='tokenize,pos,constituency', verbose=False)`, `Tree.fromstring`, `pretty_print`, spaCy `doc.sents`, `noun_chunks`, `token.dep_`, `token.head`, `displacy.render`. L3 c56–75 counts adjectives/NER types and filters reviews by sentiment; demonstrates counting linguistic cues (A) but not project-specific efficacy. Prefer simple features first; avoid installing heavy parsers just to satisfy feature count.

Possible original features (D suggestions, not copied lab feature definitions): char length, whitespace/word-token count, average word length, unique-word ratio, sentence count, mean sentence length, punctuation count or rate, exclamation/question mark rate, digit ratio, uppercase ratio, stopword ratio, paragraph/newline count. Use exact denominators with empty-text guards; explain relevance to author style vs AI processing as hypotheses, not facts. Choose ≥10 defensible distinct features, not ten nearly identical length variants. L2 c64,69,122–125 underpins counts/unique counts; L6 raw punctuation indicator underpins punctuation-based variants. Compare all-feature LR with single-feature removal (D, assignment requires comparisons but does not dictate ablation), same settings, train-only fitting, discuss correlated features and sign/scale caveats. No train-fitted `StandardScaler`/`MinMaxScaler` example found (E as lab evidence, D if justified by group). TF-IDF's inherent weighting/normalization is different from feature scaling. If using scaling, fit only on train and apply unchanged to validation; document why for differing units.

## 7. Training, metrics, hyperparameters, error analysis

- L5 p19–26 and L6 p8–12, L7 p3–7, L9 p3: iterative collect/split/preprocess/extract/select/model/train/tune/predict/evaluate pipeline. L5 p25–26/c81–83 `cross_val_score(clf, bow_train_unigrams, df_train['label'], cv=5, scoring='accuracy')`, scores.mean(); L6 c112 compares NB variants by 5-fold CV. Lab demonstrates CV on already vectorized training matrix, so feature vocabulary/selection can include inner validation fold information. For unbiased extra CV, refit every learned transformation inside each fold (D); supplied validation remains fixed for core. L5 p26 diagram distinguishes single split vs CV; don't treat lab's test selection as independent test.
- L5 p35–38, L6 p16–19, L5 c71–76, L8 c87–100: `classification_report`, `confusion_matrix`, `ConfusionMatrixDisplay`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`; plot with cmap='Blues'. Slides p37/p18 show default `f1_score(y_test,y_pred)` (binary positive-class, NOT macro-F1). Macro row explained in L5 p36/L6 p17; project requires macro-F1, so use `f1_score(..., average='macro')` or report macro row with full precision (D API adaptation, C metric). Macro average equal weight per class; weighted average weights support; although L5 p36/L6 p17 label weighted averages as better for imbalanced datasets, the assignment specifically chooses macro-F1 and minority-class performance matters; accuracy is correct fraction. For B/C majority label 1 accuracy 2/3 alone does not indicate good minority performance. Compare support and precision/recall/F1 for each label, especially C where 0 is AI-polished human.
- Confusion matrix diagram p38/p19 labels TP/FN/FP/TN with positive first; sklearn displays labels in its chosen/specified order, often 0 then 1. Keep class names correct. Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2PR/(P+R), accuracy=(TP+TN)/total. No fabricated report values.
- L7 p14 names GridSearchCV, RandomizedSearchCV, Optuna (B only in supplied materials, no executable tuning demo). Manual comparisons of unigrams/bigrams, SVD vs non-SVD, embedding width, epochs and architecture are A. Optional tuning should be controlled and modest; no Optuna needed. L9 p17–22/c49–69 shows overfitting (screenshot near perfect train accuracy vs ~0.8 test for illustrative neural model), fewer epochs, dropout, early stopping and Adam. L9 slide p20 mentions regularization L1/L2, more data, augmentation, simpler model (B as distinct techniques unless implemented by library defaults); p22 lists architecture, activation, epochs, learning rate, momentum etc. Do not add neural nets, augmentation or outside data to mandatory LR. L9 c66 early stopping uses `validation_data=(X_test, y_test)` and `EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)` then evaluates the same X_test: this is not an independent held-out estimate. For extra neural experiments use an inner training validation split if needed; not required for the classical pipeline.
- Explicit systematic individual misclassification inspection routine not found (E); assignment requires actual error analysis (C). D: use shared `errors.csv`, record challenge/id/true/predicted label, text cues, plausible explanation and limitations, compare model disagreements; at least five actual mistakes per member. Do not invent errors or infer authorship from stylistic cues alone. Keep authorized data private. Report only actual experiments.

## 8. Other taught topics: language modeling and neural networks

L4 p3–6, p18–29, c6–78: `requests.get` transcript download, `open/readlines`, `itertools.chain.from_iterable`, NLTK `padded_everygram_pipeline(n, tokenized_texts)`, `MLE(n)`, `Laplace(n)`, `.fit`, `.vocab`, `.score`, `.generate(..., text_seed=..., random_seed=123)`. Unigram/bigram/4-gram examples, manual count formulas and custom `generate_sentence` greedy path. Generator is consumed and recreated for second model. Distinguish probability language models from BoW n-grams/Word2Vec. External transcript is lab-only. Perplexity as a project feature not demonstrated; not necessary. L4 p6 broadly describes GPT and BERT as next-token models; this is a simplification (not evidence to use them). L9 p7–16/c49–69 Keras `Sequential`, Dense 32 relu + sigmoid output, `SGD`, `Adam`, `Dropout(0.5)`, `EarlyStopping`, accuracy, binary crossentropy. These are taught but not the required classical approach; avoid extra dependencies for core.

## 9. Starter-kit compatibility review (documentation only)

R and H were supplied; actual `config.py`, `data_loader.py`, `evaluation.py`, `model_contract.py`, `main.py`, `requirements.txt`, `models/`, `experiments/`, tests and data not present. Thus documented design can be reviewed, not tested or implementation-verified. No starter files modified.

| Area | Documented contract and course alignment | Potential issue / recommendation for group |
|---|---|---|
| Model interface | R §2/H contract: no-arg constructor, `fit(texts, labels)` returns self, `predict(texts)` raw strings, 1D 0/1 same order, no mutation/import execution. P p4 raw predict required. | Minimal wrapper around familiar lab functions (D). No needless inheritance, abstract factories or pipeline redesign. Verify actual model_contract and runner. |
| LR | `make_logistic_regression()` with `C=1.0`, `solver='lbfgs'`, `max_iter=1000`, `class_weight=None`, `random_state=42`. | Direct sklearn LogisticRegression not seen in lab; group choice legitimate, consistent comparisons. Do not replace with SGD or Keras unilaterally. Convergence warnings: investigate, record and agree. |
| Dependencies | R says Python 3.13.5 tested with starter and pinned requirements; labs use NLTK, numpy, pandas, sklearn, gensim, optional spaCy/stanza/TensorFlow etc. | Exact pins unknown, no install/runtime verification. Check Gensim/NumPy/SciPy wheels/import on chosen Python; NLTK data resources separately installed. No speculative version fixes. Avoid heavy unused libraries. |
| Reproducibility | SEED 42, workers=1, seed=42, PYTHONHASHSEED=42 before Python. | Lab uses seed123 except SVD42. Group seed takes priority; not guarantee cross-platform identical output. Set environment outside process as R says. |
| Loader | `load_data(challenge, data_dir)`, raw order preserved, text/label/id assumed, training ID optional, validation ID required. | Actual CSV headers and type/empty behavior unverified. Verify IDs including leading zeros, labels, counts and order; change only shared loader by agreement. |
| Evaluation | `evaluate_and_save(data, predictions, output_dir)`; report, macro-F1, predictions/errors/metrics, comparison tables, final best-core model per challenge by unrounded macro-F1, ties registry order. | Group policy, not P specification. Verify output id,label exactly; validation used for selection is not blind test. No actual score known. |
| Experiment hooks | `experiments/member2.py` / `member3.py`, `run(data_dir, output_dir, challenges)`; shared runner `--extras`, fresh model per variant. | Good minimal orchestration if actual code matches. Do not duplicate loader/evaluator. Hooks and tests not inspected. |
| Empty inputs | Group says handle empty/OOV and preserve rows. | Lab c70 lacks empty guard; add minimal D guard. Verify no vocabulary for whole training set and empty prediction batch behavior against contract. |
| Feature scaling | R optional train-fit scaling for engineered features. | Not shown as standalone scaler in labs; propose only if needed; no scaler for raw-count NB. |
| Core / extras | Core LR same factory, baseline exact; variants separate. | Keep comparison controlled, document differences; no changes to core registry or pins without approval. |

Ownership per R §3/H:

| Member | File | Class | CLI |
|---|---|---|---|
| 1 | models/baseline.py | BaselineNB | baseline |
| 1 | models/bow_lr.py | BoWLR | bow_lr |
| 1 | models/tfidf_lr.py | TfidfLR | tfidf_lr |
| 2 | models/word2vec_lr.py | Word2VecLR | word2vec_lr |
| 3 | models/features_lr.py | FeaturesLR | features_lr |

R commands: `python main.py --model word2vec_lr --challenge A`, `python main.py --model word2vec_lr`, `python main.py --model word2vec_lr --extras`, after integration `python main.py --extras`. R says `python -m unittest discover -s tests -v` for tests. These commands are documentation, not verified executable here. Preserve `results/core/<model>/<challenge>/` metrics/report/predictions/errors and final three prediction files. No source changes proposed as mandatory except adherence to assignment; resolve any implementation discrepancy with group.

## 10. Evidence gaps and safeguards

E in supplied teaching materials: explicit sklearn `LogisticRegression` constructor, standalone `StandardScaler`/`MinMaxScaler`, sklearn Pipeline/ColumnTransformer workflow, `f1_score(..., average='macro')` executable call (macro concept taught), complete Word2Vec+LR pipeline, all-OOV/empty-vector guard, CBOW training run, weighted/max pooling, FastText/Doc2Vec or contextual embedding implementation, automated tuning code, per-document misclassification analysis procedure, ablation procedure, project-ready model contract/registry/CLI. Some related theory or names are taught (see prior sections), and some gaps are required adaptations. Do not infer prohibition from E. Explicit prohibition: pretrained transformers/LLMs as classifiers and using external datasets for required training; own Word2Vec must train on supplied data. No evidence that additional ordinary NLP features or classical methods are banned, subject to assignment constraints and instructor clarification.

Uncertainties: no Lab8 slides, no separate theory lectures, no real data, no actual starter code or pins; no end-to-end execution. Review source screenshots for exact formatting if needed; exact notebook source preferred. Some notebook examples deliberately fail or assume nonempty input. For project code, choose student-understandable adaptations and explain why. The assignment's four-vs-six question discrepancy and group final-model-selection policy need confirmation. Verify data headers and installed environment before implementation.

## 11. Notebook import and API index (source audit)

Exact active import statements grouped by notebook, with all source cell locations found. This includes auxiliary imports, not recommendations to add every dependency. Commented install commands are not active imports. Notebook magics are not Python imports.

### Lab 1 — `LAB1_solutions.ipynb`

- `import math` — c61, c104
- `from math import sqrt` — c62
- `import math as m` — c63
- `import random` — c107

User-defined functions: `say_hello` (c54); `compute_final_price` (c91); `odd_or_even` (c95).

Selected call-name index (syntactic occurrences, not inferred library ownership): `append` (c26, c108); `compute_final_price` (c92); `count` (c70, c109); `items` (c30); `join` (c73); `odd_or_even` (c95); `randint` (c107, c108); `replace` (c72); `say_hello` (c55); `seed` (c108); `split` (c25, c68); `sqrt` (c61, c62, c63); `upper` (c24, c85).

### Lab 2 — `LAB2_solutions.ipynb`

- `import re` — c4, c53
- `import nltk` — c48, c59
- `from nltk.corpus import gutenberg` — c48
- `from nltk.tokenize import word_tokenize, sent_tokenize` — c59
- `from collections import Counter` — c66
- `import string` — c76
- `from nltk.corpus import stopwords` — c94
- `from wordcloud import WordCloud` — c104
- `import matplotlib.pyplot as plt` — c104
- `from nltk.stem import PorterStemmer` — c109
- `from nltk.stem import WordNetLemmatizer` — c115

User-defined functions: `uncontract` (c53); `dimensionality_info` (c69).

Selected call-name index (syntactic occurrences, not inferred library ownership): `Counter` (c66, c80, c87, c98, c105); `PorterStemmer` (c109); `WordCloud` (c104); `WordNetLemmatizer` (c115); `all` (c78); `axis` (c104); `dimensionality_info` (c70, c74, c89, c90, c106, c113, c118); `download` (c48, c59, c94, c115); `filter` (c78, c86); `findall` (c6, c7, c8, c25, c27, c29, c33, c35, c43, c45); `format` (c66, c80, c87, c98, c105); `fullmatch` (c16, c17, c37); `generate` (c104); `group` (c14); `imshow` (c104); `isalpha` (c84, c86, c124); `isdigit` (c78); `join` (c104); `lemmatize` (c117, c124); `lower` (c72, c124); `match` (c10, c11, c12, c14, c39); `most_common` (c66, c80, c87, c98, c105); `raw` (c49); `sent_tokenize` (c63); `set_title` (c104); `show` (c104); `split` (c21, c22); `stem` (c110, c112); `sub` (c19, c41, c53); `subplots` (c104); `uncontract` (c55, c122); `word_tokenize` (c60, c122); `words` (c96).

### Lab 3 — `LAB3_solutions.ipynb`

- `import nltk` — c12, c14, c34
- `from nltk import word_tokenize, pos_tag` — c14, c50
- `import spacy` — c18, c24, c27
- `from spacy.cli import download` — c18
- `from spacy.glossary import GLOSSARY` — c21, c22
- `from nltk.sentiment import SentimentIntensityAnalyzer` — c34
- `from textblob import TextBlob` — c37, c50, c69
- `import stanza` — c41
- `from nltk import Tree` — c41
- `from spacy import displacy` — c46
- `import warnings` — c50
- `import nltk, spacy` — c50
- `from collections import Counter` — c58, c61
- `import matplotlib.pyplot as plt` — c69

Selected call-name index (syntactic occurrences, not inferred library ownership): `Counter` (c58, c63); `Pipeline` (c42); `SentimentIntensityAnalyzer` (c34, c72, c74); `TextBlob` (c37, c69); `append` (c61, c69, c72, c74); `download` (c12, c14, c18, c34, c41, c50); `enumerate` (c53); `explain` (c21, c22, c27); `extend` (c56, c73, c75); `figure` (c70); `filterwarnings` (c50); `fromstring` (c42); `get` (c21, c22); `get_pipe` (c22, c27); `hist` (c70); `load` (c22, c24, c27, c50); `most_common` (c58, c63, c67); `nlp` (c24, c29, c44, c46, c61); `polarity_scores` (c34, c72, c74); `pos_tag` (c15, c56, c73, c75); `pretty_print` (c42); `render` (c47); `show` (c70); `stanza_nlp` (c42); `startswith` (c56, c73, c75); `title` (c70); `upenn_tagset` (c12); `word_tokenize` (c15, c56, c73, c75); `xlabel` (c70); `ylabel` (c70).

### Lab 4 — `LAB4_solutions.ipynb`

- `import requests` — c6
- `import nltk` — c11, c18, c40, c62
- `from nltk.tokenize import word_tokenize` — c11
- `from nltk.stem import WordNetLemmatizer` — c11
- `from itertools import chain` — c15
- `from nltk.lm.preprocessing import padded_everygram_pipeline` — c18, c40, c62
- `from nltk.lm import MLE, Laplace` — c18, c40, c62
- `import math` — c32, c54, c74

User-defined functions: `generate_sentence` (c32, c54, c74).

Selected call-name index (syntactic occurrences, not inferred library ownership): `Laplace` (c21, c43, c65); `MLE` (c20, c42, c64); `WordNetLemmatizer` (c11); `append` (c8, c32, c54, c74); `copy` (c32, c54, c74); `count` (c30, c52); `download` (c11); `fit` (c20, c21, c42, c43, c64, c65); `from_iterable` (c15); `generate` (c37, c59, c78); `generate_sentence` (c34, c35, c56, c57, c76, c77); `get` (c6); `join` (c8, c32, c37, c54, c59, c70, c72, c74, c78); `lemmatize` (c12, c33, c55, c68, c75); `lower` (c12, c33, c55, c68, c75); `open` (c6, c8); `padded_everygram_pipeline` (c20, c21, c42, c43, c64, c65); `raise_for_status` (c6); `readlines` (c8); `score` (c28, c30, c32, c50, c52, c54, c70, c72, c74); `split` (c8); `strip` (c8); `word_tokenize` (c12, c33, c55, c68, c75); `write` (c6).

### Lab 5 — `LAB5_solutions.ipynb`

- `import nltk` — c5, c31
- `from nltk.corpus import movie_reviews` — c5
- `import pandas as pd` — c20, c88
- `from sklearn.model_selection import train_test_split` — c26
- `from IPython.display import clear_output` — c31
- `from nltk.corpus import gutenberg` — c31
- `from nltk.tokenize import word_tokenize` — c31
- `from nltk.corpus import stopwords` — c31
- `from nltk.stem import PorterStemmer` — c31
- `from nltk.stem import WordNetLemmatizer` — c31
- `import re` — c31
- `import string` — c31
- `from sklearn.feature_extraction.text import CountVectorizer` — c47
- `from sklearn.naive_bayes import MultinomialNB` — c62
- `from sklearn.metrics import classification_report` — c71
- `import matplotlib.pyplot as plt` — c74
- `from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay` — c74
- `from sklearn.model_selection import cross_val_score` — c81

User-defined functions: `uncontract` (c32); `preprocessing_text` (c34); `bow_features` (c48); `get_top_ngrams` (c57).

Selected call-name index (syntactic occurrences, not inferred library ownership): `ConfusionMatrixDisplay` (c74); `CountVectorizer` (c48); `DataFrame` (c20, c48, c91); `MultinomialNB` (c64, c76, c82, c83); `WordNetLemmatizer` (c34); `bow_features` (c50, c53); `categories` (c7); `classification_report` (c71, c76); `clear_output` (c31); `confusion_matrix` (c74); `count` (c18); `cross_val_score` (c82, c83); `describe` (c102); `download` (c5, c31); `fileids` (c9, c10, c11, c13, c14, c16, c18); `fit` (c66, c76); `fit_transform` (c48); `get_feature_names_out` (c48); `get_top_ngrams` (c58, c59); `head` (c22, c51, c52, c54, c55, c96); `hist` (c107, c108); `idxmax` (c110, c111); `isalpha` (c34); `join` (c49); `lemmatize` (c34); `lower` (c34); `mean` (c82, c83, c104); `plot` (c74); `predict` (c68, c76); `preprocessing_text` (c37, c39); `raw` (c13, c14, c16); `read_csv` (c114); `show` (c74); `sort_values` (c57); `startswith` (c18); `sub` (c32); `tail` (c23, c97); `title` (c74); `to_csv` (c113); `toarray` (c48); `train_test_split` (c27); `transform` (c48); `uncontract` (c34); `word_tokenize` (c34); `words` (c34).

### Lab 6 — `LAB6_solutions.ipynb`

- `import nltk` — c5, c30, c49
- `from nltk.corpus import movie_reviews` — c5
- `import pandas as pd` — c18
- `from sklearn.model_selection import train_test_split` — c24
- `from IPython.display import clear_output` — c30
- `from nltk.corpus import gutenberg` — c30
- `from nltk.tokenize import word_tokenize` — c30
- `from nltk.corpus import stopwords` — c30
- `from nltk.stem import PorterStemmer` — c30
- `from nltk.stem import WordNetLemmatizer` — c30
- `import re` — c30
- `import string` — c30
- `from sklearn.feature_extraction.text import CountVectorizer` — c40
- `from nltk.corpus import opinion_lexicon` — c49
- `from math import log` — c77
- `from sklearn.naive_bayes import MultinomialNB` — c90, c93
- `from sklearn.metrics import accuracy_score` — c95
- `from sklearn.metrics import classification_report` — c96
- `from sklearn.ensemble import RandomForestClassifier` — c99, c103
- `import matplotlib.pyplot as plt` — c101
- `from sklearn.model_selection import cross_val_score` — c112

User-defined functions: `uncontract` (c31); `tokenize_text` (c31); `lowercase_tokens` (c31); `remove_punctuation_digits` (c31); `remove_stopwords` (c31); `perform_stemming` (c31); `perform_lemmatization` (c31); `preprocessing_text` (c33); `count_lexical_features` (c54); `get_top_features` (c103).

Selected call-name index (syntactic occurrences, not inferred library ownership): `CountVectorizer` (c42); `DataFrame` (c18, c43, c45, c59, c99, c103); `MultinomialNB` (c90, c93, c109, c112); `PorterStemmer` (c31); `RandomForestClassifier` (c99, c103); `WordNetLemmatizer` (c31); `accuracy_score` (c95, c109); `barh` (c101); `categories` (c7); `classification_report` (c96, c110); `clear_output` (c30); `count` (c16); `count_lexical_features` (c55, c57); `cross_val_score` (c112); `download` (c5, c30, c49); `drop` (c92, c103); `figure` (c101); `fileids` (c9, c11, c12, c14, c16); `fit` (c90, c93, c99, c103, c109); `fit_transform` (c43); `gca` (c101); `get_feature_names_out` (c43, c45); `get_top_features` (c104, c107); `head` (c20, c44, c46, c60, c61, c64, c65, c82, c83, c84, c86, c87, c100, c101, c103); `invert_yaxis` (c101); `isalpha` (c31); `join` (c41); `lemmatize` (c31); `log` (c78); `lower` (c31); `lowercase_tokens` (c33); `mean` (c112); `merge` (c82, c86); `negative` (c50); `perform_lemmatization` (c33); `positive` (c50); `predict` (c94, c109); `preprocessing_text` (c35, c55); `raw` (c11, c12, c14); `remove_punctuation_digits` (c33); `remove_stopwords` (c33); `round` (c112); `show` (c101); `sort_values` (c100, c103); `startswith` (c16); `stem` (c31); `sub` (c31); `tail` (c21); `title` (c101); `toarray` (c43, c45); `tokenize_text` (c33); `tolist` (c103); `train_test_split` (c25); `transform` (c45); `uncontract` (c33); `word_tokenize` (c31, c68); `words` (c31); `xlabel` (c101).

### Lab 7 — `LAB7_solutions.ipynb`

- `import nltk` — c5, c27
- `from nltk.corpus import movie_reviews` — c5
- `import pandas as pd` — c15
- `from sklearn.model_selection import train_test_split` — c21
- `from IPython.display import clear_output` — c27
- `from nltk.corpus import gutenberg` — c27
- `from nltk.tokenize import word_tokenize` — c27
- `from nltk.corpus import stopwords` — c27
- `from nltk.stem import PorterStemmer` — c27
- `from nltk.stem import WordNetLemmatizer` — c27
- `import re` — c27
- `import string` — c27
- `from sklearn.feature_extraction.text import TfidfVectorizer` — c36
- `from sklearn.decomposition import TruncatedSVD` — c47
- `from sklearn.linear_model import SGDClassifier` — c65
- `from sklearn.metrics import classification_report` — c65

User-defined functions: `uncontract` (c28); `tokenize_text` (c28); `lowercase_tokens` (c28); `remove_punctuation_digits` (c28); `remove_stopwords` (c28); `perform_stemming` (c28); `perform_lemmatization` (c28); `preprocessing_text` (c29); `tfidf_features` (c38).

Selected call-name index (syntactic occurrences, not inferred library ownership): `DataFrame` (c15, c38, c57, c58, c60); `PorterStemmer` (c28); `SGDClassifier` (c68, c73, c78, c83); `TfidfVectorizer` (c38); `TruncatedSVD` (c48, c60); `WordNetLemmatizer` (c28); `categories` (c7); `classification_report` (c70, c75, c80, c85); `clear_output` (c27); `count` (c13); `download` (c5, c27); `fileids` (c9, c11, c13); `fit` (c68, c73, c78, c83); `fit_transform` (c38, c49, c60); `get_feature_names_out` (c38); `head` (c17, c41, c42, c43, c44, c57, c58, c61, c62); `isalpha` (c28); `join` (c37); `lemmatize` (c28); `lower` (c28); `lowercase_tokens` (c29); `perform_lemmatization` (c29); `predict` (c69, c74, c79, c84); `preprocessing_text` (c31); `raw` (c11); `remove_punctuation_digits` (c29); `remove_stopwords` (c29); `startswith` (c13); `stem` (c28); `sub` (c28); `tail` (c18); `tfidf_features` (c39, c40); `toarray` (c38); `tokenize_text` (c29); `train_test_split` (c22); `transform` (c38, c49, c60); `uncontract` (c29); `word_tokenize` (c28); `words` (c28).

### Lab 8 — `LAB8_solutions.ipynb`

- `import nltk` — c5, c23
- `from nltk.corpus import movie_reviews` — c5
- `from sklearn.model_selection import train_test_split` — c17
- `from IPython.display import clear_output` — c23
- `from nltk.corpus import gutenberg` — c23
- `from nltk.tokenize import word_tokenize` — c23
- `from nltk.corpus import stopwords` — c23
- `from nltk.stem import PorterStemmer` — c23
- `from nltk.stem import WordNetLemmatizer` — c23
- `import re` — c23
- `import string` — c23
- `import gensim.downloader as api` — c33
- `import numpy` — c43
- `from gensim.models import Word2Vec` — c59
- `from sklearn.metrics.pairwise import cosine_similarity` — c65
- `from sklearn.ensemble import RandomForestClassifier` — c81
- `from sklearn.metrics import classification_report` — c81
- `import matplotlib.pyplot as plt` — c81
- `from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay` — c81

User-defined functions: `uncontract` (c24); `tokenize_text` (c24); `lowercase_tokens` (c24); `remove_punctuation_digits` (c24); `remove_stopwords` (c24); `perform_stemming` (c24); `perform_lemmatization` (c24); `preprocessing_text` (c25); `get_document_vector` (c54); `get_document_vector_our_model` (c70); `get_document_vector_our_model_300` (c75).

Selected call-name index (syntactic occurrences, not inferred library ownership): `ConfusionMatrixDisplay` (c88, c94, c100); `PorterStemmer` (c24); `RandomForestClassifier` (c85, c91, c97); `Word2Vec` (c60, c74); `WordNetLemmatizer` (c24); `append` (c44, c49, c54, c70, c75); `categories` (c7); `classification_report` (c87, c93, c99); `clear_output` (c23); `confusion_matrix` (c88, c94, c100); `cosine_similarity` (c65); `count` (c13); `download` (c5, c23); `fileids` (c9, c11, c13); `fit` (c85, c91, c97); `get_document_vector` (c55); `get_document_vector_our_model` (c71); `get_document_vector_our_model_300` (c76); `isalpha` (c24); `keys` (c35, c61); `lemmatize` (c24); `load` (c33, c68); `lower` (c24); `lowercase_tokens` (c25); `mean` (c51, c52, c54, c70, c75); `most_similar` (c41, c63, c78); `plot` (c88, c94, c100); `predict` (c86, c92, c98); `preprocessing_text` (c27); `raw` (c11); `remove_punctuation_digits` (c25); `remove_stopwords` (c25); `save` (c67); `show` (c88, c94, c100); `startswith` (c13); `stem` (c24); `sub` (c24); `title` (c88, c94, c100); `tokenize_text` (c25); `train_test_split` (c18); `uncontract` (c25); `word_tokenize` (c24); `words` (c24).

### Lab 9 — `LAB9_solutions.ipynb`

- `import nltk` — c5, c24
- `from nltk.corpus import movie_reviews` — c5
- `import pandas as pd` — c15
- `from sklearn.model_selection import train_test_split` — c18
- `from IPython.display import clear_output` — c24
- `from nltk.corpus import gutenberg` — c24
- `from nltk.tokenize import word_tokenize` — c24
- `from nltk.corpus import stopwords` — c24
- `from nltk.stem import PorterStemmer` — c24
- `from nltk.stem import WordNetLemmatizer` — c24
- `import re` — c24
- `import string` — c24
- `from sklearn.feature_extraction.text import CountVectorizer` — c33
- `import tensorflow as tf` — c42
- `from tensorflow.keras.models import Sequential` — c42
- `from tensorflow.keras.layers import Dense, Dropout` — c42
- `from tensorflow.keras.optimizers import SGD, Adam` — c42
- `from sklearn.linear_model import SGDClassifier` — c42
- `from sklearn.metrics import accuracy_score, classification_report` — c42
- `from tensorflow.keras.callbacks import EarlyStopping` — c42

User-defined functions: `uncontract` (c25); `tokenize_text` (c25); `lowercase_tokens` (c25); `remove_punctuation_digits` (c25); `remove_stopwords` (c25); `perform_stemming` (c25); `perform_lemmatization` (c25); `preprocessing_text` (c26).

Selected call-name index (syntactic occurrences, not inferred library ownership): `Adam` (c68, c69); `CountVectorizer` (c35); `DataFrame` (c15, c36, c38); `Dense` (c49, c52, c55, c57, c61, c63, c66, c68, c69); `Dropout` (c61, c63, c69); `EarlyStopping` (c66); `PorterStemmer` (c25); `SGD` (c49, c52, c55, c57, c61, c63, c66); `SGDClassifier` (c45); `Sequential` (c49, c52, c55, c57, c61, c63, c66, c68, c69); `WordNetLemmatizer` (c25); `accuracy_score` (c45, c49, c52, c55, c57, c61, c63, c66, c68, c69); `astype` (c49, c52, c55, c57, c61, c63, c66, c68, c69); `categories` (c7); `classification_report` (c46); `clear_output` (c24); `compile` (c49, c52, c55, c57, c61, c63, c66, c68, c69); `count` (c13); `download` (c5, c24); `fileids` (c9, c11, c13); `fit` (c45, c49, c52, c55, c57, c61, c63, c66, c68, c69); `fit_transform` (c36); `get_feature_names_out` (c36, c38); `head` (c37, c39); `isalpha` (c25); `join` (c34); `lemmatize` (c25); `lower` (c25); `lowercase_tokens` (c26); `perform_lemmatization` (c26); `predict` (c45, c49, c52, c55, c57, c61, c63, c66, c68, c69); `preprocessing_text` (c28); `raw` (c11); `remove_punctuation_digits` (c26); `remove_stopwords` (c26); `startswith` (c13); `stem` (c25); `sub` (c25); `toarray` (c36, c38); `tokenize_text` (c26); `train_test_split` (c19); `transform` (c38); `uncontract` (c26); `word_tokenize` (c25); `words` (c25).

## 12. Second verification pass — source coverage and corrections

**Method and limits:** Reopened all 21 original source files and the three prior Markdown files locally. Extracted text from all PDF pages and rendered each page into contact sheets; reviewed all sheets for visual structure and the assignment-relevant diagrams/code against extracted text. Parsed all 884 notebook cell sources, including Markdown and empty cells, and compared every originally labeled exact cell with the original JSON. Reviewed the Word2Vec notebook in full source order and checked core classification, engineered features, preprocessing, metrics and assignment references. Decoded and visually reviewed all nine saved PNG notebook outputs (word cloud, histograms, feature importances and confusion matrices); checked Lab 8 saved classification reports against the corresponding matrix images. No hidden code identified in those nine images. Other saved text/HTML outputs were not exhaustively audited. This is a source/static audit, not runtime validation. It does not certify every low-resolution embedded image or every saved notebook output as independently read at full size. No online sources were used. The table uses **inspected** to mean all page text extracted and rendered or all cell source parsed, with focused content review, not a claim of exhaustive independent visual transcription.

| Uploaded file (exact filename) | Inspected / total | Topics extracted | Unreadable / unprocessed sections |
|---|---:|---|---|
| `LAB1_Brief_Introduction_to_Python_and_Jupyter_Notebooks.pdf` | 70/70 pages | Jupyter, basic Python, functions, loops, libraries | p4 and pp66–70 Jupyter/UI screenshots reviewed at overview scale; small UI text not independently transcribed. |
| `LAB1_solutions.ipynb` | 110/110 cells | Python types, control flow, exercises, libraries | All 61 nonempty code cells copied; 1 empty code cells omitted from examples. Saved outputs not executed or exhaustively transcribed. |
| `LAB2_Regular Expressions_and_Pre-processing_with_NLTK.pdf` | 55/55 pages | Regex, cleaning, tokenization, normalization, vocabulary | No inaccessible page; small decorative/output-image details not separately transcribed. |
| `LAB2_solutions.ipynb` | 126/126 cells | Regex, Gutenberg, tokenization, normalization, tweets, word counts | All 72 nonempty code cells copied; 1 empty code cells omitted from examples. Saved outputs not executed or exhaustively transcribed. Reviewed plotted PNG output cells 104. |
| `LAB3_Application_of_Fundamental_NLP_Tasks.pdf` | 16/16 pages | POS, NER, sentiment, parsing | No inaccessible page; small decorative/output-image details not separately transcribed. |
| `LAB3_solutions.ipynb` | 76/76 cells | POS, NER, sentiment, parsing and review analysis | All 34 nonempty code cells copied; 1 empty code cells omitted from examples. Saved outputs not executed or exhaustively transcribed. Reviewed plotted PNG output cells 70. |
| `LAB4_Language_Modeling_with_NLTK.pdf` | 31/31 pages | N-gram LM, preprocessing, smoothing, generation | No inaccessible page; small decorative/output-image details not separately transcribed. |
| `LAB4_solutions.ipynb` | 80/80 cells | Transcript, MLE/Laplace unigram/bigram/4-gram generation | All 50 nonempty code cells copied; 2 empty code cells omitted from examples. Saved outputs not executed or exhaustively transcribed. |
| `LAB5_Text_Classification.pdf` | 40/40 pages | Pandas, BoW, NB, CV, metrics | p8–9,12,25,31,35–38 embedded output/diagram values not all transcribed; code and concepts covered. |
| `LAB5_solutions.ipynb` | 116/116 cells | BoW, NB, evaluation, CV, pandas | All 67 nonempty code cells copied; 2 empty code cells omitted from examples. Saved outputs not executed or exhaustively transcribed. Reviewed plotted PNG output cells 74, 107, 108. |
| `LAB6_Text_Classification_2.pdf` | 21/21 pages | Feature engineering, merge, RF selection, metrics | p3–4,16–19 illustrative table/metric values not all transcribed. |
| `LAB6_solutions.ipynb` | 114/114 cells | BoW, six features, RF ranking, NB, CV | All 68 nonempty code cells copied; 2 empty code cells omitted from examples. Saved outputs not executed or exhaustively transcribed. Reviewed plotted PNG output cells 101. |
| `LAB7_Text_Classification_and_Dimensionality_Reduction_with_SVD.pdf` | 19/19 pages | TF-IDF, SGD LR, tuning theory, SVD | No inaccessible page; small decorative/output-image details not separately transcribed. |
| `LAB7_solutions.ipynb` | 88/88 cells | TF-IDF, SVD, SGD logistic loss | All 53 nonempty code cells copied; 1 empty code cells omitted from examples. Saved outputs not executed or exhaustively transcribed. |
| `LAB8_solutions.ipynb` | 103/103 cells | Word2Vec: own/pretrained, mean, OOV, width, RF | All 62 nonempty code cells copied; 1 empty code cells omitted from examples. Saved outputs not executed or exhaustively transcribed. Reviewed plotted PNG output cells 88, 94, 100. |
| `LAB9_Feed_Forward_Neural_Networks_with_Keras.pdf` | 24/24 pages | Keras LR/NN, overfitting, hyperparameters | p6,14 diagrams and pp18–19 training-log screenshot reviewed; exact log values not independently transcribed. |
| `LAB9_solutions.ipynb` | 71/71 cells | BoW, SGD and Keras LR, one-hidden-layer NN, dropout/early stopping | All 34 nonempty code cells copied; 1 empty code cells omitted from examples. Saved outputs not executed or exhaustively transcribed. |
| `Text_Classification_extra_notes.pdf` | 4/4 pages | Train/test pipeline diagram, numeric features and labels, train fit/test transform | p4 diagram reviewed; prediction variable naming inconsistency documented. |
| `ΕΠΛ499 Group Project - Human, AI, and AI-assisted Text Generation Detection.pdf` | 4/4 pages | All five methods, split, labels, evaluation, contribution, competition, deliverables and grading | No inaccessible page; four-versus-six question mismatch unresolved. |
| `README.md` | 92/92 lines | Shared conventions, interfaces, setup, responsibilities, output contracts | Documentation only; actual shared source, pins and data unavailable. |
| `LLM_HANDOFF.md` | 42/42 lines | Shared conventions, interfaces, setup, responsibilities, output contracts | Documentation only; actual shared source, pins and data unavailable. |

**Totals:** 10 PDFs / 284 pages; 9 notebooks / 884 cells, 501 nonempty code cells; 2 Markdown starter documents. All 21 supplied originals accounted for. Missing Lab 8 slide PDF was never uploaded; no invented slide references.

### Code completeness comparison

| Notebook | Nonempty code cells | Original excerpts verified | Restored in appendix | Final exact coverage |
|---|---:|---:|---:|---:|
| Lab 1 | 61 | 20 | 41 | 61/61 |
| Lab 2 | 72 | 52 | 20 | 72/72 |
| Lab 3 | 34 | 27 | 7 | 34/34 |
| Lab 4 | 50 | 32 | 18 | 50/50 |
| Lab 5 | 67 | 50 | 17 | 67/67 |
| Lab 6 | 68 | 32 | 36 | 68/68 |
| Lab 7 | 53 | 30 | 23 | 53/53 |
| Lab 8 | 62 | 60 | 2 | 62/62 |
| Lab 9 | 34 | 18 | 16 | 34/34 |
| **Total** | **501** | **321** | **180** | **501/501** |

All 322 original `Exact` blocks (321 distinct cells; L8 c43 repeated) matched the original cell text (ignoring trailing newlines). The final 502 `Exact` blocks cover 501 unique nonempty cells, and every one matched the source JSON. The final collection adds 180 previously omitted code cells and 20 explicitly fenced Markdown snippets (including exercise placeholders), all compared to the corresponding original Markdown cell, plus further selected slide transcriptions. Notebook output text and pictures are not part of this code-cell coverage measure. Slide examples are selected rather than a full verbatim slide-code archive; the original PDFs remain authoritative.

### Corrections and scope decisions
- Corrected the evidence classification of `re.IGNORECASE`: actual slide code in L2 p11, not theory only. Added the example, CSV/DataFrame operations, SVD and Keras slide transcriptions; included exact fenced notebook Markdown examples as separate illustrative source.
- Restored Lab 6/7/9 repeated setup and preprocessing cells, Lab 6 BoW cells 40–46, Lab 2 final tweet exercise c122–125 and every other omitted nonempty code cell, including display-only cells. No claim that all such code is necessary for the assignment.
- Clarified Word2Vec preprocessing rationale, CBOW vs actual skip-gram runs, absence of zero-vector guard, seed/workers/min_count as group choices, and the actual RF-to-LR adaptation. Distinguished observed per-document missing-token inspection from a proposed corpus-wide OOV metric.
- Clarified the original `log` import (L6 c77–78), direct sklearn LR absence, GloVe mention vs code, report macro row vs direct macro-F1 API, grading breakdown, and starter recommendation versus requirement. Corrected source citation for PAN Subtask 2 footnote.
- No invented course algorithms or incorrect exact-code transcriptions found in the 322 checked blocks. This is not a guarantee that every prose statement or peripheral screenshot is error-free. Remaining concerns: actual CSV schemas, pinned packages and shared code unavailable; no notebook runs; no exact full-resolution transcription of every PDF visual or saved notebook output.
- The original three artifacts remain unmodified in the source folder; the verified replacements are in the verified output folder. No actual project implementation or starter-kit change.

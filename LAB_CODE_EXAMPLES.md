# EPL499 lab code examples — source excerpts

Read alongside [COURSE_REFERENCE.md](COURSE_REFERENCE.md) and original assignment. The priority order below starts with Member 2 (Word2Vec), then foundational/classification examples. **Exact notebook source** means copied from JSON cell source without rewriting code or comments; only surrounding Markdown fences and a final newline are added. Cell number is 1-based physical index including Markdown cells, not execution count. The grouped cells can depend on omitted cells and original notebook state; they are examples, not a runnable combined script. Imports listed in each heading are explanatory, not invented imports in extracted source. No notebook execution, downloaded assets, fixes, or fabricated results. Notebook magic commands are preserved as such.

Exact nonempty code-cell source is A evidence of what appears in the lab, not necessarily proof of a successful execution. Comment-only cells, notebook Markdown pseudocode, slide examples, errors and unused helpers are not evidence of a trained model. All techniques must be interpreted with their context. Assignment-required settings and group adaptations are summarized separately at the end. Do not use lab data or pretrained vectors in mandatory project training.

## 1. Word2Vec: split training and test lists

**Source:** `LAB8_solutions.ipynb`, cells 5, 7, 9, 11, 13, 17, 18, 19, 20.

**Purpose / explanation:** Prepare parallel raw text and labels, split before preprocessing.

**Required imports / context:** nltk.corpus.movie_reviews; sklearn.model_selection.train_test_split.

**Limitations / project notes:** Movie reviews and 80/20 split are lab-only; use supplied CSV split in project.

### Exact — Lab 8, cell 5

```python
import nltk
from nltk.corpus import movie_reviews

nltk.download("movie_reviews")
```

### Exact — Lab 8, cell 7

```python
movie_reviews.categories()
```

### Exact — Lab 8, cell 9

```python
print(len(movie_reviews.fileids("pos")), "positive reviews")
print(len(movie_reviews.fileids("neg")), "negative reviews")
```

### Exact — Lab 8, cell 11

```python
reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]

print("Total number of reviews:", len(reviews))
```

### Exact — Lab 8, cell 13

```python
labels = [1 if fileid.startswith('pos') else 0 for fileid in movie_reviews.fileids()]

print(labels.count(1), "positive reviews")
print(labels.count(0), "negative reviews")
```

### Exact — Lab 8, cell 17

```python
from sklearn.model_selection import train_test_split
```

### Exact — Lab 8, cell 18

```python
text_train, text_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.20, random_state=123)
```

### Exact — Lab 8, cell 19

```python
print("Number of reviews in training dataset: ", len(text_train))
print("Number of reviews in test dataset:     ", len(text_test))
```

### Exact — Lab 8, cell 20

```python
print("Number of positive reviews in training dataset: ", sum([l==1 for l in y_train]))
print("Number of negative reviews in training dataset: ", sum([l==0 for l in y_train]))

print("\n")

print("Number of positive reviews in test dataset:     ", sum([l==1 for l in y_test]))
print("Number of negative reviews in test dataset:     ", sum([l==0 for l in y_test]))
```


## 2. Word2Vec: exact preprocessing helpers and active pipeline

**Source:** `LAB8_solutions.ipynb`, cells 23, 24, 25, 27.

**Purpose / explanation:** Expand contractions, tokenize, lowercase, filter nonalphabetic tokens, remove stopwords, apply same steps to train/test.

**Required imports / context:** IPython.display.clear_output; nltk; nltk.tokenize.word_tokenize; nltk.corpus.stopwords; nltk.stem.PorterStemmer, WordNetLemmatizer; re; string.

**Limitations / project notes:** Imports unused Gutenberg and stemmers; active path does not stem/lemmatize. Downloads belong in setup, not fit/predict. Preserving some style cues is a possible project variant, not original code.

### Exact — Lab 8, cell 23

```python
from IPython.display import clear_output

clear_output()

import nltk

nltk.download("gutenberg")
from nltk.corpus import gutenberg

from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import PorterStemmer

nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

import re
import string
```

### Exact — Lab 8, cell 24

```python
def uncontract(text):
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)

    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ii]t)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Tt]here)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)

    return text


def tokenize_text(text):
    tokens = word_tokenize(text)

    return tokens


def lowercase_tokens(tokens):
    tokens = [t.lower() for t in tokens]

    return tokens


def remove_punctuation_digits(tokens):
    tokens = [t for t in tokens if t.isalpha()]

    return tokens


def remove_stopwords(tokens):
    stop_words = stopwords.words('english')

    tokens = [t for t in tokens if t not in stop_words]

    return tokens


def perform_stemming(tokens):
    ps = PorterStemmer()

    stemmed = [ps.stem(t) for t in tokens]

    return stemmed


def perform_lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()

    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]

    return lemmatized
```

### Exact — Lab 8, cell 25

```python
def preprocessing_text(text):
    text = uncontract(text)
    text = tokenize_text(text)
    text = lowercase_tokens(text)
    text = remove_punctuation_digits(text)
    text = remove_stopwords(text)

    return text
```

### Exact — Lab 8, cell 27

```python
clean_tokens_train = [preprocessing_text(t) for t in text_train]
clean_tokens_test  = [preprocessing_text(t) for t in text_test]
```


## 3. Word2Vec: train own model and inspect vectors

**Source:** `LAB8_solutions.ipynb`, cells 59, 60, 61, 62, 63, 65, 67, 68.

**Purpose / explanation:** Train 100-d skip-gram on clean_tokens_train, inspect vocabulary/vector similarity and save/load.

**Required imports / context:** gensim.models.Word2Vec; sklearn.metrics.pairwise.cosine_similarity.

**Limitations / project notes:** Lab min_count=1, seed=123, workers unspecified; group recommendations differ. No pretrained model necessary. The save/load demonstration writes a relative file and is optional.

### Exact — Lab 8, cell 59

```python
from gensim.models import Word2Vec
```

### Exact — Lab 8, cell 60

```python
# Create and train the model
model = Word2Vec(
    sentences=clean_tokens_train,
    vector_size=100,                  # dimensionality of embeddings
    window=5,                         # context window size
    min_count=1,                      # ignore words that appear less than this
    sg=1,                             # 1 = skip-gram - takes a target word and tries to predict its surrounding context words
                                      # 0 = CBOW - takes surrounding context words and tries to predict the target word
    epochs=10,                        # number of training iterations
    seed=123                          # for reproducibility
)
```

### Exact — Lab 8, cell 61

```python
# Get vocabulary and its size
print("Vocabulary:", list(model.wv.key_to_index.keys()))
print("Vocabulary size:", len(list(model.wv.key_to_index.keys())))
```

### Exact — Lab 8, cell 62

```python
# Get embedding vector for a word
print("Vector for 'character':", model.wv['character'])
```

### Exact — Lab 8, cell 63

```python
# Find most similar words
model.wv.most_similar('character')
```

### Exact — Lab 8, cell 65

```python
from sklearn.metrics.pairwise import cosine_similarity
X = model.wv['character']
Y = model.wv['development']
print(cosine_similarity([X, Y]))
print('\n')
print(cosine_similarity([X, Y])[0][1])
```

### Exact — Lab 8, cell 67

```python
# Save model
model.save("my_word2vec.model")
```

### Exact — Lab 8, cell 68

```python
# Load it again
model = Word2Vec.load("my_word2vec.model")
```


## 4. Word2Vec: OOV membership and document mean

**Source:** `LAB8_solutions.ipynb`, cells 43, 70, 71, 72.

**Purpose / explanation:** Collect vectors for known tokens and mean over rows, reuse same model on train/test.

**Required imports / context:** numpy (import in c43); fitted model from c60.

**Limitations / project notes:** No guard if document_vectors empty; mean of empty list may yield scalar NaN. Project needs fixed-size zero fallback (D), no dropping rows. The model variable is notebook-global; class state is D.

### Exact — Lab 8, cell 43

```python
import numpy
```

### Exact — Lab 8, cell 70

```python
def get_document_vector_our_model(document_tokens):

    document_vectors = []

    for t in document_tokens:

    ##########################
    # Check if token exists  #
    # in the word2vec model. #
    ##########################

        if t in model.wv:
            document_vectors.append(model.wv[t])

    return numpy.mean(document_vectors, axis=0)
```

### Exact — Lab 8, cell 71

```python
train_vectors_our_model = [get_document_vector_our_model(t) for t in clean_tokens_train]
test_vectors_our_model  = [get_document_vector_our_model(t) for t in clean_tokens_test]
```

### Exact — Lab 8, cell 72

```python
train_vectors_our_model[:2]
```


## 5. Word2Vec: controlled comparison of embedding dimension

**Source:** `LAB8_solutions.ipynb`, cells 74, 75, 76, 78.

**Purpose / explanation:** Train new 300-d model with same explicit parameters, pool and compare similarities.

**Required imports / context:** gensim.models.Word2Vec; numpy.

**Limitations / project notes:** Same empty-list limitation; exact reproducibility across runs not guaranteed; no CBOW training run shown.

### Exact — Lab 8, cell 74

```python
# Create and train the model
model300 = Word2Vec(
    sentences=clean_tokens_train,
    vector_size=300,                  # dimensionality of embeddings
    window=5,                         # context window size
    min_count=1,                      # ignore words that appear less than this
    sg=1,                             # 1 = skip-gram; 0 = CBOW
    epochs=10,                        # number of training iterations
    seed=123                          # for reproducibility
)
```

### Exact — Lab 8, cell 75

```python
def get_document_vector_our_model_300(document_tokens):

    document_vectors = []

    for t in document_tokens:

    ##########################
    # Check if token exists  #
    # in the word2vec model. #
    ##########################

        if t in model300.wv:
            document_vectors.append(model300.wv[t])

    return numpy.mean(document_vectors, axis=0)
```

### Exact — Lab 8, cell 76

```python
train_vectors_our_model_300 = [get_document_vector_our_model_300(t) for t in clean_tokens_train]
test_vectors_our_model_300  = [get_document_vector_our_model_300(t) for t in clean_tokens_test]
```

### Exact — Lab 8, cell 78

```python
model300.wv.most_similar('character')
```


## 6. Word2Vec: own embeddings into classifier and report

**Source:** `LAB8_solutions.ipynb`, cells 81, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100.

**Purpose / explanation:** Use document-vector lists directly as X_train and X_test; fit and predict with RF, compare 100/300.

**Required imports / context:** sklearn.ensemble.RandomForestClassifier; sklearn.metrics.classification_report, confusion_matrix, ConfusionMatrixDisplay; matplotlib.pyplot.

**Limitations / project notes:** Classifier is RF, NOT LR. For assignment replace with shared LR factory (C/D), retain train/test discipline. Sentiment display labels do not match A/B/C labels.

### Exact — Lab 8, cell 81

```python
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```

### Exact — Lab 8, cell 90

```python
X_train = train_vectors_our_model
X_test  = test_vectors_our_model
```

### Exact — Lab 8, cell 91

```python
clf = RandomForestClassifier(n_estimators=100, random_state=123)

clf.fit(X_train,y_train)
```

### Exact — Lab 8, cell 92

```python
y_pred = clf.predict(X_test)
```

### Exact — Lab 8, cell 93

```python
print(classification_report(y_test, y_pred))
```

### Exact — Lab 8, cell 94

```python
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
```

### Exact — Lab 8, cell 96

```python
X_train = train_vectors_our_model_300
X_test  = test_vectors_our_model_300
```

### Exact — Lab 8, cell 97

```python
clf = RandomForestClassifier(n_estimators=100, random_state=123)

clf.fit(X_train,y_train)
```

### Exact — Lab 8, cell 98

```python
y_pred = clf.predict(X_test)
```

### Exact — Lab 8, cell 99

```python
print(classification_report(y_test, y_pred))
```

### Exact — Lab 8, cell 100

```python
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
```


## 7. Word2Vec: pretrained example (reference only; not permitted for core)

**Source:** `LAB8_solutions.ipynb`, cells 32, 33, 35, 36, 38, 39, 41, 43, 44, 46, 47, 49, 51, 52, 54, 55, 84, 85, 86, 87, 88.

**Purpose / explanation:** Demonstrate GoogleNews vocabulary, analogy and mean pooling; list OOV words, evaluate pretrained representation.

**Required imports / context:** gensim.downloader as api; numpy; RF and metrics imports from c81.

**Limitations / project notes:** Large external download, prohibited for required own Word2Vec; do not execute in project. c81 import needed before RF. Mean function lacks all-OOV guard.

### Exact — Lab 8, cell 32

```python
!pip install gensim
```

### Exact — Lab 8, cell 33

```python
import gensim.downloader as api

wv_model = api.load('word2vec-google-news-300')
```

### Exact — Lab 8, cell 35

```python
vocab = list(wv_model.key_to_index.keys())

print(vocab[0:20])
```

### Exact — Lab 8, cell 36

```python
len(vocab)
```

### Exact — Lab 8, cell 38

```python
wv_model['man']
```

### Exact — Lab 8, cell 39

```python
wv_model['man'].shape
```

### Exact — Lab 8, cell 41

```python
################################################
# Paris is to France like Berlin is to _______ #
################################################

wv_model.most_similar(positive=['France', 'Berlin'], negative=['Paris'])
```

### Exact — Lab 8, cell 43

```python
import numpy
```

### Exact — Lab 8, cell 44

```python
vectors = []

for t in clean_tokens_train[0]:
    if t in wv_model: # check if our token exists in the dataset
        vectors.append(wv_model[t])
```

### Exact — Lab 8, cell 46

```python
len(clean_tokens_train[0])
```

### Exact — Lab 8, cell 47

```python
len(vectors)
```

### Exact — Lab 8, cell 49

```python
words = []

for t in clean_tokens_train[0]:
    if t not in wv_model:
        words.append(t)

words
```

### Exact — Lab 8, cell 51

```python
numpy.mean(vectors, axis=0)
```

### Exact — Lab 8, cell 52

```python
numpy.mean(vectors, axis=0).shape
```

### Exact — Lab 8, cell 54

```python
def get_document_vector(document_tokens):

    document_vectors = []

    for t in document_tokens:

    ##########################
    # Check if token exists  #
    # in the word2vec model. #
    ##########################

        if t in wv_model:
            document_vectors.append(wv_model[t])

    return numpy.mean(document_vectors, axis=0)
```

### Exact — Lab 8, cell 55

```python
train_vectors = [get_document_vector(t) for t in clean_tokens_train]
test_vectors  = [get_document_vector(t) for t in clean_tokens_test]
```

### Exact — Lab 8, cell 84

```python
X_train = train_vectors
X_test  = test_vectors
```

### Exact — Lab 8, cell 85

```python
clf = RandomForestClassifier(n_estimators=100, random_state=123)

clf.fit(X_train,y_train)
```

### Exact — Lab 8, cell 86

```python
y_pred = clf.predict(X_test)
```

### Exact — Lab 8, cell 87

```python
print(classification_report(y_test, y_pred))
```

### Exact — Lab 8, cell 88

```python
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
```


## 8. Python: foundational text operations and simple functions

**Source:** `LAB1_solutions.ipynb`, cells 13, 24, 25, 26, 30, 67, 68, 69, 70, 71, 72, 73, 79, 85, 91, 95, 98, 101, 108, 109.

**Purpose / explanation:** Lists/dictionaries, split/count/join, membership, loops, simple functions, comprehensions, random seed.

**Required imports / context:** random from c107 for c108; otherwise Python built-ins.

**Limitations / project notes:** Cells use earlier notebook variables (person c13, definition_list c71); demonstrations not one standalone script; random seed 123 is lab, group uses 42.

### Exact — Lab 1, cell 13

```python
# list
fruits = ['orange','apple']
numbers = [1,2,3]
mixed_list = ['Maria',18]

# tuple
point = (2,2)
mixed_tuple = ('x','y',3)

# set
numbers_set = {1,2,3}

# dict
person = {'name':'John','age':30}
```

### Exact — Lab 1, cell 24

```python
'hello'.upper()
```

### Exact — Lab 1, cell 25

```python
'Hello world!'.split()
```

### Exact — Lab 1, cell 26

```python
numbers_list = [1,2,3]
numbers_list.append(4)
numbers_list
```

### Exact — Lab 1, cell 30

```python
# to see all pairs of keys and values, we can write `dictionay_name.items()`

person.items()
```

### Exact — Lab 1, cell 67

```python
# 1. Create a string variable called text that contains the previous text.

text = '''Natural language processing (NLP) is the processing of natural language information by a computer.
The study of NLP, a subfield of computer science, is generally associated with artificial intelligence.
NLP is related to information retrieval, knowledge representation, computational linguistics, and more broadly with linguistics.[1]

Major processing tasks in an NLP system include: speech recognition, text classification, natural language understanding, and natural language generation.'''

text
```

### Exact — Lab 1, cell 68

```python
# 2. Split this string to a list of substrings called words using space (default separator) as the separator.

words = text.split()
words
```

### Exact — Lab 1, cell 69

```python
# 3. Find the length of the new list.

len(words)
```

### Exact — Lab 1, cell 70

```python
# 4. Count how many times the item 'natural' appears in the words list.
# Hint: use the .count(item) method.

words.count('natural')
```

### Exact — Lab 1, cell 71

```python
# 5. Create a new list called definition_list that includes the items from the 7th to the 14th (inclusive) of the words list.
# Hint: Use slicing with the syntax list[start:end]. Remember that Python counts from 0 and the end index is exclusive.

definition_list = words[6:14]
definition_list
```

### Exact — Lab 1, cell 72

```python
# 6. Remove the dot (.) from the last item of the definition_list ('computer.').
# Hint:
## Save the last item in a new variable.
## Remove the dot using string methods (e.g. .replace(old, new)).
## Replace the last item in definition_list with the corrected string.

last_item = definition_list[-1]
last_item = last_item.replace('.','')
definition_list[-1] = last_item
definition_list
```

### Exact — Lab 1, cell 73

```python
# 7. Create a new string called definition by joining the items of the definition_list.
# Hint: Use the 'separator'.join(list_name) method.

definition = ' '.join(definition_list)
definition
```

### Exact — Lab 1, cell 79

```python
string = 'Hello, world!'

if ('a' in string) and ('o' in string):
    print("Both 'a' and 'o' are present")
elif 'a' in string:
    print("Only 'a' is present")
elif 'o' in string:
    print("Only 'o' is present")
else:
    print("Neither 'a' nor 'o' is present")
```

### Exact — Lab 1, cell 85

```python
# definition_list = ['processing', 'of', 'natural', 'language', 'information', 'by', 'a', 'computer']

for word in definition_list:
    print(word[0].upper())
```

### Exact — Lab 1, cell 91

```python
def compute_final_price(original_price, discount):
    return original_price*(100-discount)/100
```

### Exact — Lab 1, cell 95

```python
def odd_or_even(number):
    if number%2==0:
        print(number,'is an even number')
    else:
        print(number,'is an odd number')

numbers = [1,3,6,8,31,37,50]

for num in numbers:
    odd_or_even(num)
```

### Exact — Lab 1, cell 98

```python
squares = [i**2 for i in range(1,11)]

squares
```

### Exact — Lab 1, cell 101

```python
# definition_list = ['processing', 'of', 'natural', 'language', 'information', 'by', 'a', 'computer']

new_list = [word for word in definition_list if 'a' in word]

new_list
```

### Exact — Lab 1, cell 108

```python
random.seed(123) # reproduce results

random_numbers = []

for i in range(15):
    random_numbers.append(random.randint(1,5))

random_numbers
```

### Exact — Lab 1, cell 109

```python
random_numbers.count(5)
```


## 9. Regex: extraction, matching and replacement

**Source:** `LAB2_solutions.ipynb`, cells 4, 6, 10, 14, 17, 19, 21, 22, 24, 25, 27, 29, 33, 35, 37, 39, 41, 43, 45.

**Purpose / explanation:** Search words/digits/hashtags, use anchors/boundaries, replacement, split, multiline.

**Required imports / context:** re.

**Limitations / project notes:** Examples are separate snippets; match may be None on different inputs; ASCII-style patterns do not cover all Unicode.

### Exact — Lab 2, cell 4

```python
import re
```

### Exact — Lab 2, cell 6

```python
text = "I have 2 cats and 14 dogs."
re.findall(r'\d+', text)
```

### Exact — Lab 2, cell 10

```python
text = "123abc"
re.match(r"\d+", text)
```

### Exact — Lab 2, cell 14

```python
text = "123abc"
m = re.match(r"\d+", text)
m.group()
```

### Exact — Lab 2, cell 17

```python
text = "012345"
re.fullmatch(r"\d+", text)
```

### Exact — Lab 2, cell 19

```python
text = "Call 123-456"
re.sub(r"\d", "X", text)
```

### Exact — Lab 2, cell 21

```python
text = "I have 2 cats and 14 dogs."
re.split(r'\d+', text)
```

### Exact — Lab 2, cell 22

```python
text = "I have 2 cats and 14 dogs."
re.split(r'\d+', text, maxsplit=1)
```

### Exact — Lab 2, cell 24

```python
text = 'The student and the professor study theology.'
```

### Exact — Lab 2, cell 25

```python
re.findall(r'the', text)
```

### Exact — Lab 2, cell 27

```python
re.findall(r'[tT]he', text)
```

### Exact — Lab 2, cell 29

```python
re.findall(r'\b[Tt]he\b', text)
```

### Exact — Lab 2, cell 33

```python
text = "Alice went to Paris with Bob and Clara."

re.findall(r'\b[A-Z]\w+', text)
```

### Exact — Lab 2, cell 35

```python
text = "Excited about #AI, #NLP, and #Python!"

re.findall(r'#\w+', text)
```

### Exact — Lab 2, cell 37

```python
list_strings = ['abc', '123', '(###)', 'epl499!', '!@#']

for s in list_strings:
    if re.fullmatch(r'\W+',s) :
        print(s)
```

### Exact — Lab 2, cell 39

```python
list_strings = ["Mr. Smith", "Mrs. Davis", "Doctor Brown", "Mr Brown"]

for s in list_strings:
    if re.match(r'(Mr|Mrs)\.', s):
        print(s)
```

### Exact — Lab 2, cell 41

```python
text = 'H  ello ,   Worl d   !  '

re.sub(r'\s', '', text)
```

### Exact — Lab 2, cell 43

```python
text = """Alice went to Paris.
Bob went to Rome.
Alice went to Berlin."""

re.findall(r"^Alice.*\.", text)
```

### Exact — Lab 2, cell 45

```python
text = """Alice went to Paris.
Bob went to Rome.
Alice went to Berlin."""

re.findall(r"^Alice.*\.", text, flags=re.MULTILINE)
```


## 10. Text normalization: contraction helper

**Source:** `LAB2_solutions.ipynb`, cells 53, 55.

**Purpose / explanation:** Use regex backreferences to expand selected contractions.

**Required imports / context:** re (imported c53).

**Limitations / project notes:** Needs text from earlier cells; not exhaustive and can discard stylistic distinctions; no extra cleaning for baseline.

### Exact — Lab 2, cell 53

```python
import re

def uncontract(text):
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)

    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ii]t)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Tt]here)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)

    return text
```

### Exact — Lab 2, cell 55

```python
text = uncontract(text)
```


## 11. Tokenization, vocabulary and filtering

**Source:** `LAB2_solutions.ipynb`, cells 59, 60, 63, 64, 66, 69, 70, 72, 74, 76, 78, 81, 82, 84, 86, 89, 90, 92, 94, 96, 98, 100, 105, 106.

**Purpose / explanation:** Tokenize and compare vocabulary sizes after lowercasing/punctuation/stopword filtering.

**Required imports / context:** nltk; word_tokenize, sent_tokenize; collections.Counter; string; nltk.corpus.stopwords.

**Limitations / project notes:** Requires text (lab Gutenberg); dimensionality_info reads global tokens; alternatives are not successive required stages. isalpha excludes tokens with digits/punctuation, not just pure punctuation.

### Exact — Lab 2, cell 59

```python
# Import libraries and download the necessery NLTK resources for tokenization

import nltk

from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')
```

### Exact — Lab 2, cell 60

```python
tokens = word_tokenize(text)
```

### Exact — Lab 2, cell 63

```python
sentences = sent_tokenize(text)
```

### Exact — Lab 2, cell 64

```python
print('Number of Tokens:       ', len(tokens))
print('Number of Unique Tokens:', len(set(tokens)))
print('Number of Sentences:    ', len(sentences))
```

### Exact — Lab 2, cell 66

```python
from collections import Counter

print('10 most common tokens:\n')

for t, f in Counter(tokens).most_common(10): print('{0:10} {1}'.format(t, f))
```

### Exact — Lab 2, cell 69

```python
#########################################
# Define an informative function to     #
# monitor the dimensionality of our     #
# data compared to the raw / uncleaned. #
#########################################

def dimensionality_info(text_tokens):

    print('Number of Tokens:                            ', len(text_tokens))
    print('Number of Unique Tokens:                     ', len(set(text_tokens)))
    print()
    print('Number of Tokens before normalization:       ', len(tokens))
    print('Number of Unique Tokens before normalization:', len(set(tokens)))
```

### Exact — Lab 2, cell 70

```python
dimensionality_info(tokens)
```

### Exact — Lab 2, cell 72

```python
clean_tokens = [t.lower() for t in tokens]
```

### Exact — Lab 2, cell 74

```python
dimensionality_info(clean_tokens)
```

### Exact — Lab 2, cell 76

```python
import string
print(string.punctuation)
```

### Exact — Lab 2, cell 78

```python
text_punctuations = list(
    filter(
        lambda t: all(c.isdigit() or c in string.punctuation for c in t), clean_tokens
    )
)
```

### Exact — Lab 2, cell 81

```python
text_punctuations = set(text_punctuations)
```

### Exact — Lab 2, cell 82

```python
clean_tokens_1 = [t for t in clean_tokens if t not in text_punctuations]
print(clean_tokens_1)
```

### Exact — Lab 2, cell 84

```python
clean_tokens_2 = [t for t in clean_tokens if t.isalpha()]
print(clean_tokens_2)
```

### Exact — Lab 2, cell 86

```python
text_punctuations = list(
    filter(
        lambda t: not t.isalpha(), clean_tokens
    )
)
```

### Exact — Lab 2, cell 89

```python
dimensionality_info(clean_tokens_1)
```

### Exact — Lab 2, cell 90

```python
dimensionality_info(clean_tokens_2)
```

### Exact — Lab 2, cell 92

```python
clean_tokens = clean_tokens_2
```

### Exact — Lab 2, cell 94

```python
from nltk.corpus import stopwords
nltk.download('stopwords')
```

### Exact — Lab 2, cell 96

```python
stop_words = stopwords.words('english')
print(stop_words)
```

### Exact — Lab 2, cell 98

```python
text_stopwords = [t for t in clean_tokens if t in stop_words]

print('Most common Stopwords:\n')

for t, f in Counter(text_stopwords).most_common(10): print('{0:25} {1}'.format(t, f))
```

### Exact — Lab 2, cell 100

```python
clean_tokens = [t for t in clean_tokens if t not in stop_words]
```

### Exact — Lab 2, cell 105

```python
print('Most common Tokens:\n')

for t, f in Counter(clean_tokens).most_common(10): print('{0:25} {1}'.format(t, f))
```

### Exact — Lab 2, cell 106

```python
dimensionality_info(clean_tokens)
```


## 12. Stemming and lemmatization

**Source:** `LAB2_solutions.ipynb`, cells 109, 110, 112, 113, 115, 117, 118.

**Purpose / explanation:** Compare stemmed and lemmatized forms and token statistics.

**Required imports / context:** nltk; nltk.stem.PorterStemmer, WordNetLemmatizer.

**Limitations / project notes:** Uses clean_tokens; no POS-aware lemmatizer here. Not part of active Lab8 preprocessing.

### Exact — Lab 2, cell 109

```python
from nltk.stem import PorterStemmer

ps = PorterStemmer()
```

### Exact — Lab 2, cell 110

```python
############################################
# Here is an example on a series of words. #
############################################

words = ["wait", "waiting", "waited", "waits"]

for w in words:

    root = ps.stem(w)

    print('-', f"{w:8} ->", root)
```

### Exact — Lab 2, cell 112

```python
stemmed_tokens = [ps.stem(t) for t in clean_tokens]
print(stemmed_tokens)
```

### Exact — Lab 2, cell 113

```python
dimensionality_info(stemmed_tokens)
```

### Exact — Lab 2, cell 115

```python
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
```

### Exact — Lab 2, cell 117

```python
lemma_tokens = [lemmatizer.lemmatize(t) for t in clean_tokens]
print(lemma_tokens)
```

### Exact — Lab 2, cell 118

```python
dimensionality_info(lemma_tokens)
```


## 13. POS tagging and named entities

**Source:** `LAB3_solutions.ipynb`, cells 12, 14, 15, 18, 24, 27, 29.

**Purpose / explanation:** Use NLTK/Penn tags and spaCy POS/NER for linguistic information.

**Required imports / context:** nltk; word_tokenize, pos_tag; spacy; spacy.cli.download.

**Limitations / project notes:** Requires installed models/resources; do not add external trained components without confirming assignment scope. Course is sentiment context.

### Exact — Lab 3, cell 12

```python
import nltk

nltk.download("tagsets")
nltk.download('tagsets_json')
nltk.help.upenn_tagset()
```

### Exact — Lab 3, cell 14

```python
import nltk
from nltk import word_tokenize, pos_tag

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

### Exact — Lab 3, cell 15

```python
sent   = "Importantly, the results show improvement."
tokens = word_tokenize(sent)
tags   = pos_tag(tokens)

print(tags)
```

### Exact — Lab 3, cell 18

```python
import spacy
from spacy.cli import download

# Download the small English model
download("en_core_web_sm")
```

### Exact — Lab 3, cell 24

```python
import spacy

nlp = spacy.load("en_core_web_sm")

sent = "Importantly, the results show improvement."
doc  = nlp(sent)

for token in doc:
    print(f"{token.text:<12} {token.pos_:<8} {token.tag_:<8}")
```

### Exact — Lab 3, cell 27

```python
import spacy

nlp = spacy.load("en_core_web_sm")

print("Named Entity labels (`ent.label_`):\n")
for label in nlp.get_pipe("ner").labels:
    print(f"{label:<12} {spacy.explain(label)}")
```

### Exact — Lab 3, cell 29

```python
text = "Apple is looking at buying a startup in San Francisco for $1 billion."
doc  = nlp(text)

for ent in doc.ents:
    print(f"{ent.text:20} | {ent.label_}")
```


## 14. Sentiment and parsing

**Source:** `LAB3_solutions.ipynb`, cells 34, 37, 41, 42, 44, 46, 47.

**Purpose / explanation:** Get sentiment scores, parse constituents/noun chunks/dependencies.

**Required imports / context:** nltk.sentiment.SentimentIntensityAnalyzer; textblob.TextBlob; stanza; nltk.Tree; spacy; displacy.

**Limitations / project notes:** Needs spaCy nlp from c24/27; parser/model downloads are optional lab tools, not mandatory. Stanza doc and spaCy doc are different types.

### Exact — Lab 3, cell 34

```python
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

texts = [
    "I love this product! It's amazing.",
    "This is okay, nothing special.",
    "I really hate this. It's the worst!"
]

for text in texts:
    scores = sia.polarity_scores(text)
    print(f"Text: {text}")
    print(f"Scores: {scores}\n")
```

### Exact — Lab 3, cell 37

```python
from textblob import TextBlob

text = "The movie was fantastic!"

blob = TextBlob(text)

print(f"Text:         {text}")
print(f"Polarity:     {blob.sentiment.polarity:.2f}")
print(f"Subjectivity: {blob.sentiment.subjectivity:.2f}")
```

### Exact — Lab 3, cell 41

```python
import stanza
from nltk import Tree

# Download the English Stanza model (run once)
stanza.download("en")
```

### Exact — Lab 3, cell 42

```python
# Create a Stanza pipeline with constituency parsing
stanza_nlp = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos,constituency",
    verbose=False
)

text = "The lawyer questioned the witness."

doc = stanza_nlp(text)

for sentence in doc.sentences:
    tree = Tree.fromstring(str(sentence.constituency))
    tree.pretty_print()
```

### Exact — Lab 3, cell 44

```python
text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text) # this is the SpaCy model

print("Noun phrases:\n")
for chunk in doc.noun_chunks:
    print(chunk.text)
```

### Exact — Lab 3, cell 46

```python
from spacy import displacy

text = "She gave me the book."
doc = nlp(text)

print("Dependency pairs:")
for token in doc:
    print(f"{token.text:<10} - {token.dep_:<10} - {token.head.text:<10}")
```

### Exact — Lab 3, cell 47

```python
displacy.render(doc, style="dep", jupyter=True)
```


## 15. Linguistic counts and sentiment grouping

**Source:** `LAB3_solutions.ipynb`, cells 50, 51, 56, 58, 61, 63, 67, 69, 70, 72, 73, 74, 75.

**Purpose / explanation:** Count adjectives, named entities and sentiment groups across toy reviews.

**Required imports / context:** warnings; nltk; spaCy; TextBlob; SentimentIntensityAnalyzer; collections.Counter; matplotlib.pyplot.

**Limitations / project notes:** Toy review list not project data; warning suppression from original setup should not hide project errors.

### Exact — Lab 3, cell 50

```python
import warnings

warnings.filterwarnings("ignore")

import nltk, spacy
from nltk import word_tokenize, pos_tag
from textblob import TextBlob

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

nlp = spacy.load("en_core_web_sm")
```

### Exact — Lab 3, cell 51

```python
reviews = [
    "Ironman is an amazing movie with a fantastic plot.",
    "Robert Downey Jr.'s performance in Ironman was amazing.",
    "The new Spiderman movie was terrible.",
    "The plot of the movie was fantastic, but the acting was terrible.",
    "I was really disappointed with the sequel.",
    "Ironman is a classic superhero film.",
    "Robert Downey Jr. is a fantastic actor.",
    "This action movie had breathtaking special effects.",
    "The storyline was a bit confusing.",
    "The director's vision was clearly executed in this film.",
    "The soundtrack of Ironman was amazing.",
    "I didn't enjoy the ending at all.",
    "Looking forward to the next movie in the series.",
    "The cinematography was stunning."
]
```

### Exact — Lab 3, cell 56

```python
all_adjectives = []

for review in reviews:
    tokens = word_tokenize(review)
    tags   = pos_tag(tokens)

    # Extract adjectives (tags starting with 'JJ')
    adjectives = [word for word, tag in tags if tag.startswith('JJ')]
    all_adjectives.extend(adjectives)

print(all_adjectives)
```

### Exact — Lab 3, cell 58

```python
from collections import Counter

# Count adjective frequencies
adjective_counts = Counter(all_adjectives)

print("Adjective frequencies:")
for adjective, count in adjective_counts.most_common(5):
    print(f"{adjective:15}: {count}")
```

### Exact — Lab 3, cell 61

```python
from collections import Counter

all_entities = []
entity_types = []

for review in reviews:
    doc = nlp(review)
    for ent in doc.ents:
        all_entities.append(ent.text)
        entity_types.append(ent.label_)

print(all_entities)
print(entity_types)
```

### Exact — Lab 3, cell 63

```python
entity_type_counts = Counter(entity_types)

print("\nNamed Entity Type frequencies:")
for entity_type, count in entity_type_counts.most_common():
    print(f"{entity_type:15}: {count}")
```

### Exact — Lab 3, cell 67

```python
# Find the most frequent entity type
most_common_type = entity_type_counts.most_common(1)[0][0]
print(f"Most frequent entity type: {most_common_type}")

for i in range(len(entity_types)):

    e_type = entity_types[i]
    if e_type == most_common_type:
        print(all_entities[i])
```

### Exact — Lab 3, cell 69

```python
import matplotlib.pyplot as plt
from textblob import TextBlob

sentiment_scores = []

for review in reviews:
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    sentiment_scores.append(polarity)

print(sentiment_scores)
```

### Exact — Lab 3, cell 70

```python
plt.figure(figsize=(12,3))
plt.hist(sentiment_scores, edgecolor='black')
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')

plt.show()
```

### Exact — Lab 3, cell 72

```python
sia = SentimentIntensityAnalyzer()

positive_reviews = []

for review in reviews:
    score = sia.polarity_scores(review)['compound']
    if score>0:
        positive_reviews.append(review)

positive_reviews
```

### Exact — Lab 3, cell 73

```python
adjectives_positive_reviews = []

for review in positive_reviews:
    tokens = word_tokenize(review)
    tags   = pos_tag(tokens)

    # Extract adjectives (tags starting with 'JJ')
    adjectives = [word for word, tag in tags if tag.startswith('JJ')]
    adjectives_positive_reviews.extend(adjectives)

adjectives_positive_reviews
```

### Exact — Lab 3, cell 74

```python
sia = SentimentIntensityAnalyzer()

negative_reviews = []

for review in reviews:
    score = sia.polarity_scores(review)['compound']
    if score<0:
        negative_reviews.append(review)

negative_reviews
```

### Exact — Lab 3, cell 75

```python
adjectives_negative_reviews = []

for review in negative_reviews:
    tokens = word_tokenize(review)
    tags   = pos_tag(tokens)

    # Extract adjectives (tags starting with 'JJ')
    adjectives = [word for word, tag in tags if tag.startswith('JJ')]
    adjectives_negative_reviews.extend(adjectives)

adjectives_negative_reviews
```


## 16. N-gram language modeling (not BoW/Word2Vec)

**Source:** `LAB4_solutions.ipynb`, cells 11, 12, 15, 18, 19, 20, 21, 27, 28, 30, 32, 33, 34, 35, 37, 41, 42, 43, 48, 50, 52, 54, 59, 63, 64, 65, 66, 68, 70, 72, 74, 78.

**Purpose / explanation:** Train MLE/Laplace n-gram models, compare probabilities and generate sentences; recreate consumed generator.

**Required imports / context:** nltk; word_tokenize; WordNetLemmatizer; itertools.chain; nltk.lm.preprocessing.padded_everygram_pipeline; nltk.lm.MLE, Laplace; math.

**Limitations / project notes:** Requires brp_sentences (c6–8 downloads/reads external transcript; not copied here). Lab only; not required project language model. Do not use external training corpus.

### Exact — Lab 4, cell 11

```python
import nltk

from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('punkt')

nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
```

### Exact — Lab 4, cell 12

```python
tokenized_texts = [word_tokenize(t) for t in brp_sentences]
tokenized_texts = [[t.lower() for t in tokens] for tokens in tokenized_texts]
tokenized_texts = [[lemmatizer.lemmatize(t) for t in tokens] for tokens in tokenized_texts]
```

### Exact — Lab 4, cell 15

```python
from itertools import chain

flat_tokens = list(chain.from_iterable(tokenized_texts)) # flatten list of tokens

vocabulary = list(set(flat_tokens)) # keep only unique tokens

print('Vocabulary size:', len(vocabulary))
```

### Exact — Lab 4, cell 18

```python
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace
```

### Exact — Lab 4, cell 19

```python
n = 1   # unigram
```

### Exact — Lab 4, cell 20

```python
# MLE - without smoothing

train_data, vocab = padded_everygram_pipeline(n, tokenized_texts)

model_mle = MLE(n)
model_mle.fit(train_data, vocab)
print(f"MLE model vocabulary size: {len(model_mle.vocab)}")
```

### Exact — Lab 4, cell 21

```python
# Laplace smoothing

# Need to recreate train_data since it's a generator that was consumed.
train_data, vocab = padded_everygram_pipeline(n, tokenized_texts)

model_laplace = Laplace(n)
model_laplace.fit(train_data, vocab)
print(f"Laplace model vocabulary size: {len(model_laplace.vocab)}")
```

### Exact — Lab 4, cell 27

```python
test_unigrams = [('i'), ('want'), ('meal'), ('hello'), ('CPU')]
```

### Exact — Lab 4, cell 28

```python
print(f"{'Unigram':<20} {'MLE':<12} {'Laplace':<12}")
print("-" * 45)

for w in test_unigrams:

    mle_prob = model_mle.score(w)
    laplace_prob = model_laplace.score(w)

    print(f"({w}){' ' * (18 - len(w))} {mle_prob:<12.6f} {laplace_prob:<12.6f}")
```

### Exact — Lab 4, cell 30

```python
print(f"{'Unigram':<20} {'MLE':<12} {'Laplace':<12} {'custom_MLE':<12} {'custom_Laplace':<12}")
print("-" * 75)

for w in test_unigrams:

    count_w = flat_tokens.count(w)

    N = len(flat_tokens)

    vocabulary_size = len(model_mle.vocab) # same for both models

    custom_mle_prob = count_w/N
    custom_laplace_prob = (count_w+1)/(N+vocabulary_size)

    # trained models
    mle_prob = model_mle.score(w)
    laplace_prob = model_laplace.score(w)

    print(f"({w}){' ' * (18 - len(w))} {mle_prob:<12.6f} {laplace_prob:<12.6f} {custom_mle_prob:<12.6f} {custom_laplace_prob:<12.6f}")
```

### Exact — Lab 4, cell 32

```python
import math

def generate_sentence(model, prefix=None, max_length=20):

    # find the n of the n-gram
    n_model = model.order

    # add '<s>' as the first token
    if prefix is None:
        prefix = ['<s>']
    else:
        if not prefix or prefix[0] != '<s>':
            prefix = ['<s>'] + prefix

    words = prefix.copy()

    # generate tokens until the end of sentence '</s' is generated or the maximum length is reached
    while words[-1] != '</s>' and len(words) < max_length:

        best_word, best_prob = None, 0

        # if the model is for unigram, then the context is an empty list
        if n_model==1 :
            context = []
        else :
            context = words[-n_model+1:]

        # find the probabilities of all words in the vocabulary and keep the word with the maximum probability
        for word in model.vocab:
            prob = model.score(word, context)
            if prob > best_prob:
                best_word, best_prob = word, prob

        if best_word:
            words.append(best_word)
        else:
            words.append('</s>')
            break

    # print result but first remove the start and end of sentence tokens ('<s>' and '</s' respectively)
    result_words = []
    for word in words:
        if word not in ['<s>', '</s>']:
            result_words.append(word)

    return ' '.join(result_words)
```

### Exact — Lab 4, cell 33

```python
# Test with different starting phrases
test_phrases = ["i want", "i do not", "where", "where could i get"]

# preprocessing - same steps as for the training set

test_phrases = [word_tokenize(t) for t in test_phrases]
test_phrases = [[t.lower() for t in tokens] for tokens in test_phrases]
test_phrases = [[lemmatizer.lemmatize(t) for t in tokens] for tokens in test_phrases]
```

### Exact — Lab 4, cell 34

```python
print("=== SENTENCE GENERATION (Most Probable Path) MLE ===")
for phrase in test_phrases:
    sentence = generate_sentence(model_mle, phrase)
    print(f"'{phrase}' → '{sentence}'")
```

### Exact — Lab 4, cell 35

```python
print("=== SENTENCE GENERATION (Most Probable Path) Laplace ===")
for phrase in test_phrases:
    sentence = generate_sentence(model_laplace, phrase)
    print(f"'{phrase}' → '{sentence}'")
```

### Exact — Lab 4, cell 37

```python
text = 'i want'

' '.join([text]+model_mle.generate(20, text_seed=[text], random_seed=123))
```

### Exact — Lab 4, cell 41

```python
# Define n for bigram

n = 2   # bigram
```

### Exact — Lab 4, cell 42

```python
# Train model_mle and print the vocabulary size.

train_data, vocab = padded_everygram_pipeline(n, tokenized_texts)

model_mle = MLE(n)
model_mle.fit(train_data, vocab)
print(f"MLE model vocabulary size: {len(model_mle.vocab)}")
```

### Exact — Lab 4, cell 43

```python
# Train model_laplace and print the vocabulary size.

# Need to recreate train_data since it's a generator that was consumed.
train_data, vocab = padded_everygram_pipeline(n, tokenized_texts)

model_laplace = Laplace(n)
model_laplace.fit(train_data, vocab)
print(f"Laplace model vocabulary size: {len(model_laplace.vocab)}")
```

### Exact — Lab 4, cell 48

```python
test_bigrams = [
    ('i', 'want'),          # Very frequent
    ('want', 'to'),         # Frequent
    ('to', 'go'),           # Less frequent
    ('expensive', 'meal'),  # Rare
    ('want', 'pizza'),      # Unseen bigram
    ('hello', 'world'),     # Completely unseen
]
```

### Exact — Lab 4, cell 50

```python
print(f"{'Bigram':<20} {'MLE':<12} {'Laplace':<12}")
print("-" * 45)

for w1, w2 in test_bigrams:

    mle_prob = model_mle.score(w2, [w1])
    laplace_prob = model_laplace.score(w2, [w1])

    print(f"({w1}, {w2}){' ' * (16 - len(w1) - len(w2))} {mle_prob:<12.6f} {laplace_prob:<12.6f}")
```

### Exact — Lab 4, cell 52

```python
print(f"{'Bigram':<20} {'MLE':<12} {'Laplace':<12} {'custom_MLE':<12} {'custom_Laplace':<12}")
print("-" * 75)

for w1, w2 in test_bigrams:

    count_w1w2 = 0
    count_context = 0

    for tokens in tokenized_texts:
        for j in range(len(tokens)-1) :
            if tokens[j]==w1 and tokens[j+1]==w2:
                count_w1w2 += 1

        count_context += tokens.count(w1)

    vocabulary_size = len(model_mle.vocab) # same for both models

    custom_mle_prob = count_w1w2/count_context if count_context != 0 else 0
    custom_laplace_prob = (count_w1w2+1)/(count_context+vocabulary_size)

    # trained models
    mle_prob = model_mle.score(w2, [w1])
    laplace_prob = model_laplace.score(w2, [w1])

    print(f"({w1}, {w2}){' ' * (16 - len(w1) - len(w2))} {mle_prob:<12.6f} {laplace_prob:<12.6f} {custom_mle_prob:<12.6f} {custom_laplace_prob:<12.6f}")
```

### Exact — Lab 4, cell 54

```python
import math

def generate_sentence(model, prefix=None, max_length=20):

    n_model = model.order

    if prefix is None:
        prefix = ['<s>']
    else:
        if not prefix or prefix[0] != '<s>':
            prefix = ['<s>'] + prefix

    words = prefix.copy()

    while words[-1] != '</s>' and len(words) < max_length:

        best_word, best_prob = None, 0
        context = words[-n_model+1:]

        for word in model.vocab:
            prob = model.score(word, context)
            if prob > best_prob:
                best_word, best_prob = word, prob

        if best_word:
            words.append(best_word)
        else:
            words.append('</s>')
            break

    result_words = []
    for word in words:
        if word not in ['<s>', '</s>']:
            result_words.append(word)

    return ' '.join(result_words)
```

### Exact — Lab 4, cell 59

```python
text = 'i'

' '.join([text]+model_mle.generate(20, text_seed=[text], random_seed=123))
```

### Exact — Lab 4, cell 63

```python
# Define n - for example n=4

n = 4
```

### Exact — Lab 4, cell 64

```python
# Train model_mle and print the vocabulary size.

train_data, vocab = padded_everygram_pipeline(n, tokenized_texts)

model_mle = MLE(n)
model_mle.fit(train_data, vocab)
print(f"MLE model vocabulary size: {len(model_mle.vocab)}")
```

### Exact — Lab 4, cell 65

```python
# Train model_laplace and print the vocabulary size.

# Need to recreate train_data since it's a generator that was consumed.
train_data, vocab = padded_everygram_pipeline(n, tokenized_texts)

model_laplace = Laplace(n)
model_laplace.fit(train_data, vocab)
print(f"Laplace model vocabulary size: {len(model_laplace.vocab)}")
```

### Exact — Lab 4, cell 66

```python
test_ngrams = [
    'i want to go',
    'more than five minutes',
    'to spend more than'
]
```

### Exact — Lab 4, cell 68

```python
# preprocessing

test_ngrams = [word_tokenize(t) for t in test_ngrams]
test_ngrams = [[t.lower() for t in tokens] for tokens in test_ngrams]
test_ngrams = [[lemmatizer.lemmatize(t) for t in tokens] for tokens in test_ngrams]
```

### Exact — Lab 4, cell 70

```python
print(f"{'Ngram':<32} {'MLE':<12} {'Laplace':<12}")
print("-" * 55)

for tokens in test_ngrams:

    mle_prob = model_mle.score(tokens[-1], tokens[-n:-1])
    laplace_prob = model_laplace.score(tokens[-1], tokens[-n:-1])

    print(f"({' '.join(tokens)}){' ' * (30 - len(' '.join(tokens)))} {mle_prob:<12.6f} {laplace_prob:<12.6f}")
```

### Exact — Lab 4, cell 72

```python
print(f"{'Ngram':<32} {'MLE':<12} {'Laplace':<12} {'custom_MLE':<12} {'custom_Laplace':<12}")
print("-" * 90)

for text in test_ngrams:

    count_text = 0
    count_context = 0

    for tokens in tokenized_texts:
        for j in range(len(tokens)-n) :
            if tokens[j:j+n]==text:
                count_text += 1 # count appearances of whole text

        for j in range(len(tokens)-n+1) :
            if tokens[j:j+n-1]==text[:-1]:
                count_context += 1 # count appearances of context (text - last token)

    vocabulary_size = len(model_mle.vocab) # same for both models

    custom_mle_prob = count_text/count_context if count_context != 0 else 0
    custom_laplace_prob = (count_text+1)/(count_context+vocabulary_size)

    # trained models
    mle_prob = model_mle.score(text[-1], text[-n:-1])
    laplace_prob = model_laplace.score(text[-1], text[-n:-1])

    print(f"({' '.join(text)}){' ' * (30 - len(' '.join(text)))} {mle_prob:<12.6f} {laplace_prob:<12.6f} {custom_mle_prob:<12.6f} {custom_laplace_prob:<12.6f}")
```

### Exact — Lab 4, cell 74

```python
import math

def generate_sentence(model, prefix=None, max_length=20):

    n_model = model.order

    if prefix is None:
        prefix = ['<s>']
    else:
        if not prefix or prefix[0] != '<s>':
            prefix = ['<s>'] + prefix

    words = prefix.copy()

    while words[-1] != '</s>' and len(words) < max_length:

        best_word, best_prob = None, 0
        context = words[-n_model+1:]

        for word in model.vocab:
            prob = model.score(word, context)
            if prob > best_prob:
                best_word, best_prob = word, prob

        if best_word:
            words.append(best_word)
        else:
            words.append('</s>')
            break

    result_words = []
    for word in words:
        if word not in ['<s>', '</s>']:
            result_words.append(word)

    return ' '.join(result_words)
```

### Exact — Lab 4, cell 78

```python
text = 'i'

' '.join([text]+model_mle.generate(20, text_seed=[text], random_seed=123))
```


## 17. Pandas and splitting classification data

**Source:** `LAB5_solutions.ipynb`, cells 5, 16, 18, 20, 26, 27, 28, 29, 88, 89, 90, 91, 93, 94, 96, 97, 99, 100, 101, 102, 104, 105, 107, 108, 110, 111, 113, 114.

**Purpose / explanation:** Build text/label DataFrame, split; demonstrate simple dataframe operations and CSV I/O.

**Required imports / context:** nltk.corpus.movie_reviews; pandas as pd; sklearn.model_selection.train_test_split.

**Limitations / project notes:** Movie-review examples and celebrity examples are lab-only. Use shared CSV loader in actual project. Notebook cells can overwrite variables.

### Exact — Lab 5, cell 5

```python
import nltk
from nltk.corpus import movie_reviews

nltk.download("movie_reviews")
```

### Exact — Lab 5, cell 16

```python
reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]

print("Total number of reviews:", len(reviews))
```

### Exact — Lab 5, cell 18

```python
labels = [1 if fileid.startswith('pos') else 0 for fileid in movie_reviews.fileids()]

print(labels.count(1), "positive reviews")
print(labels.count(0), "negative reviews")
```

### Exact — Lab 5, cell 20

```python
import pandas as pd

df = pd.DataFrame({'text':reviews,'label':labels})
```

### Exact — Lab 5, cell 26

```python
from sklearn.model_selection import train_test_split
```

### Exact — Lab 5, cell 27

```python
df_train, df_test = train_test_split(df, test_size=0.20, random_state=123)
```

### Exact — Lab 5, cell 28

```python
print("Number of reviews in training dataset: ", df_train.shape[0])
print("Number of reviews in test dataset:     ", df_test.shape[0])
```

### Exact — Lab 5, cell 29

```python
print("Number of positive reviews in training dataset: ", sum(df_train.label==1))
print("Number of negative reviews in training dataset: ", sum(df_train.label==0))

print("\n")

print("Number of positive reviews in test dataset:     ", sum(df_test.label==1))
print("Number of negative reviews in test dataset:     ", sum(df_test.label==0))
```

### Exact — Lab 5, cell 88

```python
import pandas as pd
```

### Exact — Lab 5, cell 89

```python
names = ["Taylor Swift",
        "Leonardo DiCaprio",
        "Beyoncé",
        "Elon Musk",
        "Rihanna",
        "Tom Cruise",
        "Adele",
        "Chris Hemsworth",
        "Jennifer Lopez",
        "Dwayne Johnson"]

age = [36, 51, 45, 55, 38, 64, 38, 43, 57, 54]
```

### Exact — Lab 5, cell 90

```python
celebrities_dict = {'name':names, 'age':age}

celebrities_dict
```

### Exact — Lab 5, cell 91

```python
celebrities_df = pd.DataFrame(celebrities_dict)

celebrities_df
```

### Exact — Lab 5, cell 93

```python
children = [0, 0, 3, 14, 3, 3, 1, 3, 2, 3]
```

### Exact — Lab 5, cell 94

```python
celebrities_df['children'] = children

celebrities_df
```

### Exact — Lab 5, cell 96

```python
celebrities_df.head(3)
```

### Exact — Lab 5, cell 97

```python
celebrities_df.tail(3)
```

### Exact — Lab 5, cell 99

```python
celebrities_df.shape
```

### Exact — Lab 5, cell 100

```python
celebrities_df.columns
```

### Exact — Lab 5, cell 101

```python
celebrities_df.dtypes
```

### Exact — Lab 5, cell 102

```python
celebrities_df.describe()
```

### Exact — Lab 5, cell 104

```python
celebrities_df['age'].mean()
```

### Exact — Lab 5, cell 105

```python
celebrities_df['children'].sum()
```

### Exact — Lab 5, cell 107

```python
celebrities_df['age'].plot.hist()
```

### Exact — Lab 5, cell 108

```python
celebrities_df['children'].plot.hist()
```

### Exact — Lab 5, cell 110

```python
# find index of maximum value

celebrities_df['children'].idxmax()
```

### Exact — Lab 5, cell 111

```python
celebrities_df.iloc[celebrities_df['children'].idxmax()]
```

### Exact — Lab 5, cell 113

```python
celebrities_df.to_csv('celebrities.csv', index=False)
```

### Exact — Lab 5, cell 114

```python
celebrities = pd.read_csv('celebrities.csv')
celebrities
```


## 18. Classification preprocessing before BoW

**Source:** `LAB5_solutions.ipynb`, cells 31, 32, 34, 39.

**Purpose / explanation:** Uncontract, tokenize, lower, filter, remove stopwords, lemmatize.

**Required imports / context:** IPython.display.clear_output; nltk; word_tokenize; stopwords; WordNetLemmatizer; re; other original imports retained.

**Limitations / project notes:** Lab sentiment preprocessing is not required by assignment; baseline explicitly raw text. Check actual helper order, use same preprocessing for both splits.

### Exact — Lab 5, cell 31

```python
from IPython.display import clear_output

clear_output()

import nltk

nltk.download("gutenberg")
from nltk.corpus import gutenberg

from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import PorterStemmer

nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

import re
import string
```

### Exact — Lab 5, cell 32

```python
# uncontract() function

def uncontract(text):
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)

    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ii]t)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Tt]here)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)

    return text
```

### Exact — Lab 5, cell 34

```python
def preprocessing_text(text):
    text = uncontract(text)
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t.isalpha()]

    stop_words = stopwords.words('english')
    tokens = [t for t in tokens if t not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens
```

### Exact — Lab 5, cell 39

```python
clean_tokens_train = [preprocessing_text(t) for t in df_train['text']]
clean_tokens_test  = [preprocessing_text(t) for t in df_test['text']]
```


## 19. CountVectorizer: training-only vocabulary, unigrams and bigrams

**Source:** `LAB5_solutions.ipynb`, cells 47, 48, 49, 50, 53, 57, 58, 59.

**Purpose / explanation:** Fit count vocabulary on training data, transform test; inspect top n-grams.

**Required imports / context:** sklearn.feature_extraction.text.CountVectorizer; pandas as pd.

**Limitations / project notes:** Lab 1000 max features, min_df10/max_df0.80, explicit no lowercase since already processed. Assignment baseline defaults except exact required params; avoid dense matrices if not needed. Helper returns matrices, not fitted vectorizer, so raw prediction wrapper must store it.

### Exact — Lab 5, cell 47

```python
from sklearn.feature_extraction.text import CountVectorizer
```

### Exact — Lab 5, cell 48

```python
def bow_features(training_data, test_data, ngram_range, max_features):

    n_grams = CountVectorizer(
        ngram_range  = ngram_range,
        max_features = max_features,
        lowercase    = False,
        tokenizer    = None,
        preprocessor = None,
        stop_words   = None,
        min_df       = 10,
        max_df       = 0.80
        )

    bow_train = n_grams.fit_transform(training_data)

    bow_train = bow_train.toarray()
    bow_train = pd.DataFrame(bow_train)
    bow_train.columns = n_grams.get_feature_names_out()

    bow_test = n_grams.transform(test_data)

    bow_test = bow_test.toarray()
    bow_test = pd.DataFrame(bow_test)
    bow_test.columns = n_grams.get_feature_names_out()

    return bow_train, bow_test
```

### Exact — Lab 5, cell 49

```python
# join tokens after preprocessing

clean_text_train = [' '.join(tokens) for tokens in clean_tokens_train]
clean_text_test  = [' '.join(tokens) for tokens in clean_tokens_test]
```

### Exact — Lab 5, cell 50

```python
bow_train_unigrams, bow_test_unigrams = bow_features(clean_text_train, clean_text_test, (1,1), 1000) # for unigrams
```

### Exact — Lab 5, cell 53

```python
bow_train_bigrams, bow_test_bigrams = bow_features(clean_text_train, clean_text_test, (2,2), 1000) # for bigrams
```

### Exact — Lab 5, cell 57

```python
def get_top_ngrams(bow, n=10):

    sum_words = bow.sum().sort_values(ascending=False)[0:n]

    return sum_words
```

### Exact — Lab 5, cell 58

```python
get_top_ngrams(bow_train_unigrams, 10)
```

### Exact — Lab 5, cell 59

```python
get_top_ngrams(bow_train_bigrams, 10)
```


## 20. Multinomial Naive Bayes, evaluation and CV

**Source:** `LAB5_solutions.ipynb`, cells 62, 64, 66, 68, 71, 74, 76, 81, 82, 83.

**Purpose / explanation:** Fit NB, predict labels, report and compare n-grams, 5-fold CV.

**Required imports / context:** sklearn.naive_bayes.MultinomialNB; sklearn.metrics.classification_report, confusion_matrix, ConfusionMatrixDisplay; matplotlib.pyplot; sklearn.model_selection.cross_val_score.

**Limitations / project notes:** Lab CV uses already vectorized train data, so inner fold feature leakage possible. Refit representation inside folds for valid extra CV. Exact baseline uses alpha=1.0; no extra preprocessing.

### Exact — Lab 5, cell 62

```python
from sklearn.naive_bayes import MultinomialNB
```

### Exact — Lab 5, cell 64

```python
clf_unigrams = MultinomialNB()
```

### Exact — Lab 5, cell 66

```python
X_train = bow_train_unigrams
y_train = df_train['label']

clf_unigrams.fit(X_train, y_train)
```

### Exact — Lab 5, cell 68

```python
X_test = bow_test_unigrams

y_pred = clf_unigrams.predict(X_test)
```

### Exact — Lab 5, cell 71

```python
from sklearn.metrics import classification_report

print(classification_report(df_test['label'], y_pred))
```

### Exact — Lab 5, cell 74

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create and display the confusion matrix
cm = confusion_matrix(df_test['label'], y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
```

### Exact — Lab 5, cell 76

```python
clf_bigrams = MultinomialNB()

X_train = bow_train_bigrams
y_train = df_train['label']

clf_bigrams.fit(X_train, y_train)

X_test = bow_test_bigrams

y_pred = clf_bigrams.predict(X_test)

print(classification_report(df_test['label'], y_pred))
```

### Exact — Lab 5, cell 81

```python
from sklearn.model_selection import cross_val_score
```

### Exact — Lab 5, cell 82

```python
clf = MultinomialNB()

scores = cross_val_score(clf, bow_train_unigrams, df_train['label'], cv=5, scoring='accuracy')
print("Cross-validation scores - accuracy for 5-folds:", scores)
print("Average accuracy:", scores.mean())
```

### Exact — Lab 5, cell 83

```python
clf = MultinomialNB()

scores = cross_val_score(clf, bow_train_bigrams, df_train['label'], cv=5, scoring='accuracy')
print("Cross-validation scores - accuracy for 5-folds:", scores)
print("Average accuracy:", scores.mean())
```


## 21. Engineered lexical and linguistic features

**Source:** `LAB6_solutions.ipynb`, cells 49, 50, 54, 55, 57, 59, 63, 67, 68, 70, 75, 77, 78, 82, 83, 84, 86, 87.

**Purpose / explanation:** Count positive/negative tokens, difference, no/exclamation indicators, log token count; merge features.

**Required imports / context:** nltk.corpus.opinion_lexicon; pandas as pd; nltk.word_tokenize; math.log.

**Limitations / project notes:** Requires preprocessed tokens and bow matrices from c30–45; log(0) undefined; negative difference invalid for NB. Preserve raw text for style features. Lexicon use subject to project data policy.

### Exact — Lab 6, cell 49

```python
import nltk
from nltk.corpus import opinion_lexicon

nltk.download('opinion_lexicon')
```

### Exact — Lab 6, cell 50

```python
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

print("Number of positive words:", len(positive_words))
print("Number of negative words:", len(negative_words))
```

### Exact — Lab 6, cell 54

```python
def count_lexical_features(tokens):
    pos_count = sum(1 for w in tokens if w in positive_words)
    neg_count = sum(1 for w in tokens if w in negative_words)

    return {'positive_tokens': pos_count, 'negative_tokens': neg_count}
```

### Exact — Lab 6, cell 55

```python
count_lexical_features(preprocessing_text("The movie was amazing and full of wonderful scenes, but the ending was terrible and disappointing."))
```

### Exact — Lab 6, cell 57

```python
lexical_features_train_dict = [count_lexical_features(tokens) for tokens in clean_tokens_train]
lexical_features_test_dict  = [count_lexical_features(tokens) for tokens in clean_tokens_test]
```

### Exact — Lab 6, cell 59

```python
lexical_features_train = pd.DataFrame(lexical_features_train_dict)
lexical_features_test  = pd.DataFrame(lexical_features_test_dict)
```

### Exact — Lab 6, cell 63

```python
lexical_features_train['sentiment_difference'] = lexical_features_train['positive_tokens']-lexical_features_train['negative_tokens']
lexical_features_test['sentiment_difference']  = lexical_features_test['positive_tokens']-lexical_features_test['negative_tokens']
```

### Exact — Lab 6, cell 67

```python
raw_text_train = [text for text in df_train['text']]
raw_text_test  = [text for text in df_test['text']]
```

### Exact — Lab 6, cell 68

```python
tokens_train = [word_tokenize(text) for text in raw_text_train]
tokens_test  = [word_tokenize(text) for text in raw_text_test]
```

### Exact — Lab 6, cell 70

```python
existence_no_train = [1 if 'no' in tokens else 0 for tokens in tokens_train]
existence_no_test  = [1 if 'no' in tokens else 0 for tokens in tokens_test]
```

### Exact — Lab 6, cell 75

```python
existence_exclamation_train = [1 if '!' in text else 0 for text in raw_text_train]
existence_exclamation_test  = [1 if '!' in text else 0 for text in raw_text_test]
```

### Exact — Lab 6, cell 77

```python
from math import log
```

### Exact — Lab 6, cell 78

```python
log_tokens_train = [log(len(tokens)) for tokens in tokens_train]
log_tokens_test  = [log(len(tokens)) for tokens in tokens_test]
```

### Exact — Lab 6, cell 82

```python
# Using the next command, we can merge the two dataframes on their index.
# Check https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html for more options

X_train = pd.merge(bow_train, lexical_features_train, left_index=True, right_index=True)
X_train.head()
```

### Exact — Lab 6, cell 83

```python
X_train['existence_no'] = existence_no_train
X_train['existence_!'] = existence_exclamation_train
X_train['log_number_tokens'] = log_tokens_train
X_train.head()
```

### Exact — Lab 6, cell 84

```python
y_train = df_train['label']
y_train.head()
```

### Exact — Lab 6, cell 86

```python
X_test = pd.merge(bow_test, lexical_features_test, left_index=True, right_index=True)
X_test['existence_no'] = existence_no_test
X_test['existence_!'] = existence_exclamation_test
X_test['log_number_tokens'] = log_tokens_test
X_test.head()
```

### Exact — Lab 6, cell 87

```python
y_test = df_test['label']
y_test.head()
```


## 22. Feature selection and NB constraints

**Source:** `LAB6_solutions.ipynb`, cells 90, 92, 93, 94, 95, 96, 99, 100, 101, 103, 107, 109, 110, 112.

**Purpose / explanation:** Original first NB attempt fails with negative features; remove difference, RF rank top features, compare models.

**Required imports / context:** sklearn.naive_bayes.MultinomialNB; sklearn.ensemble.RandomForestClassifier; sklearn.metrics.accuracy_score, classification_report; pandas; matplotlib; cross_val_score.

**Limitations / project notes:** c90 is intentional failure; c103 excludes negative sentiment_difference, not required for LR. CV after selection may leak inner folds. RF importance does not establish causal contribution.

### Exact — Lab 6, cell 90

```python
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X_train, y_train)
```

### Exact — Lab 6, cell 92

```python
X_train_nb = X_train.drop(columns=['sentiment_difference'])
X_test_nb  = X_test.drop(columns=['sentiment_difference'])
```

### Exact — Lab 6, cell 93

```python
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X_train_nb, y_train)
```

### Exact — Lab 6, cell 94

```python
y_pred = clf.predict(X_test_nb)
```

### Exact — Lab 6, cell 95

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```

### Exact — Lab 6, cell 96

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

### Exact — Lab 6, cell 99

```python
from sklearn.ensemble import RandomForestClassifier

# train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=123)
model.fit(X_train, y_train)

# get feature importances
importances = model.feature_importances_

# create a DataFrame with feature names and their importance
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
})
```

### Exact — Lab 6, cell 100

```python
# sort features by importance and get the top 100
top_features = feature_importances.sort_values(by='importance', ascending=False).head(100)
```

### Exact — Lab 6, cell 101

```python
import matplotlib.pyplot as plt

# select top 20 features for plotting
top_n = top_features.head(20)

# plot
plt.figure(figsize=(10,6))
plt.barh(top_n['feature'], top_n['importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Features')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()
```

### Exact — Lab 6, cell 103

```python
from sklearn.ensemble import RandomForestClassifier

def get_top_features(X_train, y_train, n):

    model = RandomForestClassifier(n_estimators=100, random_state=123)
    model.fit(X_train.drop(columns=['sentiment_difference']), y_train)

    importances = model.feature_importances_

    feature_importances = pd.DataFrame({
        'feature': X_train.drop(columns=['sentiment_difference']).columns,
        'importance': importances
    })

    top_features = feature_importances.sort_values(by='importance', ascending=False).head(n)

    return top_features['feature'].tolist()
```

### Exact — Lab 6, cell 107

```python
top_100 = get_top_features(X_train, y_train, 100)
```

### Exact — Lab 6, cell 109

```python
clf2 = MultinomialNB()

clf2.fit(X_train[top_100],y_train)

y_pred2 = clf2.predict(X_test[top_100])

accuracy_score(y_test, y_pred2)
```

### Exact — Lab 6, cell 110

```python
print(classification_report(y_test, y_pred2))
```

### Exact — Lab 6, cell 112

```python
from sklearn.model_selection import cross_val_score

clf1 = MultinomialNB()
scores1 = cross_val_score(clf1, X_train_nb, y_train, cv=5)
print('Mean accuracy using all features:', round(scores1.mean(),3))

clf2 = MultinomialNB()
scores2 = cross_val_score(clf2, X_train[top_100], y_train, cv=5)
print('Mean accuracy using top 100 features:', round(scores2.mean(),3))
```


## 23. TF-IDF: train and test representations

**Source:** `LAB7_solutions.ipynb`, cells 36, 37, 38, 39, 40.

**Purpose / explanation:** Fit TF-IDF on clean train, transform clean test, unigrams/bigrams.

**Required imports / context:** sklearn.feature_extraction.text.TfidfVectorizer; pandas as pd.

**Limitations / project notes:** Uses lab parameters not assignment; dense conversion only for inspection; store fitted vectorizer for raw prediction.

### Exact — Lab 7, cell 36

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

### Exact — Lab 7, cell 37

```python
# join tokens after preprocessing

clean_text_train = [' '.join(tokens) for tokens in clean_tokens_train]
clean_text_test  = [' '.join(tokens) for tokens in clean_tokens_test]
```

### Exact — Lab 7, cell 38

```python
def tfidf_features(training_data, test_data, ngram_range, max_features):

    tfidf = TfidfVectorizer(
        ngram_range  = ngram_range,
        max_features = max_features,
        lowercase    = False,
        tokenizer    = None,
        preprocessor = None,
        stop_words   = None,
        min_df       = 10,
        max_df       = 0.80
        )

    tfidf_train = tfidf.fit_transform(training_data)

    tfidf_train = tfidf_train.toarray()
    tfidf_train = pd.DataFrame(tfidf_train)
    tfidf_train.columns = tfidf.get_feature_names_out()

    tfidf_test = tfidf.transform(test_data)

    tfidf_test = tfidf_test.toarray()
    tfidf_test = pd.DataFrame(tfidf_test)
    tfidf_test.columns = tfidf.get_feature_names_out()

    return tfidf_train, tfidf_test
```

### Exact — Lab 7, cell 39

```python
tfidf_train_unigrams, tfidf_test_unigrams = tfidf_features(clean_text_train, clean_text_test, (1,1), 1000)
```

### Exact — Lab 7, cell 40

```python
tfidf_train_bigrams, tfidf_test_bigrams = tfidf_features(clean_text_train, clean_text_test, (2,2), 1000)
```


## 24. SVD dimensionality reduction (optional)

**Source:** `LAB7_solutions.ipynb`, cells 47, 48, 49, 51, 53, 57, 58, 60.

**Purpose / explanation:** Reduce TF-IDF to 50 components, inspect variance/singular values, transform held-out using trained SVD.

**Required imports / context:** sklearn.decomposition.TruncatedSVD; pandas as pd.

**Limitations / project notes:** Not required in project; not a replacement for required plain TF-IDF. Components less interpretable.

### Exact — Lab 7, cell 47

```python
from sklearn.decomposition import TruncatedSVD
```

### Exact — Lab 7, cell 48

```python
svd = TruncatedSVD(n_components=50, random_state=42)
```

### Exact — Lab 7, cell 49

```python
# train model and transform training data
tfidf_train_unigrams_svd = svd.fit_transform(tfidf_train_unigrams)

# transform test data using the trained model
tfidf_test_unigrams_svd  = svd.transform(tfidf_test_unigrams)
```

### Exact — Lab 7, cell 51

```python
print('Size of tfidf_train_unigrams:     ',tfidf_train_unigrams.shape)
print('Size of tfidf_train_unigrams_svd: ',tfidf_train_unigrams_svd.shape)
print()
print('Size of tfidf_test_unigrams:      ',tfidf_test_unigrams.shape)
print('Size of tfidf_test_unigrams_svd:  ',tfidf_test_unigrams_svd.shape)
```

### Exact — Lab 7, cell 53

```python
print('Variance captured by each of the 50 features')
print(svd.explained_variance_ratio_)
print()
print('Total variance captured by the 50 features')
print(svd.explained_variance_ratio_.sum())
print()
print('Singular values of the 50 features')
print(svd.singular_values_)
```

### Exact — Lab 7, cell 57

```python
tfidf_train_unigrams_svd_df = pd.DataFrame(tfidf_train_unigrams_svd)
tfidf_train_unigrams_svd_df.head()
```

### Exact — Lab 7, cell 58

```python
tfidf_test_unigrams_svd_df = pd.DataFrame(tfidf_test_unigrams_svd)
tfidf_test_unigrams_svd_df.head()
```

### Exact — Lab 7, cell 60

```python
svd = TruncatedSVD(n_components=50, random_state=42)

# train model and transform training data
tfidf_train_bigrams_svd = svd.fit_transform(tfidf_train_bigrams)
tfidf_train_bigrams_svd_df = pd.DataFrame(tfidf_train_bigrams_svd)

# transform test data using the trained model
tfidf_test_bigrams_svd  = svd.transform(tfidf_test_bigrams)
tfidf_test_bigrams_svd_df = pd.DataFrame(tfidf_test_bigrams_svd)
```


## 25. Logistic regression using SGDClassifier

**Source:** `LAB7_solutions.ipynb`, cells 65, 67, 68, 69, 70, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85.

**Purpose / explanation:** Compare TF-IDF unigram/bigram and SVD using logistic loss.

**Required imports / context:** sklearn.linear_model.SGDClassifier; sklearn.metrics.classification_report.

**Limitations / project notes:** Not direct sklearn LogisticRegression; group shared factory differs. Retain exact lab parameters as source, not as group settings.

### Exact — Lab 7, cell 65

```python
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report
```

### Exact — Lab 7, cell 67

```python
X_train = tfidf_train_unigrams
X_test  = tfidf_test_unigrams

y_train = df_train['label']
y_test  = df_test['label']
```

### Exact — Lab 7, cell 68

```python
clf = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, random_state=123)

clf.fit(X_train,y_train)
```

### Exact — Lab 7, cell 69

```python
y_pred = clf.predict(X_test)
```

### Exact — Lab 7, cell 70

```python
print(classification_report(y_test, y_pred))
```

### Exact — Lab 7, cell 72

```python
X_train = tfidf_train_bigrams
X_test  = tfidf_test_bigrams

y_train = df_train['label']
y_test  = df_test['label']
```

### Exact — Lab 7, cell 73

```python
clf = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, random_state=123)

clf.fit(X_train,y_train)
```

### Exact — Lab 7, cell 74

```python
y_pred = clf.predict(X_test)
```

### Exact — Lab 7, cell 75

```python
print(classification_report(y_test, y_pred))
```

### Exact — Lab 7, cell 77

```python
X_train = tfidf_train_unigrams_svd_df
X_test  = tfidf_test_unigrams_svd_df

y_train = df_train['label']
y_test  = df_test['label']
```

### Exact — Lab 7, cell 78

```python
clf = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, random_state=123)

clf.fit(X_train,y_train)
```

### Exact — Lab 7, cell 79

```python
y_pred = clf.predict(X_test)
```

### Exact — Lab 7, cell 80

```python
print(classification_report(y_test, y_pred))
```

### Exact — Lab 7, cell 82

```python
X_train = tfidf_train_bigrams_svd_df
X_test  = tfidf_test_bigrams_svd_df

y_train = df_train['label']
y_test  = df_test['label']
```

### Exact — Lab 7, cell 83

```python
clf = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, random_state=123)

clf.fit(X_train,y_train)
```

### Exact — Lab 7, cell 84

```python
y_pred = clf.predict(X_test)
```

### Exact — Lab 7, cell 85

```python
print(classification_report(y_test, y_pred))
```


## 26. Count BoW plus logistic regression directly

**Source:** `LAB9_solutions.ipynb`, cells 33, 34, 35, 36, 38, 40, 42, 45, 46.

**Purpose / explanation:** Train/test count matrix then SGD logistic regression, report.

**Required imports / context:** CountVectorizer; pandas; tensorflow and keras imports in c42; SGDClassifier; metrics.

**Limitations / project notes:** TensorFlow imports are unnecessary for SGD; code exactly preserved. Group LR factory and assignment vectorizer settings override.

### Exact — Lab 9, cell 33

```python
from sklearn.feature_extraction.text import CountVectorizer
```

### Exact — Lab 9, cell 34

```python
# join tokens after preprocessing

clean_text_train = [' '.join(tokens) for tokens in clean_tokens_train]
clean_text_test = [' '.join(tokens) for tokens in clean_tokens_test]
```

### Exact — Lab 9, cell 35

```python
# Define the model
n_grams = CountVectorizer(
        ngram_range  = (1,1),
        max_features = 1000,
        lowercase    = False,
        tokenizer    = None,
        preprocessor = None,
        stop_words   = None,
        min_df       = 10,
        max_df       = 0.80
      )
```

### Exact — Lab 9, cell 36

```python
# Fit the model and transform the training set.
X_train = n_grams.fit_transform(clean_text_train)

X_train = X_train.toarray()
X_train = pd.DataFrame(X_train)
X_train.columns = n_grams.get_feature_names_out()
```

### Exact — Lab 9, cell 38

```python
# Apply the model to the test set.
X_test = n_grams.transform(clean_text_test)

X_test = X_test.toarray()
X_test = pd.DataFrame(X_test)
X_test.columns = n_grams.get_feature_names_out()
```

### Exact — Lab 9, cell 40

```python
# Define y_train, y_test

y_train = df_train['label']
y_test = df_test['label']
```

### Exact — Lab 9, cell 42

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.callbacks import EarlyStopping
```

### Exact — Lab 9, cell 45

```python
clf = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, random_state=123)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred)
```

### Exact — Lab 9, cell 46

```python
print(classification_report(y_test, y_pred))
```


## 27. Keras logistic regression and one-hidden-layer networks (optional course topic)

**Source:** `LAB9_solutions.ipynb`, cells 49, 52, 55, 57, 61, 63, 66, 68, 69.

**Purpose / explanation:** Demonstrate single neuron, hidden layer, epoch change, dropout, early stopping and Adam.

**Required imports / context:** tensorflow.keras.models.Sequential; layers.Dense, Dropout; optimizers.SGD, Adam; callbacks.EarlyStopping; sklearn.metrics.accuracy_score.

**Limitations / project notes:** c66 uses X_test/y_test as validation_data and then evaluates it again: not independent evaluation. For project no neural network needed. Fit/threshold results may be 2D; not group raw text contract. Imports c42.

### Exact — Lab 9, cell 49

```python
# Logistic Regression = single layer, sigmoid activation
model_lr = Sequential([
    Dense(1, input_dim=X_train.shape[1], activation='sigmoid')  # single output neuron
])

model_lr.compile(optimizer=SGD(learning_rate=0.01),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# Train the model
model_lr.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Predict
y_pred_prob = model_lr.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
```

### Exact — Lab 9, cell 52

```python
# Logistic Regression = single layer, sigmoid activation
model_lr = Sequential([
    Dense(1, input_dim=X_train.shape[1], activation='sigmoid')  # single output neuron
])

model_lr.compile(optimizer=SGD(learning_rate=0.01),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# Train the model
model_lr.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# Predict
y_pred_prob = model_lr.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
```

### Exact — Lab 9, cell 55

```python
# Neural Network with one hidden layer
model_nn = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),   # hidden layer
    Dense(1, activation='sigmoid')                              # output layer
])

model_nn.compile(optimizer=SGD(learning_rate=0.01),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# Train the model
model_nn.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Predict
y_pred_prob_nn = model_nn.predict(X_test)
y_pred_nn = (y_pred_prob_nn > 0.5).astype(int)

# Evaluate
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
```

### Exact — Lab 9, cell 57

```python
# Neural Network with one hidden layer
model_nn = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),   # hidden layer
    Dense(1, activation='sigmoid')                              # output layer
])

model_nn.compile(optimizer=SGD(learning_rate=0.01),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# Train the model
model_nn.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# Predict
y_pred_prob_nn = model_nn.predict(X_test)
y_pred_nn = (y_pred_prob_nn > 0.5).astype(int)

# Evaluate
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
```

### Exact — Lab 9, cell 61

```python
# Logistic Regression = single layer, sigmoid activation
model_lr = Sequential([
    Dropout(0.5, input_dim=X_train.shape[1]),
    Dense(1, activation='sigmoid')  # single output neuron
])

model_lr.compile(optimizer=SGD(learning_rate=0.01),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# Train the model
model_lr.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# Predict
y_pred_prob = model_lr.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
```

### Exact — Lab 9, cell 63

```python
# Neural Network with one hidden layer
model_nn = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),    # hidden layer
    Dropout(0.5),
    Dense(1, activation='sigmoid')                               # output layer
])

model_nn.compile(optimizer=SGD(learning_rate=0.01),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# Train the model
model_nn.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# Predict
y_pred_prob_nn = model_nn.predict(X_test)
y_pred_nn = (y_pred_prob_nn > 0.5).astype(int)

# Evaluate
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
```

### Exact — Lab 9, cell 66

```python
model_nn = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(1, activation='sigmoid')
])

model_nn.compile(
    optimizer=SGD(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

model_nn.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Predict
y_pred_prob_nn = model_nn.predict(X_test)
y_pred_nn = (y_pred_prob_nn > 0.5).astype(int)

# Evaluate
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
```

### Exact — Lab 9, cell 68

```python
# Change to Adam optimizer

model_nn = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(1, activation='sigmoid')
])

model_nn.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_nn.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16
)

# Predict
y_pred_prob_nn = model_nn.predict(X_test)
y_pred_nn = (y_pred_prob_nn > 0.5).astype(int)

# Evaluate
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
```

### Exact — Lab 9, cell 69

```python
# Change to Adam optimizer and add a Dropout layer

model_nn = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_nn.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_nn.fit(X_train, y_train, epochs=20,batch_size=16)

# Predict
y_pred_prob_nn = model_nn.predict(X_test)
y_pred_nn = (y_pred_prob_nn > 0.5).astype(int)

# Evaluate
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))
```

## Slide-only and visual code evidence

**Transcription from slides (reconstructed layout, not verbatim notebook code).** The following code is from slide screenshots/text; line wraps/spacing may be normalized. No project code is attributed to the instructor.

### Metrics API — Lab 5 PDF p37 / Lab 6 PDF p18

Purpose: evaluate true versus predicted labels; imports included. Limitation: default binary F1, precision, recall are not macro averages.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

### Dataframe merges — Lab 6 PDF pp3–4

Purpose: join columns by index or id. Requires `import pandas as pd`; the slide snippet assumes it. Index alignment must correspond to same documents.

```python
df1 = pd.DataFrame({'id':[1001, 1002, 1003, 1004],
                    'name':['Anna', 'Mary', 'George', 'Peter']})
df2 = pd.DataFrame({'age':[25,33,18,50]})
df_new = pd.merge(df1, df2, left_index=True, right_index=True)
```

```python
df1 = pd.DataFrame({'id':[1001, 1002, 1003, 1004],
                    'name':['Anna', 'Mary', 'George', 'Peter']})
df2 = pd.DataFrame({'id':[1001, 1002, 1004, 1003],
                    'age':[25,33,18,50]})
df_new = pd.merge(df1, df2, left_on='id', right_on='id')
```

### Contractions alternative — Lab 2 PDF p19

Purpose: expand contractions via library; separate dependency not needed for project.

```python
import contractions
text = contractions.fix(text)
```

### Extra notes PDF p4 — diagram transcription (pseudocode)

The visual splits before preprocessing, then fits vectorizer on training only. The slide contains placeholders; the prediction box appears to call the prediction y_test, whereas evaluation box refers to y_pred. Do not execute as-is. Requires relevant chosen vectorizer/classifier and split variables.

```text
vectorizer = ...
X_train = vectorizer.fit_transform(training_data)
OR
vectorizer = ...
vectorizer.fit(training_data)
X_train = vectorizer.transform(training_data)
X_test = vectorizer.transform(test_data)
clf = ...
clf.fit(X_train, y_train)
y_test = clf.predict(X_test)
```

## Assignment and group adaptations (C/D, NOT lab code)

- P p2 exact baseline configuration: `CountVectorizer(max_features=10000, min_df=5)` and `MultinomialNB(alpha=1.0)` with other defaults and raw input; BoW+LR and TF-IDF+LR preserve the two vectorizer parameters.
- R/H group LR factory: `make_logistic_regression()` from `config`; lab uses SGDClassifier. No complete assignment implementation is supplied here.
- Word2Vec: Lab8 c60 and c70 are closest references; use group seed/workers and suggested min_count=2 if agreed; store trained model and classifier as instance state; use zero vector with same dimension if no known tokens; train only on supplied training texts. These are D adaptations.
- Metrics: `f1_score(y_test, y_pred, average="macro")` is a project adaptation of the taught API (macro metric is C).
- Preserve raw-text `fit(texts, labels)` / `predict(texts)` contract. No changes to starter infrastructure. See COURSE_REFERENCE for detailed rationale, gaps and compatibility review.

**First-pass extraction:** 322 excerpts covering 321 distinct notebook code cells. Second-pass appendix below restores all 180 omitted nonempty code cells; it also includes fenced Markdown snippets and further slide examples. See the coverage table in COURSE_REFERENCE.md.


## Supplementary exact notebook cells — second-pass completeness audit

These cells were omitted from the initial topic excerpts, often because they repeated another lab or only displayed intermediate values. They are restored here to preserve full context. Every nonempty code cell is now represented under an `Exact` heading (Lab 8 c43 appears twice for context). The preceding topical sections remain the recommended entry point. The following is not a combined runnable script; original imports, comments, notebook-specific data and errors remain unchanged.

### Lab 1 — `LAB1_solutions.ipynb`

**Source:** cells 3, 8, 10, 11, 15, 16, 17, 18, 19, 20, 22, 28, 29, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 46, 47, 48, 49, 50, 51, 54, 55, 58, 61, 62, 63. **Purpose / explanation:** Python basics, collections, control flow, functions and math. **Imports / context:** Python built-ins; math or random only where imported in cells. **Limitations:** Includes deliberately failing dictionary access (c29) and educational fragments. Not a project pipeline.

### Exact — Lab 1, cell 3

```python
print('Hello, world!')
```

### Exact — Lab 1, cell 8

```python
# int
age = 25

# float
discount = 0.30

# complex
b = 2+3j

# bool
is_valid = True
is_correct = False

# str
country = 'Cyprus'
capital = "Nicosia"
phrase = '''
Hello World!
'''

# None
x = None
```

### Exact — Lab 1, cell 10

```python
print(age)
```

### Exact — Lab 1, cell 11

```python
age
```

### Exact — Lab 1, cell 15

```python
color = 'purple'
```

### Exact — Lab 1, cell 16

```python
# Zero indexing
print(color[0])
print(color[4])
```

### Exact — Lab 1, cell 17

```python
# Negative indexing
print(color[-1])
print(color[-4])
```

### Exact — Lab 1, cell 18

```python
# Slicing
print(color[0:4])
print(color[-6:-1])
```

### Exact — Lab 1, cell 19

```python
# Slicing with step
print(color[1:5:2])
```

### Exact — Lab 1, cell 20

```python
# Exercise

print(color[::-1])
```

### Exact — Lab 1, cell 22

```python
print('first item:', numbers[0])    # zero indexing
print('last item: ', numbers[-1])   # negative indexing
```

### Exact — Lab 1, cell 28

```python
# dictionaty_name['key']

person['name']
```

### Exact — Lab 1, cell 29

```python
# this cell should raise an error, because the `person` dictionary does not contain a `nationality` key

person['nationality']
```

### Exact — Lab 1, cell 32

```python
number = 10
if number > 0:
    print('positive')
```

### Exact — Lab 1, cell 33

```python
number = -5
if number > 0:
    print('positive')
else:
    print('not positive')
```

### Exact — Lab 1, cell 34

```python
number = 0
if number > 0:
    print('positive')
elif number < 0:
    print('negative')
else:
    print('zero')
```

### Exact — Lab 1, cell 36

```python
for i in range(5):
    print(i)
```

### Exact — Lab 1, cell 37

```python
for x in [5,6,7,8]:
    print(x)
```

### Exact — Lab 1, cell 38

```python
for i in range(3):
    print(i)
else:
    print('Hello!')
```

### Exact — Lab 1, cell 39

```python
for i in ['x','y','z']:
    for j in [1,2]:
        print(i,j)
```

### Exact — Lab 1, cell 40

```python
c = 0
while c<5:
    print(c)
    c+=1
```

### Exact — Lab 1, cell 41

```python
for i in range(5):
    if i==3:
        break
    print(i)
```

### Exact — Lab 1, cell 42

```python
for i in range(5):
    if i==3:
        continue
    print(i)
```

### Exact — Lab 1, cell 46

```python
a = '5'
a
```

### Exact — Lab 1, cell 47

```python
type(a)
```

### Exact — Lab 1, cell 48

```python
b = int(a) # convert to integer
b
```

### Exact — Lab 1, cell 49

```python
type(b)
```

### Exact — Lab 1, cell 50

```python
c = float(b) # convert to float
c
```

### Exact — Lab 1, cell 51

```python
type(c)
```

### Exact — Lab 1, cell 54

```python
def say_hello():
    print('Hello!')
```

### Exact — Lab 1, cell 55

```python
say_hello()
```

### Exact — Lab 1, cell 58

```python
numbers = [i for i in range(5)]
numbers
```

### Exact — Lab 1, cell 61

```python
import math
math.sqrt(25)
```

### Exact — Lab 1, cell 62

```python
from math import sqrt
sqrt(25)
```

### Exact — Lab 1, cell 63

```python
import math as m
m.sqrt(25)
```

**Source:** cells 76, 82, 88, 92, 104, 107. **Purpose / explanation:** Text manipulation, basic exercises and random examples. **Imports / context:** Python built-ins; imports in c104 and c107. **Limitations:** Examples share earlier notebook state; toy values and random draws are not project results.

### Exact — Lab 1, cell 76

```python
num = 5

if num%2==0:
  print('even number')
else:
  print('odd number')
```

### Exact — Lab 1, cell 82

```python
numbers = [1,3,6,8,31,37,50]

for number in numbers:
    if number%2==0:
        print(number,'is an even number')
    else:
        print(number,'is an odd number')
```

### Exact — Lab 1, cell 88

```python
for i in range(1,11):
    print(i, 2*i)
```

### Exact — Lab 1, cell 92

```python
compute_final_price(100,30)
```

### Exact — Lab 1, cell 104

```python
import math

math.pi
```

### Exact — Lab 1, cell 107

```python
import random

random.randint(1,5) # this can be different
```

### Lab 2 — `LAB2_solutions.ipynb`

**Source:** cells 7, 8, 11, 12, 16. **Purpose / explanation:** Regex boundary, matching and extraction cases. **Imports / context:** re (c4). **Limitations:** No-match examples are intentional; example strings are illustrative.

### Exact — Lab 2, cell 7

```python
text = "My password is abc123."
re.findall(r'\d+', text)
```

### Exact — Lab 2, cell 8

```python
text = "This sentence does not contain any number."
re.findall(r'\d+', text)
```

### Exact — Lab 2, cell 11

```python
text = "My password is abc123."
re.match(r"\d+", text)
```

### Exact — Lab 2, cell 12

```python
text = "My password is 00abc123."
re.match(r"\d+", text)
```

### Exact — Lab 2, cell 16

```python
text = "123abc"
re.fullmatch(r"\d+", text)
```

**Source:** cells 48, 49, 50, 56, 61, 73, 80, 87, 101, 103, 104. **Purpose / explanation:** Corpus setup, tokenization and token inspection. **Imports / context:** NLTK, Gutenberg, Counter, string, wordcloud, matplotlib, stopwords and stemmers as imported in c48–115. **Limitations:** Lab corpus is not project training data. Visualization is optional; inspection outputs are not features by themselves.

### Exact — Lab 2, cell 48

```python
import nltk
nltk.download("gutenberg")
from nltk.corpus import gutenberg
```

### Exact — Lab 2, cell 49

```python
text = gutenberg.raw('carroll-alice.txt')[:11517] # get only the first chapter
```

### Exact — Lab 2, cell 50

```python
print(text)
```

### Exact — Lab 2, cell 56

```python
print(text)
```

### Exact — Lab 2, cell 61

```python
print(tokens)
```

### Exact — Lab 2, cell 73

```python
print(clean_tokens)
```

### Exact — Lab 2, cell 80

```python
print('Most common Punctuation:\n')

for t, f in Counter(text_punctuations).most_common(10): print('{0:10} {1}'.format(t, f))
```

### Exact — Lab 2, cell 87

```python
print('Most common Punctuation:\n')

for t, f in Counter(text_punctuations).most_common(10): print('{0:10} {1}'.format(t, f))
```

### Exact — Lab 2, cell 101

```python
print(clean_tokens)
```

### Exact — Lab 2, cell 103

```python
!pip install wordcloud
```

### Exact — Lab 2, cell 104

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

fig, axes = plt.subplots(figsize=(16, 5))

wordcloud = WordCloud().generate(' '.join(clean_tokens))

axes.imshow(wordcloud, interpolation='bilinear')
axes.axis("off")
axes.set_title('Tokens')

plt.show()
```

**Source:** cells 122, 123, 124, 125. **Purpose / explanation:** End-to-end normalization and vocabulary exercise. **Imports / context:** Previously defined uncontract, word_tokenize, stop_words, lemmatizer. **Limitations:** Toy tweets are not project data. Compare counts before/after cleaning; not proof that aggressive cleaning helps classification.

### Exact — Lab 2, cell 122

```python
tweets = [
"Can't believe it's already 2026!!! Time flies 😅",
"Just finished the course — scored 95% on the final 🎉 #proud",
"Email me at test.user@example.com if you’re interested.",
"Wow... this movie was AMAZING!!! 10/10 would watch again.",
"U.S.A. vs Germany tonight at 8:30pm — who’s watching?"
]

tweets = [uncontract(tweet) for tweet in tweets]

tokens = [word_tokenize(tweet) for tweet in tweets]
```

### Exact — Lab 2, cell 123

```python
vocabulary = []
for t in tokens:
    vocabulary += t
vocabulary = set(vocabulary)

print("Vocabulary size after tokenization:", len(vocabulary))
```

### Exact — Lab 2, cell 124

```python
tokens = [[t.lower() for t in token] for token in tokens]
tokens = [[t for t in token if t.isalpha()] for token in tokens]
tokens = [[t for t in token if t not in stop_words] for token in tokens]
tokens = [[lemmatizer.lemmatize(t) for t in token] for token in tokens]
```

### Exact — Lab 2, cell 125

```python
vocabulary = []
for t in tokens:
    vocabulary += t
vocabulary = set(vocabulary)

print("Vocabulary size after all steps:", len(vocabulary))
```

### Lab 3 — `LAB3_solutions.ipynb`

**Source:** cells 5, 6, 7, 8, 21, 22. **Purpose / explanation:** Setup, POS and spaCy label inspection. **Imports / context:** NLTK, spaCy, stanza, TextBlob as applicable; resources in original setup cells. **Limitations:** Needs the lab language resources. Do not add these merely for Word2Vec.

### Exact — Lab 3, cell 5

```python
# !pip install -U pip setuptools==68.0.0 wheel==0.38.4
# !pip install -U spacy==3.6.1 pydantic==1.10.22
```

### Exact — Lab 3, cell 6

```python
!pip install spacy
```

### Exact — Lab 3, cell 7

```python
!pip install textblob
```

### Exact — Lab 3, cell 8

```python
!pip install stanza
```

### Exact — Lab 3, cell 21

```python
############################
# POS tags from token.pos_ #
############################

from spacy.glossary import GLOSSARY

print("Universal POS tags (token.pos_):")

universal_pos = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X", "SPACE"
]

for tag in universal_pos:
    explanation = spacy.explain(tag) or GLOSSARY.get(tag, "No description available")
    print(f"{tag:<7} {explanation}")
```

### Exact — Lab 3, cell 22

```python
############################
# POS tags from token.tag_ #
############################

from spacy.glossary import GLOSSARY

nlp = spacy.load("en_core_web_sm")

print("Fine-grained POS tags (Penn Treebank):")
for tag in nlp.get_pipe("tagger").labels:
    explanation = spacy.explain(tag) or GLOSSARY.get(tag, "No description available")
    print(f"{tag: <8} {explanation}")
```

**Source:** cells 53. **Purpose / explanation:** Linguistic feature exploration on reviews. **Imports / context:** NLTK and spaCy setup c50, earlier review/sentence state. **Limitations:** Lab corpus and sample review displays are not project input.

### Exact — Lab 3, cell 53

```python
for i, review in enumerate(reviews):
    print(f"Review {i+1:2}: {review}")
```

### Lab 4 — `LAB4_solutions.ipynb`

**Source:** cells 4, 6, 8, 9, 13, 23, 24. **Purpose / explanation:** Corpus download, preprocessing and unigram inspection. **Imports / context:** requests, nltk, itertools.chain, math; c6–18 and earlier cells. **Limitations:** External transcript is lab-only; commented shell commands remain comments. Some cells only display values.

### Exact — Lab 4, cell 4

```python
# !wget https://raw.githubusercontent.com/wooters/berp-trans/refs/heads/master/transcript.txt

# !head -2 transcript.txt
```

### Exact — Lab 4, cell 6

```python
import requests

url = "https://raw.githubusercontent.com/wooters/berp-trans/master/transcript.txt"
response = requests.get(url)
response.raise_for_status()  # check for errors

with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(response.text)
```

### Exact — Lab 4, cell 8

```python
brp_sentences = []

with open('transcript.txt', 'r') as f:

    for l in f.readlines():

        s = ' '.join(l.split(' ')[1:])
        brp_sentences.append(s.strip())
```

### Exact — Lab 4, cell 9

```python
brp_sentences
```

### Exact — Lab 4, cell 13

```python
tokenized_texts
```

### Exact — Lab 4, cell 23

```python
for t in model_mle.vocab :
    if t not in flat_tokens :
        print(t)
```

### Exact — Lab 4, cell 24

```python
for t in model_laplace.vocab :
    if t not in flat_tokens :
        print(t)
```

**Source:** cells 40, 45, 46, 55, 56, 57. **Purpose / explanation:** Bigram setup and probability/generation examples. **Imports / context:** NLTK MLE/Laplace, padded_everygram_pipeline, math; c40–54. **Limitations:** Requires tokenized transcript and earlier model variables; not project Word2Vec.

### Exact — Lab 4, cell 40

```python
# Import libraries

import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace
```

### Exact — Lab 4, cell 45

```python
for t in model_mle.vocab :
    if t not in flat_tokens :
        print(t)
```

### Exact — Lab 4, cell 46

```python
for t in model_laplace.vocab :
    if t not in flat_tokens :
        print(t)
```

### Exact — Lab 4, cell 55

```python
# Test with different starting phrases
test_phrases = ["i want", "i do not", "where", "where could i get"]

# preprocessing

test_phrases = [word_tokenize(t) for t in test_phrases]
test_phrases = [[t.lower() for t in tokens] for tokens in test_phrases]
test_phrases = [[lemmatizer.lemmatize(t) for t in tokens] for tokens in test_phrases]
```

### Exact — Lab 4, cell 56

```python
print("=== SENTENCE GENERATION (Most Probable Path) MLE ===")
for phrase in test_phrases:
    sentence = generate_sentence(model_mle, phrase)
    print(f"'{phrase}' → '{sentence}'")
```

### Exact — Lab 4, cell 57

```python
print("=== SENTENCE GENERATION (Most Probable Path) Laplace ===")
for phrase in test_phrases:
    sentence = generate_sentence(model_laplace, phrase)
    print(f"'{phrase}' → '{sentence}'")
```

**Source:** cells 62, 69, 75, 76, 77. **Purpose / explanation:** Four-gram setup and probability/generation examples. **Imports / context:** NLTK MLE/Laplace, padded_everygram_pipeline, math; c62–74. **Limitations:** Requires prior corpus and model state; not a classifier or a project perplexity example.

### Exact — Lab 4, cell 62

```python
# Import libraries

import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace
```

### Exact — Lab 4, cell 69

```python
test_ngrams
```

### Exact — Lab 4, cell 75

```python
# Test with different starting phrases
test_phrases = ["i want", "i do not", "where", "where could i get"]

# preprocessing

test_phrases = [word_tokenize(t) for t in test_phrases]
test_phrases = [[t.lower() for t in tokens] for tokens in test_phrases]
test_phrases = [[lemmatizer.lemmatize(t) for t in tokens] for tokens in test_phrases]
```

### Exact — Lab 4, cell 76

```python
print("=== SENTENCE GENERATION (Most Probable Path) MLE ===")
for phrase in test_phrases:
    sentence = generate_sentence(model_mle, phrase)
    print(f"'{phrase}' → '{sentence}'")
```

### Exact — Lab 4, cell 77

```python
print("=== SENTENCE GENERATION (Most Probable Path) Laplace ===")
for phrase in test_phrases:
    sentence = generate_sentence(model_laplace, phrase)
    print(f"'{phrase}' → '{sentence}'")
```

### Lab 5 — `LAB5_solutions.ipynb`

**Source:** cells 7, 9, 10, 11, 13, 14, 22, 23, 36, 37, 40, 41. **Purpose / explanation:** Dataset inspection, split and preprocessing checks. **Imports / context:** nltk.movie_reviews, pandas, sklearn.model_selection, original preprocessing functions. **Limitations:** Lab-only data and random split; project uses provided split.

### Exact — Lab 5, cell 7

```python
movie_reviews.categories()
```

### Exact — Lab 5, cell 9

```python
### positive reviews

movie_reviews.fileids("pos")[0:10]
```

### Exact — Lab 5, cell 10

```python
# negative reviews

movie_reviews.fileids("neg")[0:10]
```

### Exact — Lab 5, cell 11

```python
print(len(movie_reviews.fileids("pos")), "positive reviews")
print(len(movie_reviews.fileids("neg")), "negative reviews")
```

### Exact — Lab 5, cell 13

```python
# positive

print(movie_reviews.raw(movie_reviews.fileids("pos")[0]))
```

### Exact — Lab 5, cell 14

```python
# negative

print(movie_reviews.raw(movie_reviews.fileids("neg")[0]))
```

### Exact — Lab 5, cell 22

```python
df.head()
```

### Exact — Lab 5, cell 23

```python
df.tail()
```

### Exact — Lab 5, cell 36

```python
df.text[0]
```

### Exact — Lab 5, cell 37

```python
preprocessing_text(df.text[0])
```

### Exact — Lab 5, cell 40

```python
clean_tokens_train[0]
```

### Exact — Lab 5, cell 41

```python
clean_tokens_test[0]
```

**Source:** cells 51, 52, 54, 55, 69. **Purpose / explanation:** BoW inspection and predictions. **Imports / context:** CountVectorizer, pandas, trained clf, prior feature matrices. **Limitations:** Display-only checks are not a separate modeling method.

### Exact — Lab 5, cell 51

```python
bow_train_unigrams.head()
```

### Exact — Lab 5, cell 52

```python
bow_test_unigrams.head()
```

### Exact — Lab 5, cell 54

```python
bow_train_bigrams.head()
```

### Exact — Lab 5, cell 55

```python
bow_test_bigrams.head()
```

### Exact — Lab 5, cell 69

```python
y_pred
```

### Lab 6 — `LAB6_solutions.ipynb`

**Source:** cells 5, 7, 9, 11, 12, 14, 16, 18, 20, 21, 24, 25, 26, 27, 30, 31, 33, 35. **Purpose / explanation:** Full dataset and preprocessing setup. **Imports / context:** nltk.movie_reviews, pandas, train_test_split, NLTK word_tokenize, stopwords, stemmers, re, string. **Limitations:** Training/test split and NLTK downloads are lab-specific; no external corpus or split for core project.

### Exact — Lab 6, cell 5

```python
import nltk
from nltk.corpus import movie_reviews

nltk.download("movie_reviews")
```

### Exact — Lab 6, cell 7

```python
movie_reviews.categories()
```

### Exact — Lab 6, cell 9

```python
print(len(movie_reviews.fileids("pos")), "positive reviews")
print(len(movie_reviews.fileids("neg")), "negative reviews")
```

### Exact — Lab 6, cell 11

```python
# positive

print(movie_reviews.raw(movie_reviews.fileids("pos")[0]))
```

### Exact — Lab 6, cell 12

```python
# negative

print(movie_reviews.raw(movie_reviews.fileids("neg")[0]))
```

### Exact — Lab 6, cell 14

```python
reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]

print("Total number of reviews:", len(reviews))
```

### Exact — Lab 6, cell 16

```python
labels = [1 if fileid.startswith('pos') else 0 for fileid in movie_reviews.fileids()]

print(labels.count(1), "positive reviews")
print(labels.count(0), "negative reviews")
```

### Exact — Lab 6, cell 18

```python
import pandas as pd

df = pd.DataFrame({'text':reviews,'label':labels})
```

### Exact — Lab 6, cell 20

```python
df.head()
```

### Exact — Lab 6, cell 21

```python
df.tail()
```

### Exact — Lab 6, cell 24

```python
from sklearn.model_selection import train_test_split
```

### Exact — Lab 6, cell 25

```python
df_train, df_test = train_test_split(df, test_size=0.20, random_state=123)
```

### Exact — Lab 6, cell 26

```python
print("Number of reviews in training dataset: ", df_train.shape[0])
print("Number of reviews in test dataset:     ", df_test.shape[0])
```

### Exact — Lab 6, cell 27

```python
print("Number of positive reviews in training dataset: ", sum(df_train.label==1))
print("Number of negative reviews in training dataset: ", sum(df_train.label==0))

print("\n")

print("Number of positive reviews in test dataset:     ", sum(df_test.label==1))
print("Number of negative reviews in test dataset:     ", sum(df_test.label==0))
```

### Exact — Lab 6, cell 30

```python
from IPython.display import clear_output

clear_output()

import nltk

nltk.download("gutenberg")
from nltk.corpus import gutenberg

from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import PorterStemmer

nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

import re
import string
```

### Exact — Lab 6, cell 31

```python
def uncontract(text):
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)

    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ii]t)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Tt]here)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)

    return text


def tokenize_text(text):
    tokens = word_tokenize(text)

    return tokens


def lowercase_tokens(tokens):
    tokens = [t.lower() for t in tokens]

    return tokens


def remove_punctuation_digits(tokens):
    tokens = [t for t in tokens if t.isalpha()]

    return tokens


def remove_stopwords(tokens):
    stop_words = stopwords.words('english')

    tokens = [t for t in tokens if t not in stop_words]

    return tokens


def perform_stemming(tokens):
    ps = PorterStemmer()

    stemmed = [ps.stem(t) for t in tokens]

    return stemmed


def perform_lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()

    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]

    return lemmatized
```

### Exact — Lab 6, cell 33

```python
def preprocessing_text(text):
    text = uncontract(text)
    text = tokenize_text(text)
    text = lowercase_tokens(text)
    text = remove_punctuation_digits(text)
    text = remove_stopwords(text)
    text = perform_lemmatization(text)

    return text
```

### Exact — Lab 6, cell 35

```python
clean_tokens_train = [preprocessing_text(t) for t in df_train['text']]
clean_tokens_test  = [preprocessing_text(t) for t in df_test['text']]
```

**Source:** cells 40, 41, 42, 43, 44, 45, 46. **Purpose / explanation:** CountVectorizer setup and train/test feature extraction. **Imports / context:** CountVectorizer (c40), pandas, clean_tokens_train/test. **Limitations:** These were absent from first excerpt collection. Lab vectorizer parameters differ from assignment baseline.

### Exact — Lab 6, cell 40

```python
from sklearn.feature_extraction.text import CountVectorizer
```

### Exact — Lab 6, cell 41

```python
# join tokens after preprocessing

clean_text_train = [' '.join(tokens) for tokens in clean_tokens_train]
clean_text_test  = [' '.join(tokens) for tokens in clean_tokens_test]
```

### Exact — Lab 6, cell 42

```python
# Define the model
n_grams = CountVectorizer(
        ngram_range  = (1,1),
        max_features = 1000,
        lowercase    = False,
        tokenizer    = None,
        preprocessor = None,
        stop_words   = None,
        min_df       = 10,
        max_df       = 0.80
      )
```

### Exact — Lab 6, cell 43

```python
# Fit the model and transform the training set.
bow_train = n_grams.fit_transform(clean_text_train)

bow_train = bow_train.toarray()
bow_train = pd.DataFrame(bow_train)
bow_train.columns = n_grams.get_feature_names_out()
```

### Exact — Lab 6, cell 44

```python
bow_train.head()
```

### Exact — Lab 6, cell 45

```python
# Apply the model to the test set.
bow_test = n_grams.transform(clean_text_test)

bow_test = bow_test.toarray()
bow_test = pd.DataFrame(bow_test)
bow_test.columns = n_grams.get_feature_names_out()
```

### Exact — Lab 6, cell 46

```python
bow_test.head()
```

**Source:** cells 51, 52, 60, 61, 64, 65, 69, 71, 76, 79. **Purpose / explanation:** Feature inspection. **Imports / context:** opinion_lexicon, earlier DataFrames, raw tokens and math.log. **Limitations:** Retain alignment; nonempty-token assumption and external-resource restrictions apply.

### Exact — Lab 6, cell 51

```python
positive_words
```

### Exact — Lab 6, cell 52

```python
negative_words
```

### Exact — Lab 6, cell 60

```python
lexical_features_train.head()
```

### Exact — Lab 6, cell 61

```python
lexical_features_test.head()
```

### Exact — Lab 6, cell 64

```python
lexical_features_train.head()
```

### Exact — Lab 6, cell 65

```python
lexical_features_test.head()
```

### Exact — Lab 6, cell 69

```python
print(tokens_train[0])
print(tokens_test[0])
```

### Exact — Lab 6, cell 71

```python
existence_no_train[0:10]
```

### Exact — Lab 6, cell 76

```python
existence_exclamation_train[0:10]
```

### Exact — Lab 6, cell 79

```python
log_tokens_train[0:10]
```

**Source:** cells 104. **Purpose / explanation:** Feature ranking and classifier inspection. **Imports / context:** RandomForestClassifier, get_top_features (c103), existing X_train/y_train. **Limitations:** Feature selection must be re-fit per training fold for independent CV.

### Exact — Lab 6, cell 104

```python
get_top_features(X_train, y_train, 10)
```

### Lab 7 — `LAB7_solutions.ipynb`

**Source:** cells 5, 7, 9, 11, 13, 15, 17, 18, 21, 22, 23, 24, 27, 28, 29, 31. **Purpose / explanation:** Full dataset, split and preprocessing setup. **Imports / context:** nltk.movie_reviews, pandas, train_test_split, NLTK tokenization/stemmers/stopwords, re, string. **Limitations:** Lab-only corpus and split; code assumes NLTK resources.

### Exact — Lab 7, cell 5

```python
import nltk
from nltk.corpus import movie_reviews

nltk.download("movie_reviews")
```

### Exact — Lab 7, cell 7

```python
movie_reviews.categories()
```

### Exact — Lab 7, cell 9

```python
print(len(movie_reviews.fileids("pos")), "positive reviews")
print(len(movie_reviews.fileids("neg")), "negative reviews")
```

### Exact — Lab 7, cell 11

```python
reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]

print("Total number of reviews:", len(reviews))
```

### Exact — Lab 7, cell 13

```python
labels = [1 if fileid.startswith('pos') else 0 for fileid in movie_reviews.fileids()]

print(labels.count(1), "positive reviews")
print(labels.count(0), "negative reviews")
```

### Exact — Lab 7, cell 15

```python
import pandas as pd

df = pd.DataFrame({'text':reviews,'label':labels})
```

### Exact — Lab 7, cell 17

```python
df.head()
```

### Exact — Lab 7, cell 18

```python
df.tail()
```

### Exact — Lab 7, cell 21

```python
from sklearn.model_selection import train_test_split
```

### Exact — Lab 7, cell 22

```python
df_train, df_test = train_test_split(df, test_size=0.20, random_state=123)
```

### Exact — Lab 7, cell 23

```python
print("Number of reviews in training dataset: ", df_train.shape[0])
print("Number of reviews in test dataset:     ", df_test.shape[0])
```

### Exact — Lab 7, cell 24

```python
print("Number of positive reviews in training dataset: ", sum(df_train.label==1))
print("Number of negative reviews in training dataset: ", sum(df_train.label==0))

print("\n")

print("Number of positive reviews in test dataset:     ", sum(df_test.label==1))
print("Number of negative reviews in test dataset:     ", sum(df_test.label==0))
```

### Exact — Lab 7, cell 27

```python
from IPython.display import clear_output

clear_output()

import nltk

nltk.download("gutenberg")
from nltk.corpus import gutenberg

from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import PorterStemmer

nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

import re
import string
```

### Exact — Lab 7, cell 28

```python
def uncontract(text):
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)

    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ii]t)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Tt]here)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)

    return text


def tokenize_text(text):
    tokens = word_tokenize(text)

    return tokens


def lowercase_tokens(tokens):
    tokens = [t.lower() for t in tokens]

    return tokens


def remove_punctuation_digits(tokens):
    tokens = [t for t in tokens if t.isalpha()]

    return tokens


def remove_stopwords(tokens):
    stop_words = stopwords.words('english')

    tokens = [t for t in tokens if t not in stop_words]

    return tokens


def perform_stemming(tokens):
    ps = PorterStemmer()

    stemmed = [ps.stem(t) for t in tokens]

    return stemmed


def perform_lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()

    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]

    return lemmatized
```

### Exact — Lab 7, cell 29

```python
def preprocessing_text(text):
    text = uncontract(text)
    text = tokenize_text(text)
    text = lowercase_tokens(text)
    text = remove_punctuation_digits(text)
    text = remove_stopwords(text)
    text = perform_lemmatization(text)

    return text
```

### Exact — Lab 7, cell 31

```python
clean_tokens_train = [preprocessing_text(t) for t in df_train['text']]
clean_tokens_test  = [preprocessing_text(t) for t in df_test['text']]
```

**Source:** cells 41, 42, 43, 44. **Purpose / explanation:** TF-IDF feature inspections. **Imports / context:** TfidfVectorizer, pandas, earlier fitted matrices. **Limitations:** No fitting on validation; inspections are display-only.

### Exact — Lab 7, cell 41

```python
tfidf_train_unigrams.head()
```

### Exact — Lab 7, cell 42

```python
tfidf_test_unigrams.head()
```

### Exact — Lab 7, cell 43

```python
tfidf_train_bigrams.head()
```

### Exact — Lab 7, cell 44

```python
tfidf_test_bigrams.head()
```

**Source:** cells 55, 61, 62. **Purpose / explanation:** SVD output inspection. **Imports / context:** TruncatedSVD, pandas, previously fitted SVD matrices. **Limitations:** SVD is optional, not the required TF-IDF LR model.

### Exact — Lab 7, cell 55

```python
tfidf_train_unigrams_svd
```

### Exact — Lab 7, cell 61

```python
tfidf_train_bigrams_svd_df.head()
```

### Exact — Lab 7, cell 62

```python
tfidf_test_bigrams_svd_df.head()
```

### Lab 8 — `LAB8_solutions.ipynb`

**Source:** cells 45, 56. **Purpose / explanation:** Word2Vec display-only cells. **Imports / context:** Pretrained wv_model and prior vectors/train_vectors; numpy for prior mean pooling. **Limitations:** Displays only. Pretrained path is lab evidence, not a permitted mandatory-project representation.

### Exact — Lab 8, cell 45

```python
vectors
```

### Exact — Lab 8, cell 56

```python
train_vectors[:2]
```

### Lab 9 — `LAB9_solutions.ipynb`

**Source:** cells 5, 7, 9, 11, 13, 15, 18, 19, 20, 21, 24, 25, 26, 28. **Purpose / explanation:** Full data and preprocessing setup. **Imports / context:** nltk.movie_reviews, pandas, train_test_split, NLTK resources, re, string. **Limitations:** External lab corpus and split must not be copied into core project.

### Exact — Lab 9, cell 5

```python
import nltk
from nltk.corpus import movie_reviews

nltk.download("movie_reviews")
```

### Exact — Lab 9, cell 7

```python
movie_reviews.categories()
```

### Exact — Lab 9, cell 9

```python
print(len(movie_reviews.fileids("pos")), "positive reviews")
print(len(movie_reviews.fileids("neg")), "negative reviews")
```

### Exact — Lab 9, cell 11

```python
reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]

print("Total number of reviews:", len(reviews))
```

### Exact — Lab 9, cell 13

```python
labels = [1 if fileid.startswith('pos') else 0 for fileid in movie_reviews.fileids()]

print(labels.count(1), "positive reviews")
print(labels.count(0), "negative reviews")
```

### Exact — Lab 9, cell 15

```python
import pandas as pd

df = pd.DataFrame({'text':reviews,'label':labels})
```

### Exact — Lab 9, cell 18

```python
from sklearn.model_selection import train_test_split
```

### Exact — Lab 9, cell 19

```python
df_train, df_test = train_test_split(df, test_size=0.20, random_state=123)
```

### Exact — Lab 9, cell 20

```python
print("Number of reviews in training dataset: ", df_train.shape[0])
print("Number of reviews in test dataset:     ", df_test.shape[0])
```

### Exact — Lab 9, cell 21

```python
print("Number of positive reviews in training dataset: ", sum(df_train.label==1))
print("Number of negative reviews in training dataset: ", sum(df_train.label==0))

print("\n")

print("Number of positive reviews in test dataset:     ", sum(df_test.label==1))
print("Number of negative reviews in test dataset:     ", sum(df_test.label==0))
```

### Exact — Lab 9, cell 24

```python
from IPython.display import clear_output

clear_output()

import nltk

nltk.download("gutenberg")
from nltk.corpus import gutenberg

from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import PorterStemmer

nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

import re
import string
```

### Exact — Lab 9, cell 25

```python
def uncontract(text):
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)

    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ii]t)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Tt]here)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)

    return text


def tokenize_text(text):
    tokens = word_tokenize(text)

    return tokens


def lowercase_tokens(tokens):
    tokens = [t.lower() for t in tokens]

    return tokens


def remove_punctuation_digits(tokens):
    tokens = [t for t in tokens if t.isalpha()]

    return tokens


def remove_stopwords(tokens):
    stop_words = stopwords.words('english')

    tokens = [t for t in tokens if t not in stop_words]

    return tokens


def perform_stemming(tokens):
    ps = PorterStemmer()

    stemmed = [ps.stem(t) for t in tokens]

    return stemmed


def perform_lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()

    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]

    return lemmatized
```

### Exact — Lab 9, cell 26

```python
def preprocessing_text(text):
    text = uncontract(text)
    text = tokenize_text(text)
    text = lowercase_tokens(text)
    text = remove_punctuation_digits(text)
    text = remove_stopwords(text)
    text = perform_lemmatization(text)

    return text
```

### Exact — Lab 9, cell 28

```python
clean_tokens_train = [preprocessing_text(t) for t in df_train['text']]
clean_tokens_test  = [preprocessing_text(t) for t in df_test['text']]
```

**Source:** cells 37, 39. **Purpose / explanation:** BoW matrix inspection. **Imports / context:** CountVectorizer, pandas, X_train/X_test. **Limitations:** Display-only checks; no neural model required by assignment.

### Exact — Lab 9, cell 37

```python
X_train.head()
```

### Exact — Lab 9, cell 39

```python
X_test.head()
```

## Additional exact fenced snippets from notebook Markdown cells
These are instructor-provided illustrative snippets or exercises, not executed code cells. Cell numbers are physical indexes. Preserve placeholders and incomplete syntax as printed. Imports are included where given; snippets may depend on variables introduced elsewhere in the same notebook.

### Exact Markdown snippet — Lab 1, cell 53, block 1

```python
def function_name(arguments):
	# code block
```

### Exact Markdown snippet — Lab 1, cell 53, block 2

```python
function_name()
```

### Exact Markdown snippet — Lab 2, cell 42, block 1

```text
text = """Alice went to Paris.
Bob went to Rome.
Alice went to Berlin."""
```

### Exact Markdown snippet — Lab 2, cell 121, block 1

```python
tweets = [
"Can't believe it's already 2026!!! Time flies 😅",
"Just finished the course — scored 95% on the final 🎉 #proud",
"Email me at test.user@example.com if you’re interested.",
"Wow... this movie was AMAZING!!! 10/10 would watch again.",
"U.S.A. vs Germany tonight at 8:30pm — who’s watching?"
]
```

### Exact Markdown snippet — Lab 5, cell 17, block 1

```python
for fileid in movie_reviews.fileids()
```

### Exact Markdown snippet — Lab 5, cell 45, block 1

```python
from sklearn.feature_extraction.text import CountVectorizer

n_grams = CountVectorizer(
        ngram_range  = (1,1),
        max_features = 1000,
        lowercase    = False,
        tokenizer    = None,
        preprocessor = None,
        stop_words   = None,
        min_df       = 10,
        max_df       = 0.80
    )
```

### Exact Markdown snippet — Lab 5, cell 45, block 2

```python
# train model, and also create the features for the training set
bow_train = n_grams.fit_transform(training_data)

# process the features for the training set, so that they can be in a dataframe format
bow_train = bow_train.toarray()
bow_train = pd.DataFrame(bow_train)
bow_train.columns = n_grams.get_feature_names_out()
```

### Exact Markdown snippet — Lab 5, cell 45, block 3

```python
# transform the test set to features, using the pre-trained model n_grams
bow_test = n_grams.transform(test_data)

bow_test = bow_test.toarray()
bow_test = pd.DataFrame(bow_test)
bow_test.columns = n_grams.get_feature_names_out()
```

### Exact Markdown snippet — Lab 5, cell 45, block 4

```python
' '.join(tokens)
```

### Exact Markdown snippet — Lab 5, cell 61, block 1

```python
from sklearn.naive_bayes import MultinomialNB

# create and run classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# make predictions on test set
y_pred = clf.predict(X_test)
```

### Exact Markdown snippet — Lab 5, cell 70, block 1

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
```

### Exact Markdown snippet — Lab 5, cell 79, block 1

```python
from sklearn.model_selection import cross_val_score

# initialize the classifier
clf = MultinomialNB()

# perform 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores - accuracy for 5-folds:", scores)
print("Average accuracy:", scores.mean())
```

### Exact Markdown snippet — Lab 6, cell 89, block 1

```python
from sklearn.naive_bayes import MultinomialNB

# create and run classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# make predictions on test set
y_pred = clf.predict(X_test)
```

### Exact Markdown snippet — Lab 7, cell 34, block 1

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    ngram_range  = (1, 1),
    max_features = 1000,
    lowercase    = False,
    tokenizer    = None,
    preprocessor = None,
    stop_words   = None,
    min_df       = 10,
    max_df       = 0.8
)
```

### Exact Markdown snippet — Lab 7, cell 34, block 2

```python
# train model, and also create the features for the training set
tfidf_train = tfidf.fit_transform(training_data)

# process the features for the training set, so that they can be in a dataframe format
tfidf_train = tfidf_train.toarray()
tfidf_train = pd.DataFrame(tfidf_train)
tfidf_train.columns = tfidf.get_feature_names_out()
```

### Exact Markdown snippet — Lab 7, cell 34, block 3

```python
# transform the test set to features, using the pre-trained model ngrams
tfidf_test = tfidf.transform(test_data)

tfidf_test = tfidf_test.toarray()
tfidf_test = pd.DataFrame(tfidf_test)
tfidf_test.columns = tfidf.get_feature_names_out()
```

### Exact Markdown snippet — Lab 7, cell 34, block 4

```python
' '.join(tokens)
```

### Exact Markdown snippet — Lab 7, cell 64, block 1

```python
from sklearn.linear_model import SGDClassifier

# create and run classifier
clf = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, random_state=123)
clf.fit(X_train, y_train)

# make predictions on test set
y_pred = clf.predict(X_test)
```

### Exact Markdown snippet — Lab 8, cell 80, block 1

```python
from sklearn.ensemble import RandomForestClassifier

# create and run classifier
clf = RandomForestClassifier(n_estimators=100, random_state=123)
clf.fit(X_train, y_train)

# make predictions on test set
y_pred = clf.predict(X_test)
```

### Exact Markdown snippet — Lab 9, cell 44, block 1

```python
from sklearn.linear_model import SGDClassifier

# create and run classifier
clf = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, random_state=123)
clf.fit(X_train, y_train)

# make predictions on test set
y_pred = clf.predict(X_test)
```

## Additional slide transcriptions (second pass; reconstructed line wraps)

### Lab 1 PDF p12 — library tokenization
Purpose: demonstrate Python simplicity and NLTK tokenization. Imports shown; tokenizer data must be available. Not Word2Vec-specific.

```python
# Tokenizing a sentence in Python
from nltk.tokenize import word_tokenize
print(word_tokenize("Hello world, how are you?"))
```

### Lab 2 PDF p11 — case-insensitive regex
Purpose: compare case-sensitive and insensitive matches. Requires `import re` (p9). This was slide code, not merely theory. The other flags `MULTILINE`, `DOTALL` are listed on that slide; `MULTILINE` also has notebook code (c45).

```python
text = "Dogs are great pets. I have a dog as my pet."
re.findall(r'dog', text)
re.findall(r'dog', text, flags=re.IGNORECASE)
```

### Lab 5 PDF p13 — CSV IO
Purpose: load/save a dataframe; requires pandas, original filename is illustrative. The project uses its existing loader/evaluator instead. The wrapped comment has been joined.

```python
import pandas as pd
# load dataframe from a .csv file
df = pd.read_csv("dataframe.csv")
# save dataframe to a .csv file - set index=False when we don’t want to save the index
df.to_csv("dataframe.csv", index=False)
```

### Lab 5 PDF p14 — table of DataFrame operations
Purpose: data handling examples, not a single runnable sequence. Requires pandas, defined frames and variables; names are placeholders. These are individual table entries.

```python
df.copy()
pd.concat([df1, df2])
pd.merge(df1, df2, on='column')
df.join(other_df)
df['new'] = df['col1'] + df['col2']
df['col'].astype('int')
df['col'].apply(func)
df['col'].map({'A':1, 'B':0})
df['col'].replace(old, new)
```

### Lab 6 PDF p5 — two ways to construct DataFrames
Purpose: assemble columns or rows (useful for feature dictionaries); requires `import pandas as pd`. Toy data only.

```python
# --- Method 1: Using a dictionary of columns ---
# key = column name, value = list of all values in that column
df = pd.DataFrame({
    'id': [1001, 1002, 1003, 1004],
    'name': ['Anna', 'Mary', 'George', 'Peter']})

# --- Method 2: Using a list of dictionaries ---
# dictionary = row, key-value pairs = column names and values.
df_data = [
    {'id': 1001, 'name': 'Anna'},
    {'id': 1002, 'name': 'Mary'},
    {'id': 1003, 'name': 'George'},
    {'id': 1004, 'name': 'Peter'}
]
df = pd.DataFrame(df_data)
```

### Lab 7 PDF p17 — SVD train/test transform
Purpose: project into 50 dimensions, inspect explained variance. Requires pandas, original TF-IDF variables; not needed for mandatory project. Same fitting direction as the notebook.

```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=42)
# train model and transform training data
tfidf_train_unigrams_svd = svd.fit_transform(tfidf_train_unigrams)
tfidf_train_unigrams_svd_df = pd.DataFrame(tfidf_train_unigrams_svd)
# transform test data using the trained model
tfidf_test_unigrams_svd  = svd.transform(tfidf_test_unigrams)
tfidf_test_unigrams_svd_df = pd.DataFrame(tfidf_test_unigrams_svd)
# variance captured by each of the features
print(svd.explained_variance_ratio_)
# total variance captured by the features
print(svd.explained_variance_ratio_.sum())
```

### Lab 9 PDF p7 — logistic regression in Keras
Purpose: demonstrate sigmoid single-neuron model, not the shared sklearn LR implementation. Imports are from Lab 9 notebook c42 (Sequential, Dense, SGD, accuracy_score). Requires defined feature arrays and labels.

```python
model_lr = Sequential([
    Dense(1, input_dim=X_train.shape[1], activation='sigmoid')
])
model_lr.compile(optimizer=SGD(learning_rate=0.01),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
model_lr.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)
y_pred_prob = model_lr.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
accuracy_score(y_test, y_pred)
```

**Important slide caution:** the threshold diagram on Lab 9 p6 uses a threshold at 0.5; the code on p7/11 uses `> 0.5`, so exactly 0.5 is class 0 in the actual code. Follow the actual code rather than silently changing it. The project LR factory handles its own decisions.

**Second-pass validation:** 502 exact code-cell blocks covering all 501 unique nonempty original notebook code cells (L8 c43 intentionally repeated). All block bodies matched their original cell sources excluding trailing newlines. All 20 extracted fenced Markdown snippets matched their original Markdown cells. The slide transcriptions are selected examples, not an exhaustive transcription of every slide image. See COURSE_REFERENCE.md §12 for coverage and limitations.

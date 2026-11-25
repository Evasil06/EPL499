# Member 2 Word2Vec error analysis

These seven examples are actual errors from the original 100-dimensional Skip-gram + Logistic Regression runs. They span both error directions in all three challenges and include documents with a wide range of vocabulary coverage. Coverage counts token occurrences after the original tokenizer, including repeats, and checks them against the fitted vocabulary for that challenge. The excerpts are from the labeled validation files.

## Challenge A

### `epl499_89a300ef22887e17dabe` · 0 → 1

The fully human-written review was predicted to be AI-polished. All 59 tokens were in vocabulary.

> All 3 seem unconvinced of the contributions, point to several shortcomings, and recommend rejection.

**Observation** The text gives a formal recommendation and refers to other reviewers in short, direct sentences. The tokenizer removes the digit in “All 3.”

**Hypothesis** Review vocabulary such as “contributions,” “shortcomings,” and “recommend” can occur in both original and polished reviews. A mean over those words can give more weight to the review genre than to the author’s choice to repeat the numerical reference.

### `epl499_871534d5829f60760593` · 1 → 0

The AI-polished review was predicted to be fully human-written. All 47 tokens were in vocabulary.

> The paper proposes a new memory module to complement existing neural network models.

**Observation** The review opens with a technical summary, then evaluates the study in separate Pros and Cons lists. Its vocabulary is drawn from the same peer-review genre as the preceding case.

**Hypothesis** The model retains the words “pros” and “cons,” but the mean cannot represent where the review switches between its summary, strengths, and criticism. That leaves shared review vocabulary as a potential source of confusion even with full coverage.

## Challenge B

### `epl499_5503408625799eeeab08` · 0 → 1

The fully human-written article was predicted to be AI-written and machine-humanized. The vocabulary covered 936 of 969 tokens.

> It is the second game in The Legend of Spyro trilogy, released for the Game Boy Advance, Nintendo DS, PlayStation 2 and Wii.

**Observation** The entry resembles a reference article. It names platforms, games, and characters, with individual sections covering gameplay, plot, and reception.

**Hypothesis** Recurring franchise and gameplay words contribute repeatedly to the document mean. The resulting topical signal could resemble a machine-written game summary. Section boundaries and the digit in “PlayStation 2” are unavailable as explicit cues.

### `epl499_05948c0159bfeb27ae29` · 1 → 0

The AI-written, machine-humanized advice was predicted to be fully human-written. The vocabulary covered 182 of 291 tokens, leaving 109 out of vocabulary (37.46%).

> Give yerself time to prepere. You're about to have a harde convewation that coud ewasie leece towne into a fowht if you're not cawefuw.

**Observation** The advice uses phonetic spellings and altered words throughout. The excerpt still follows the structure of an ordinary advice paragraph.

**Hypothesis** Averaging drops every out-of-vocabulary occurrence. That leaves conventional advice vocabulary with a larger share of the mean and removes many surface forms that could signal the document’s style. High OOV here identifies information loss. The role of the discarded tokens in the prediction would require a separate intervention to test.

### `epl499_2026f8cc21795c1630c6` · 1 → 0

The AI-written, machine-humanized advice was predicted to be fully human-written. All 211 tokens were in vocabulary.

> Do you sell things to people from your office, or online, by phone or mail?

**Observation** The opening uses a series of direct questions and contractions. The rest explains practical obligations in a conversational second-person register.

**Hypothesis** The mean keeps common words from the questions, while punctuation removal and pooling discard their interrogative punctuation and sequence. The remaining mix of second-person and consumer-advice vocabulary may resemble a human-written explanation. Full coverage makes vocabulary loss an unlikely explanation for this example.

## Challenge C

### `epl499_6efc84ffab3bbc8b1142` · 0 → 1

The AI-polished human essay was predicted to be AI-written and machine-humanized. The vocabulary covered 319 of 320 tokens.

> Some years ago, before my uncle became a doctor, I saw him struggling with anxiety attacks due to the intense studying at medical school.

**Observation** The document begins with “Here is the revised essay” and develops a first-person family anecdote into general advice about extracurricular activities.

**Hypothesis** The representation loses the position of the introductory editing note and the transition from anecdote to advice. It can encode words from both parts, but the same blend of personal and advisory vocabulary could occur under either of C’s two AI-processed labels.

### `epl499_6e7d0719ed1ea063c8ad` · 1 → 0

The AI-written, machine-humanized narrative was predicted to be AI-polished human writing. All 327 tokens were in vocabulary.

> As a Seagoing Cowboy, I was granted a unique opportunity to traverse the world and contribute to the betterment of others' lives.

**Observation** The piece mixes historical details about a livestock relief program with a first-person account, reflective language, and a closing recommendation.

**Hypothesis** Familiar personal-narrative vocabulary and specific historical vocabulary can also occur in a polished human account. The mean contains those words while losing the sequence that connects claimed experiences to the program details. This makes their co-occurrence a weak clue to original authorship.

## Patterns across the original validation runs

Token-level OOV rates were higher for errors than correct predictions in A’s fully human-written class (4.55% versus 2.48%) and B’s machine-humanized class (2.06% versus 1.08%). In C’s AI-polished class, the reverse held (1.26% versus 2.32%). These rates divide total unknown token occurrences by total retained alphabetic token occurrences in each group. Every validation document had at least one known token, so the zero-vector fallback was unused in these runs. The opposite patterns and the fully covered errors limit how much overall error can be attributed to vocabulary coverage alone.

B and C have the same 1:2 label ratio. In C, 378 of the 1,125 AI-polished human documents were predicted to be machine-humanized, compared with 140 of the 1,125 fully human-written documents in B. The observed recall gaps are compatible with majority-class preference. The larger gap in C also coincides with a change in what class 0 means and with both classes having undergone AI processing. The comparison does not isolate a causal effect of imbalance.

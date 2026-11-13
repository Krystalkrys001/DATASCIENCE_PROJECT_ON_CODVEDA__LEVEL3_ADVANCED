# Level 3 — Task 2: Natural Language Processing (NLP) — SMS Spam Classification

## Project Overview
An end-to-end NLP classification pipeline built on `spam.csv` (5,572 raw SMS messages) to separate legitimate messages (`ham`) from promotional/phishing messages (`spam`) using NLTK linguistic preprocessing, leakage-free TF-IDF vectorization, and multi-classifier benchmarking.

## Pipeline & Leakage Safeguards
1. **Schema Cleaning & Deduplication:** Removed three empty CSV artifact columns (`Unnamed: 2–4`) and dropped **403 exact duplicate messages**, leaving **5,169 unique SMS records** (`87.37%` ham, `12.63%` spam) so identical texts never leak across train/test splits.
2. **NLTK Linguistic Normalization:** Lowercased text, stripped punctuation/digits via regex, tokenized with `word_tokenize`, filtered English stopwords, and lemmatized tokens via `WordNetLemmatizer`.
3. **Stratified Split & TF-IDF:** Executed an 80/20 stratified train/test split (`4,135` train / `1,034` test) *prior* to fitting `TfidfVectorizer(max_features=3000, ngram_range=(1, 2))`.

## Test Set Benchmark Results (N = 1,034)
| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 0.9700 | **0.9717** | 0.7863 | 0.8692 |
| **Logistic Regression** | 0.9613 | 0.9596 | 0.7252 | 0.8261 |
| **Linear SVM (`LinearSVC`)** | **0.9768** | 0.9496 | **0.8626** | **0.9040** |

* **Linear SVM Confusion Matrix:** `897` True Negatives, `6` False Positives (<0.7% false alarm rate), `18` False Negatives, `113` True Positives.
* **Top Spam TF-IDF Indicators:** `www`, `mobile`, `txt`, `claim`, `service`, `prize`, `http`.

## Tech Stack
`Python 3.12` • `Pandas` • `NumPy` • `NLTK` • `Scikit-learn` • `Matplotlib` • `Seaborn`
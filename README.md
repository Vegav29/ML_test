# Natural Language Processing (NLP) Projects

This repository showcases various NLP-based projects demonstrating techniques such as Bag-of-Words (BoW), TF-IDF, embeddings (GloVe and Word2Vec), sentiment analysis, summarization, and evaluation metrics (ROUGE, BLEU).

---

## **1. Bag-of-Words (BoW) and TF-IDF Representation**

### **Objective:**
- Represent textual data as numerical vectors using Bag-of-Words and TF-IDF techniques.

### **Implementation:**
- BoW: Uses `CountVectorizer` to convert text into word frequency counts.
- TF-IDF: Uses `TfidfVectorizer` to compute term importance based on frequency and inverse document frequency.

### **Steps:**
1. Tokenize and vectorize the corpus using BoW.
2. Calculate term weights using TF-IDF.
3. Display the resulting numerical representations and feature names.

### **Code Snippet:**
```python
bow_array, bow_features = bow_representation(corpus)
tfidf_array, tfidf_features = tfidf_representation(corpus)
```

---

## **2. Embeddings: GloVe and Word2Vec**

### **Objective:**
- Represent words as dense vectors using pre-trained embeddings (GloVe) and Word2Vec.

### **Implementation:**
- **GloVe:** Load pre-trained embeddings using `gensim` (glove-wiki-gigaword-100).
- **Word2Vec:** Train embeddings using Word2Vec on a custom corpus.

### **Steps:**
1. Load pre-trained GloVe embeddings.
2. Train Word2Vec embeddings on the input corpus.
3. Retrieve word vectors for comparison.

### **Code Snippet:**
```python
glove_model = load_glove_embeddings()
word2vec_model = train_word2vec(corpus)
print(glove_model['product'])  # GloVe vector
print(word2vec_model.wv['product'])  # Word2Vec vector
```

---

## **3. Sentiment Analysis on Custom Dataset**

### **Objective:**
- Perform sentiment analysis on dialogues from the **Her-The-Movie** dataset.

### **Tools & Libraries:**
- Hugging Face `pipeline` for sentiment analysis.
- Dataset: `HaltiaAI/Her-The-Movie-Samantha-and-Theodore-Dataset`
- Output: Annotated dataset with sentiment labels (Positive/Negative).

### **Steps:**
1. Load the dataset and preprocess column names.
2. Analyze sentiment for both `Prompt` and `Response` columns.
3. Save the results as a CSV file.
4. Display sentiment distribution.

### **Code Snippet:**
```python
df['Prompt_Sentiment'] = analyze_sentiments(df['Prompt'].tolist())
df['Response_Sentiment'] = analyze_sentiments(df['Response'].tolist())
df.to_csv("sentiment_analysis_results_separate.csv", index=False)
```

---

## **4. Text Summarization and Evaluation**

### **Objective:**
- Generate summaries for articles using Transformer-based models and evaluate them using ROUGE and BLEU metrics.

### **Dataset:**
- `xsum` dataset (test subset).

### **Tools & Libraries:**
- **Model:** `facebook/bart-large-cnn` for summarization.
- **Evaluation:**
  - ROUGE scores (`rouge_score` library)
  - BLEU scores (`nltk.translate.bleu_score`)

### **Steps:**
1. Preprocess the text (lowercasing, removing stopwords and special characters).
2. Generate summaries using the BART model.
3. Evaluate the generated summaries against reference summaries using ROUGE and BLEU.
4. Display evaluation metrics and sample results.

### **Code Snippet:**
```python
rouge_scores = compute_rouge(generated_summaries, reference_summaries)
bleu_scores = compute_bleu(generated_summaries, reference_summaries)
print(f"ROUGE-1 F1: {rouge1:.4f}, BLEU: {mean_bleu:.4f}")
```

---

## **Project Requirements**

Install the following libraries before running the code:

```bash
pip install numpy pandas transformers torch nltk rouge-score datasets gensim
```

### **Additional Downloads:**
- Pre-trained GloVe embeddings: `gensim.downloader`.
- NLTK stopwords: Run `nltk.download('stopwords')`.

---

## **Execution Steps**

1. Clone the repository:
   ```bash
   git clone <repo-link>
   cd <repo-folder>
   ```
2. Run individual scripts:
   - `bow_tfidf.py` for BoW and TF-IDF.
   - `embeddings.py` for GloVe and Word2Vec.
   - `sentiment_analysis.py` for sentiment analysis.
   - `summarization_eval.py` for summarization and evaluation.
3. Outputs are saved in CSV files or displayed on the console.

---

## **Results**

### **Sample Outputs:**
- **Sentiment Analysis:**
   | Dialogue              | Sentiment |
   |-----------------------|-----------|
   | I love this product   | Positive  |
   | I hate this product   | Negative  |

- **ROUGE and BLEU Scores (Summarization):**
   ```
   ROUGE-1 F1 Score: 0.5241
   ROUGE-2 F1 Score: 0.3782
   ROUGE-L F1 Score: 0.4913
   BLEU Score: 0.3021
   ```

---

## **License**
This project is licensed under the MIT License. Feel free to use and extend it!

---

## **Contact**
For any queries or contributions, reach out at [your_email@example.com](mailto:your_email@example.com).

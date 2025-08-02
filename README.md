# Sentiment Analysis of AI Education Discussions on X (Indonesia)

This repository contains a Jupyter Notebook (`TML_Analisis_Sentimen_Bahasa_Indonesia.ipynb`) that performs sentiment analysis on Indonesian-language posts and comments from X (formerly Twitter) related to "Pendidikan AI" (AI Education). The project aims to classify these discussions into 'positive', 'negative', or 'neutral' sentiments using various machine learning models.

## 🎯 Project Goal

The primary goal of this project is to analyze public sentiment regarding AI education in Indonesia based on social media data. By classifying posts and comments, we can gain insights into the general perception, identify key concerns, and highlight areas of support or criticism.

## 📊 Dataset

The dataset used in this analysis consists of **over 500 posts and comments scraped from X** (formerly Twitter) that discuss "Pendidikan AI" (AI Education) in Indonesia. A portion of this data (`x-data-labeled.csv`) has been manually labeled into three sentiment categories:
* `positive`
* `negative`
* `neutral`

The notebook also includes an optional step to predict sentiment on an unlabeled dataset (`x-data-all.csv`).

## 🛠️ Methodology

The sentiment analysis pipeline involves several key steps:

### 1. Data Preprocessing
Before model training, raw text data undergoes thorough preprocessing to ensure quality and consistency:
* **Lowercasing:** All text is converted to lowercase.
* **URL Removal:** URLs (http/https links, www links) are removed.
* **Mention Removal:** User mentions (e.g., `@username`) are removed.
* **Hashtag Handling:** Hashtag symbols (`#`) are removed, but the hashtagged words are retained (e.g., `#AIeducation` becomes `AIeducation`).
* **Punctuation Removal:** All punctuation marks are removed.
* **Elongation Handling:** Repeated characters are reduced (e.g., `baguuuus` becomes `bagus`).
* **Whitespace Trimming:** Leading/trailing whitespaces are removed.
* **Slang Replacement:** Common Indonesian slang words are replaced with their formal equivalents (e.g., `ga` to `tidak`, `bgt` to `sangat`).
* **Emoji Replacement:** Emojis are replaced with corresponding sentiment labels (e.g., `😡` to `negatif`, `😍` to `positif`).
* **Stopword Removal:** Common Indonesian stopwords (e.g., `yang`, `dan`, `di`) are removed using `nltk.corpus.stopwords`.
* **Stemming:** Words are reduced to their root form using the Sastrawi Stemmer (e.g., `pendidikan` becomes `didik`).
* **Sarcasm Markers:** Specific sarcasm phrases (e.g., `banget ya`, `ndasmu`) are identified and added as a "sarcasm" token to the text to potentially aid the model.

### 2. Feature Extraction
* **TF-IDF Vectorization:** Text data is converted into numerical features using `TfidfVectorizer`. This technique reflects the importance of a word in a document relative to the entire corpus.
    * `ngram_range=(1, 2)`: Considers both single words (unigrams) and two-word phrases (bigrams).
    * `max_features=7000`: Limits the vocabulary size to the top 7000 most frequent terms.
    * `min_df=5`: Ignores terms that appear in less than 5 documents.
    * `max_df=0.8`: Ignores terms that appear in more than 80% of the documents.

### 3. Data Splitting & Resampling
* **Train-Test Split:** The labeled dataset is split into training (80%) and testing (20%) sets using `train_test_split` with `stratify=y` to maintain the original class distribution in both sets.
* **ADASYN Oversampling:** To address class imbalance in the training data, the ADASYN (Adaptive Synthetic Sampling) technique is applied. This generates synthetic samples for minority classes, balancing the dataset for better model performance. The `sampling_strategy` is set to oversample 'negative' and 'positive' to 148 samples, and 'neutral' to 250 samples, based on observations from the original data distribution. `n_neighbors=1` is used for robustness with sparse data.

### 4. Model Training & Hyperparameter Tuning
Two classification models are explored:

* **LinearSVC (Linear Support Vector Classifier):**
    * Hyperparameters are tuned using `GridSearchCV` with 5-fold cross-validation.
    * `param_grid` includes `C` (regularization parameter), `penalty` (`l1`, `l2`), `loss` (`hinge`, `squared_hinge`), and `class_weight` (`balanced`, `None`).
    * `dual=False` is explicitly set for `l1` penalty.
    * `scoring='f1_weighted'` is used as the evaluation metric, which is suitable for imbalanced datasets.

* **Multinomial Naive Bayes (MNB):**
    * Hyperparameter `alpha` (Laplace/Lidstone smoothing parameter) is tuned using `GridSearchCV`.
    * `param_grid` includes `alpha` values like `0.01, 0.1, 0.5, 1.0, 10.0`.
    * `scoring='f1_weighted'` is used for evaluation.

### 5. Evaluation
Model performance is assessed using:
* **Classification Report:** Provides precision, recall, F1-score, and support for each class.
* **Confusion Matrix:** Visualizes the number of correct and incorrect predictions for each class.

### 6. Prediction on Unlabeled Data
The best-performing model (Multinomial Naive Bayes with `alpha=0.01` and a neutral threshold of `0.3`) is then used to predict sentiment on the unlabeled `x-data-all.csv` dataset.

* **Threshold Adjustment for 'Neutral' Class:** An optional step is included to adjust the prediction threshold for the 'neutral' class. This can be useful for fine-tuning the balance between precision and recall for this specific class, especially in cases where 'neutral' might be harder to distinguish or less critical than 'positive'/'negative' sentiments. The notebook demonstrates trying various thresholds (e.g., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4) and evaluating the classification report and confusion matrix for each.

### 7. Visualization
* **Predicted Label Distribution:** A bar plot shows the count of each predicted sentiment label in the unlabeled data.
* **Word Cloud:** A word cloud is generated from all preprocessed text (both labeled and unlabeled) to visualize the most frequent terms.
* **Sample Comments:** Displays a few sample comments for each predicted sentiment category.

## 🚀 Key Findings (Based on provided output)

### Linear SVC (Tuned Model):
* **Best Parameters:** `{'C': 10, 'class_weight': 'balanced', 'loss': 'squared_hinge', 'penalty': 'l2'}`
* **Best Cross-Validation F1-weighted Score:** `0.872`
* **Test Set Accuracy:** `0.65`
* **F1-scores on Test Set:**
    * Negative: `0.73`
    * Neutral: `0.18` (This indicates difficulty in classifying neutral sentiment with this model configuration)
    * Positive: `0.72`

### Multinomial Naive Bayes (MNB):
* **Best Parameters:** `{'alpha': 0.01}`
* **Best Cross-Validation F1-weighted Score:** `0.908`
* **Test Set Accuracy:** `0.56` (without threshold adjustment)
* **F1-scores on Test Set (without threshold adjustment):**
    * Negative: `0.63`
    * Neutral: `0.30`
    * Positive: `0.57`

### Multinomial Naive Bayes with Neutral Threshold Adjustment:
The notebook explores various thresholds for the 'neutral' class. For instance, with `NEUTRAL_THRESHOLD = 0.3`:
* **Test Set Accuracy:** `0.59`
* **F1-scores on Test Set:**
    * Negative: `0.67`
    * Neutral: `0.45` (Improved compared to no threshold adjustment)
    * Positive: `0.57`

The threshold adjustment for the 'neutral' class significantly improved its recall and F1-score, indicating that the model was initially too conservative in predicting 'neutral' sentiment.

### Predicted Sentiment Distribution on Unlabeled Data:
(Based on the MNB model with `NEUTRAL_THRESHOLD = 0.3`)
* Negative: 67
* Neutral: 52
* Positive: 46

The analysis suggests a slightly more negative sentiment overall in the scraped X data regarding AI education in Indonesia, followed by neutral and positive.

## ⚙️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Place your data files:**
    * Ensure `x-data-labeled.csv` (the labeled dataset) is in the same directory as the notebook.
    * If you have unlabeled data for prediction, place `x-data-all.csv` in the same directory.
3.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook TML_Analisis_Sentimen_Bahasa_Indonesia.ipynb
    ```
4.  **Install dependencies:** Run the first few cells in the notebook to install necessary libraries (`nltk`, `wordcloud`, `Sastrawi`, `scikit-learn`, `imbalanced-learn`, `transformers`, `torch`).
5.  **Run all cells:** Execute all cells in the notebook sequentially. The notebook will load data, preprocess it, train and evaluate models, and perform predictions on unlabeled data if `x-data-all.csv` is present.

## 📦 Dependencies

The project relies on the following Python libraries:
* `pandas`
* `numpy`
* `re`
* `string`
* `nltk` (Natural Language Toolkit)
* `wordcloud`
* `Sastrawi` (Indonesian Stemmer)
* `scikit-learn` (for TF-IDF, LinearSVC, MultinomialNB, GridSearchCV, train_test_split, classification_report, confusion_matrix)
* `imbalanced-learn` (for SMOTE/ADASYN)
* `matplotlib`
* `seaborn`

## 🔮 Future Work

* **More Advanced Models:** Explore deep learning models (e.g., LSTMs, BERT-based models) for potentially higher accuracy, especially with larger datasets.
* **Larger and More Diverse Dataset:** Expand the dataset to include more posts and comments from various platforms and time periods to improve generalization.
* **Domain-Specific Lexicon:** Develop a more comprehensive slang dictionary and emoji mapping specifically tailored to Indonesian social media discourse on education and technology.
* **Aspect-Based Sentiment Analysis:** Instead of just overall sentiment, identify specific aspects of AI education (e.g., curriculum, teacher training, student readiness) and analyze sentiment towards each.
* **Real-time Analysis:** Implement a system for real-time sentiment analysis of incoming social media data.

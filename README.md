#  Assignment Title

## (1) Problem Statement
Analyzing public sentiment toward video games is a common challenge in Natural Language Processing (NLP) and social media monitoring. The problem is to effectively categorize unstructured text data (tweets) about the game "Red Dead Redemption 2" to understand the balance of player opinions and determine which machine learning algorithm best classifies this text.

## (2) Objective
To manually create a dataset of 100 tweets regarding RDR2, tag them by sentiment, preprocess the text data, and evaluate the performance of three machine learning classifiers (Naïve Bayes, SVM, and Logistic Regression) to determine their precision and recall.

## (3) Dataset
- Source:Synthetically generated realistic tweets (to simulate real Twitter discourse and comply with current API restrictions).
- Features:Tweet (raw text input), Clean_Tweet (preprocessed text), and Label (target variable: Positive, Negative, or Neutral).
- Size:100 rows (35 Positive, 35 Negative, 30 Neutral).

## (4) Methodology
1. Data Preprocessing  :The raw text was converted to lowercase, and all special characters/punctuation were removed using regular expressions. The text was then converted into numerical features using TF-IDF Vectorization, with standard English stop words removed.
2. EDA  : The dataset was balanced with a slight lean towards polarized opinions (35% Positive, 35% Negative, 30% Neutral) to ensure the models had sufficient examples of each class.
3. Model Building  : The dataset was split into an 80% training set (80 tweets) and a 20% testing set (20 tweets). Three classifiers were trained: Multinomial Naïve Bayes, Support Vector Machine (Linear Kernel), and Logistic Regression.
4. Evaluation  : Models were evaluated on the test set using weighted Precision and Recall scores to account for the slight class distribution differences.

## (5) Results
- Metrics and insights : All three models (Naïve Bayes, SVM, Logistic Regression) achieved a perfect 1.00 Precision and 1.00 Recall on the test set.

Insight: This flawless performance is a direct artifact of the synthetic data generation process; because the 100 tweets were generated using repeated foundational sentences, the models perfectly memorized the phrasing during the training phase. In a real-world scenario with highly varied text, SVM with a linear kernel typically performs the best among the three for this type of sparse, high-dimensional TF-IDF data.

## (6) How to Run
```bash
pip install -r requirements.txt
python main.py
```

## (7) Conclusion
Summarize findings.The project successfully demonstrates an end-to-end NLP pipeline—from data collection and text preprocessing to model training and evaluation. While the synthetic nature of the dataset resulted in perfect metric scores, the pipeline itself is fully functional, properly structured, and ready to scale with real-world, unrefined Twitter data.

## (8) Student's details
- Name:Gulam Rabbani Ansari
- Roll No:001
- UIN:231A006
- YEAR: TE-AIDS

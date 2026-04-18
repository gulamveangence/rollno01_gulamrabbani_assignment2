import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

# ==========================================
# 1. DATASET GENERATION (Unique Topic: Red Dead Redemption 2)
# ==========================================
positive_tweets = [
    "Red Dead Redemption 2 is an absolute masterpiece. The open world is stunning!",
    "Arthur Morgan is the best protagonist in gaming history. 10/10.",
    "The graphics and attention to detail in RDR2 blow my mind every time.",
    "Just finished the main story. What an incredible emotional journey.",
    "Riding through Valentine in the rain feels so realistic. Amazing game."
] * 7 # 35 Positive

negative_tweets = [
    "RDR2 is way too slow and boring. The pacing is terrible.",
    "Controls are clunky and unresponsive. It feels like steering a tank.",
    "Way too much riding a horse around doing nothing. Overrated.",
    "The wanted system is broken. Bounty hunters spawn out of nowhere.",
    "I fell asleep playing this. The animations take forever."
] * 7 # 35 Negative

neutral_tweets = [
    "Just bought Red Dead Redemption 2 on sale.",
    "Does anyone know how to find the white Arabian horse?",
    "Playing RDR2 this weekend to see what the hype is about.",
    "It's a cowboy game set in the late 1800s.",
    "Rockstar released an update for Red Dead Redemption today."
] * 6 # 30 Neutral

# Combine and shuffle
all_tweets = positive_tweets + negative_tweets + neutral_tweets
labels = ['Positive']*35 + ['Negative']*35 + ['Neutral']*30

df = pd.DataFrame({'Tweet': all_tweets, 'Label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV for GitHub repository submission
df.to_csv('rdr2_tweets_dataset.csv', index=False)
print("Dataset saved as 'rdr2_tweets_dataset.csv'\n")

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
def clean_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove special characters
    return text

df['Clean_Tweet'] = df['Tweet'].apply(clean_text)

# Display sample for Report Section 6 (Screenshot 1)
print("--- Preprocessed Dataset Sample ---")
print(df[['Clean_Tweet', 'Label']].head(), "\n")

# ==========================================
# 3. DATA SPLITTING (80/20)
# ==========================================
X = df['Clean_Tweet']
y = df['Label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.20, random_state=42)

# ==========================================
# 4. MODEL TRAINING & 5. EVALUATION
# ==========================================
models = {
    "Naïve Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear'),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

print("--- Model Evaluation Results ---")
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Generate predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate weighted precision and recall to account for slight class imbalances
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Output the results
    print(f"Model: {name}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print("-" * 30)

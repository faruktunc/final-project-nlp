# improved_preprocessing.py
import os
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download NLTK data (needed for first run)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Set paths relative to the script location
base_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(base_dir, "..", "data", "processed", "real_estate_data.csv")
output_dir = os.path.join(base_dir, "..", "data", "processed")
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(input_csv)

# Initialize stemmers and lemmatizers
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


# Processing functions
def preprocess_text(text):
    """Process text to get clean tokens"""
    if pd.isna(text):
        return []

    # Convert to lowercase and remove punctuation
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Filter out stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    return tokens


def lemmatize_tokens(tokens):
    """Apply lemmatization to tokens"""
    return [lemmatizer.lemmatize(word) for word in tokens]


def stem_tokens(tokens):
    """Apply stemming to tokens"""
    return [stemmer.stem(word) for word in tokens]


# Apply preprocessing
df['clean_tokens'] = df['description'].apply(preprocess_text)

# Create versions with different processing
df['lemmatized_tokens'] = df['clean_tokens'].apply(lemmatize_tokens)
df['stemmed_tokens'] = df['clean_tokens'].apply(stem_tokens)

# Convert token lists to strings for easier model training later
df['lemmatized_text'] = df['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))
df['stemmed_text'] = df['stemmed_tokens'].apply(lambda tokens: ' '.join(tokens))

# Save to CSVs
processed_csv = os.path.join(output_dir, "processed_real_estate.csv")
lemmatized_csv = os.path.join(output_dir, "lemmatized_real_estate.csv")
stemmed_csv = os.path.join(output_dir, "stemmed_real_estate.csv")

# Save full processed dataset
df.to_csv(processed_csv, index=False)
print(f"[✓] Processing complete. Data saved to '{processed_csv}'")

# Save specialized versions for direct use in models
df[['url', 'price', 'location', 'description', 'lemmatized_text']].to_csv(lemmatized_csv, index=False)
df[['url', 'price', 'location', 'description', 'stemmed_text']].to_csv(stemmed_csv, index=False)
print(f"[✓] Lemmatized data saved to '{lemmatized_csv}'")
print(f"[✓] Stemmed data saved to '{stemmed_csv}'")

# Print some statistics
print("\nData Statistics:")
print(f"Total documents: {len(df)}")
print(f"Average tokens per document: {df['clean_tokens'].apply(len).mean():.1f}")
print(f"Unique tokens before lemmatization: {len(set([t for tokens in df['clean_tokens'] for t in tokens]))}")
print(f"Unique tokens after lemmatization: {len(set([t for tokens in df['lemmatized_tokens'] for t in tokens]))}")
print(f"Unique tokens after stemming: {len(set([t for tokens in df['stemmed_tokens'] for t in tokens]))}")
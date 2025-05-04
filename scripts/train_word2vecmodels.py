# improved_word2vec_training.py
import os
import time
import pandas as pd
from gensim.models import Word2Vec
import logging

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set paths
base_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_path = os.path.join(base_dir, "..", "data", "processed", "processed_real_estate.csv")
models_dir = os.path.join(base_dir, "..", "models")
os.makedirs(models_dir, exist_ok=True)

# Initialize list to store model training results
model_results = []

# Read preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv(processed_data_path)


# Convert string representations of token lists back to actual lists
def str_to_tokens(token_str):
    if pd.isna(token_str):
        return []
    # The tokens are stored as a string representation of a list
    # We need to convert back to an actual list of strings
    try:
        # Handle the format: "['token1', 'token2', ...]"
        if token_str.startswith('[') and token_str.endswith(']'):
            # Remove brackets and split by commas
            items = token_str[1:-1].split(', ')
            # Remove quotes from each item
            return [item.strip("'\"") for item in items if item.strip("'\"")]
        else:
            # If it's just a space-separated string
            return token_str.split()
    except:
        return []


# Prepare corpora for training
print("Preparing text corpora...")
lemmatized_corpus = []
stemmed_corpus = []

# Extract token lists from dataframe
for _, row in df.iterrows():
    lemma_tokens = str_to_tokens(row.get('lemmatized_tokens', ''))
    stem_tokens = str_to_tokens(row.get('stemmed_tokens', ''))

    if lemma_tokens:
        lemmatized_corpus.append(lemma_tokens)
    if stem_tokens:
        stemmed_corpus.append(stem_tokens)

print(f"Loaded {len(lemmatized_corpus)} documents for lemmatized corpus")
print(f"Loaded {len(stemmed_corpus)} documents for stemmed corpus")

# Training parameters
parameters = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300},
]


def train_and_save_model(corpus, params, name_prefix):
    """Train and save a Word2Vec model with given parameters"""
    if not corpus or len(corpus) == 0:
        print(f"WARNING: Empty corpus for {name_prefix}. Skipping model training.")
        return None, None, None, None

    # Set skip-gram parameter (sg=1 for skip-gram, sg=0 for CBOW)
    sg = 1 if params['model_type'] == 'skipgram' else 0

    print(
        f"Training {name_prefix} model: {params['model_type']}, window={params['window']}, dim={params['vector_size']}")

    # Start timing the training process
    start_time = time.time()

    # Train model
    model = Word2Vec(
        corpus,
        vector_size=params['vector_size'],
        window=params['window'],
        min_count=1,  # Keep even rare words
        sg=sg,
        workers=4  # Use multiple cores for faster training
    )

    # Calculate training time
    training_time = time.time() - start_time

    # Save model
    filename = f"{name_prefix}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}.model"
    model_path = os.path.join(models_dir, filename)
    model.save(model_path)

    # Get some basic model statistics
    vocab_size = len(model.wv.index_to_key)
    print(f"Model saved: {filename} (vocabulary size: {vocab_size})")
    print(f"Training time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

    # Return model info for summary table
    return model_path, filename, vocab_size, training_time


# Train all models
print("\nTraining Word2Vec models...")

# Overall timing for all models
overall_start_time = time.time()

for param in parameters:
    # Train lemmatized model
    lemma_model_path, lemma_filename, lemma_vocab_size, lemma_training_time = train_and_save_model(
        lemmatized_corpus, param, "lemmatized")

    if lemma_filename:
        # Add lemmatized model results to summary
        model_results.append({
            "Model Adı": lemma_filename,
            "Model Tipi": param['model_type'],
            "İşlenen Veri": "Lemmatized",
            "Pencere (Window)": param['window'],
            "Vektör Boyutu": param['vector_size'],
            "Eğitim Süresi (sn)": f"{lemma_training_time:.2f}",
            "Kelime Sayısı": lemma_vocab_size
        })

    # Train stemmed model
    stem_model_path, stem_filename, stem_vocab_size, stem_training_time = train_and_save_model(
        stemmed_corpus, param, "stemmed")

    if stem_filename:
        # Add stemmed model results to summary
        model_results.append({
            "Model Adı": stem_filename,
            "Model Tipi": param['model_type'],
            "İşlenen Veri": "Stemmed",
            "Pencere (Window)": param['window'],
            "Vektör Boyutu": param['vector_size'],
            "Eğitim Süresi (sn)": f"{stem_training_time:.2f}",
            "Kelime Sayısı": stem_vocab_size
        })

# Calculate total time for all models
total_time = time.time() - overall_start_time
print(f"\nTotal training time for all models: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
print("\nTraining complete! All models saved to:", models_dir)

# Print model summary table without using tabulate
print("\n\n" + "=" * 110)
print("MODEL EĞİTİM SONUÇLARI ÖZET TABLOSU")
print("=" * 110)

# Define column headers and widths
headers = ["Model Adı", "Model Tipi", "İşlenen Veri", "Pencere", "Vektör Boyutu", "Eğitim Süresi (sn)", "Kelime Sayısı"]
widths = [40, 10, 13, 8, 15, 18, 15]

# Print header row
header_format = ''.join(f'{{:{w}}}' for w in widths)
print(header_format.format(*headers))
print("-" * 110)

# Print each data row
for result in model_results:
    row = [
        result["Model Adı"],
        result["Model Tipi"],
        result["İşlenen Veri"],
        str(result["Pencere (Window)"]),
        str(result["Vektör Boyutu"]),
        result["Eğitim Süresi (sn)"],
        str(result["Kelime Sayısı"])
    ]
    print(header_format.format(*row))

print("=" * 110)
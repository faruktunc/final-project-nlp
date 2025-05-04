import os
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import numpy as np
from collections import Counter
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NLTK verilerini indir
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Dosya yolları
base_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(base_dir, "..", "data", "processed", "real_estate_data.csv")
output_dir = os.path.join(base_dir, "..", "data", "processed")
plots_dir = os.path.join(base_dir, "..", "plots")
os.makedirs(plots_dir, exist_ok=True)

# Veriyi oku
df = pd.read_csv(raw_data_path)
# NaN değerleri temizle ve boş string olmadığından emin ol
descriptions = df['description'].fillna('').astype(str).tolist()
descriptions = [desc for desc in descriptions if desc.strip()]  # Boş stringleri filtrele


# Önişleme fonksiyonları
def tokenize_text(text):
    return word_tokenize(text.lower())


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token.isalpha() and token not in stop_words]


def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


# Ham veri için tüm belgeleri birleştir ve tokenize et
raw_tokens = []
for desc in descriptions:
    raw_tokens.extend(tokenize_text(desc))

# Temizlenmiş tokenler
clean_tokens = []
for desc in descriptions:
    tokens = tokenize_text(desc)
    clean_tokens.extend(remove_stopwords(tokens))

# Lemmatize ve stem tokenler
lemmatized_tokens = lemmatize_tokens(clean_tokens)
stemmed_tokens = stem_tokens(clean_tokens)


# CSV'leri oluştur
def create_csv_from_tokens(tokens, filename):
    # Token listesini döküman başına grupla (burada her description bir döküman olarak kabul edilir)
    docs = []
    for desc in descriptions:
        if not desc or not desc.strip():  # Boş string kontrolü
            continue

        processed_tokens = []
        tokens_in_desc = tokenize_text(desc)
        tokens_in_desc = remove_stopwords(tokens_in_desc)

        if 'lemmatized' in filename:
            processed_tokens = lemmatize_tokens(tokens_in_desc)
        elif 'stemmed' in filename:
            processed_tokens = stem_tokens(tokens_in_desc)

        if processed_tokens:  # Boş token listesi kontrolü
            docs.append(" ".join(processed_tokens))
        else:
            docs.append("")  # Boş metin ekle ama en azından sırayı koru

    # DataFrame oluştur ve kaydet
    output_df = pd.DataFrame({'processed_text': docs})
    output_path = os.path.join(output_dir, filename)
    output_df.to_csv(output_path, index=False)
    print(f"CSV kaydedildi: {output_path}")
    return output_path


lemmatized_csv = create_csv_from_tokens(lemmatized_tokens, "lemmatized_data.csv")
stemmed_csv = create_csv_from_tokens(stemmed_tokens, "stemmed_data.csv")


# Zipf analizi fonksiyonu
def plot_zipf(tokens, title, filename):
    # Kelime frekanslarını hesapla
    word_counts = Counter(tokens)

    # Frekans sırasına göre sırala
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    ranks = np.arange(1, len(sorted_word_counts) + 1)
    frequencies = [count for word, count in sorted_word_counts]

    # Log-log grafiği çiz
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, marker='.', linestyle='none')

    # Teorik Zipf eğrisi (1/rank)
    k = frequencies[0]  # en sık geçen kelimenin frekansı
    theoretical_zipf = [k / r for r in ranks]
    plt.loglog(ranks, theoretical_zipf, 'r-', alpha=0.7, label='Teorik Zipf (1/rank)')

    plt.title(title)
    plt.xlabel('Rank (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Grafiği kaydet
    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Zipf grafiği kaydedildi: {output_path}")
    return output_path


# Zipf grafiklerini oluştur
raw_zipf_plot = plot_zipf(raw_tokens, 'Zipf Analizi - Ham Veri', 'raw_zipf.png')
lemmatized_zipf_plot = plot_zipf(lemmatized_tokens, 'Zipf Analizi - Lemmatized Veri', 'lemmatized_zipf.png')
stemmed_zipf_plot = plot_zipf(stemmed_tokens, 'Zipf Analizi - Stemmed Veri', 'stemmed_zipf.png')

# Veri setinin istatistiklerini yazdır
print("\nVeri Seti İstatistikleri:")
print(f"Döküman Sayısı: {len(descriptions)}")
print(f"Ham Token Sayısı: {len(raw_tokens)}")
print(f"Temizlenmiş Token Sayısı: {len(clean_tokens)}")
print(f"Lemmatized Token Sayısı: {len(lemmatized_tokens)}")
print(f"Stemmed Token Sayısı: {len(stemmed_tokens)}")
print(f"Ham Veri Benzersiz Kelime Sayısı: {len(set(raw_tokens))}")
print(f"Lemmatized Veri Benzersiz Kelime Sayısı: {len(set(lemmatized_tokens))}")
print(f"Stemmed Veri Benzersiz Kelime Sayısı: {len(set(stemmed_tokens))}")
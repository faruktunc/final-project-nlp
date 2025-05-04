import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Dosya yolları
base_dir = os.path.dirname(os.path.abspath(__file__))
lemmatized_path = os.path.join(base_dir, "..", "data", "processed", "lemmatized_data.csv")
stemmed_path = os.path.join(base_dir, "..", "data", "processed", "stemmed_data.csv")
output_dir = os.path.join(base_dir, "..", "data", "processed")
plots_dir = os.path.join(base_dir, "..", "plots")

# Dizinleri oluştur
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Verileri oku
lemmatized_df = pd.read_csv(lemmatized_path)
stemmed_df = pd.read_csv(stemmed_path)

# NaN değerleri temizle
lemmatized_df['processed_text'] = lemmatized_df['processed_text'].fillna('')
stemmed_df['processed_text'] = stemmed_df['processed_text'].fillna('')


# TF-IDF vektörleştirme fonksiyonu
def create_tfidf_matrix(texts, max_features=None):
    """
    Metinler için TF-IDF vektörleştirme yapar ve DataFrame olarak döndürür
    """
    # NaN değerleri kontrol et ve temizle
    texts = ['' if pd.isna(text) else text for text in texts]

    # Hocanın koduna benzer şekilde TF-IDF vektörleştirmeyi yap
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Özellik isimleri
    feature_names = vectorizer.get_feature_names_out()

    # TF-IDF matrix'i DataFrame'e dönüştür
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    return df_tfidf, tfidf_matrix, vectorizer, feature_names


# Lemmatized veriler için TF-IDF
print("Lemmatized veri için TF-IDF hesaplanıyor...")
lemmatized_texts = lemmatized_df['processed_text'].tolist()
lemmatized_tfidf_df, lemmatized_tfidf_matrix, lemmatized_vectorizer, lemmatized_features = create_tfidf_matrix(
    lemmatized_texts)

# Stemmed veriler için TF-IDF
print("Stemmed veri için TF-IDF hesaplanıyor...")
stemmed_texts = stemmed_df['processed_text'].tolist()
stemmed_tfidf_df, stemmed_tfidf_matrix, stemmed_vectorizer, stemmed_features = create_tfidf_matrix(stemmed_texts)

# CSV olarak kaydet
lemmatized_tfidf_df.to_csv(os.path.join(output_dir, "tfidf_lemmatized.csv"), index=False)
stemmed_tfidf_df.to_csv(os.path.join(output_dir, "tfidf_stemmed.csv"), index=False)

print(f"Lemmatized TF-IDF DataFrame kaydedildi. Boyut: {lemmatized_tfidf_df.shape}")
print(f"Stemmed TF-IDF DataFrame kaydedildi. Boyut: {stemmed_tfidf_df.shape}")

# İlk 5 satırı göster (hocanın kodundaki gibi)
print("\nLemmatized TF-IDF matrisinin ilk 5 satırı:")
print(lemmatized_tfidf_df.head())

# Hocanın kodundaki gibi ilk cümle için TF-IDF skorlarını analiz et
if not lemmatized_tfidf_df.empty:
    print("\nİlk cümlede en yüksek TF-IDF skoruna sahip 5 kelime:")
    first_sentence_vector = lemmatized_tfidf_df.iloc[0]
    top_5_words = first_sentence_vector.sort_values(ascending=False).head(5)
    print(top_5_words)


# TF-IDF vektörlerinden örnek görselleştirme
def plot_tfidf_heatmap(tfidf_df, title, filename, sample_size=10, top_terms=20):
    """
    TF-IDF vektörlerinin örnek bir görselleştirmesini oluşturur
    """
    # Örnek dökümanları ve en yüksek TF-IDF değerlerine sahip terimleri seç
    if len(tfidf_df) > sample_size:
        sample_df = tfidf_df.sample(sample_size)
    else:
        sample_df = tfidf_df

    # Sütunlardaki ortalama değerlere göre en önemli terimleri bul
    top_terms_idx = tfidf_df.mean().nlargest(top_terms).index
    sample_df = sample_df[top_terms_idx]

    plt.figure(figsize=(12, 8))
    plt.imshow(sample_df.values, cmap='viridis', aspect='auto')
    plt.colorbar(label='TF-IDF Score')
    plt.xticks(range(len(top_terms_idx)), top_terms_idx, rotation=90)
    plt.yticks(range(len(sample_df)), [f"Doc {i + 1}" for i in range(len(sample_df))])
    plt.title(title)
    plt.tight_layout()

    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"TF-IDF görselleştirme kaydedildi: {output_path}")


# Görselleştirmeleri oluştur
plot_tfidf_heatmap(lemmatized_tfidf_df, "Lemmatized TF-IDF Vektörleri", "lemmatized_tfidf_heatmap.png")
plot_tfidf_heatmap(stemmed_tfidf_df, "Stemmed TF-IDF Vektörleri", "stemmed_tfidf_heatmap.png")


# Hocanın kodundaki gibi kelime benzerliği analizi ekle
def analyze_word_similarity(tfidf_matrix, feature_names, target_word, top_n=5):
    """
    Belirli bir kelimeye en benzer kelimeleri bulur
    """
    if target_word not in feature_names:
        print(f"'{target_word}' kelimesi veri setinde bulunamadı.")
        return

    # Hedef kelimenin indeksini bul
    target_index = feature_names.tolist().index(target_word)

    # Hedef kelimenin vektörünü al
    target_vector = tfidf_matrix[:, target_index].toarray()

    # Tüm kelimelerin vektörlerini al
    tfidf_vectors = tfidf_matrix.toarray()

    # Kosinüs benzerliğini hesapla
    similarities = cosine_similarity(target_vector.T, tfidf_vectors.T)
    similarities = similarities.flatten()

    # En benzer kelimeleri bul (hedef kelime dahil)
    top_indices = similarities.argsort()[-(top_n + 1):][::-1]

    print(f"\n'{target_word}' kelimesine en benzer {top_n} kelime:")
    for i, index in enumerate(top_indices):
        if i == 0 and feature_names[index] == target_word:
            continue  # Kelimenin kendisini atla
        print(f"{feature_names[index]}: {similarities[index]:.4f}")



# Yoksa veri setindeki ilk kelimeyi kullanabiliriz
sample_word = 'house'
if lemmatized_features is not None and len(lemmatized_features) > 0:
    if sample_word not in lemmatized_features:
        sample_word = lemmatized_features[0]

    analyze_word_similarity(lemmatized_tfidf_matrix, lemmatized_features, sample_word)
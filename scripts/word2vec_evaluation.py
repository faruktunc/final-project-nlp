import os
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import numpy as np

# Dosya yolları
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "..", "models")
plots_dir = os.path.join(base_dir, "..", "plots")
data_processed_dir = os.path.join(base_dir, "..", "data", "processed","similar_words")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(data_processed_dir, exist_ok=True)

# Önemli kelimeler (gayrimenkul alanında)
target_words = ["house", "apartment", "room", "kitchen", "bathroom", "garden", "price"]

# Tüm modelleri yükle ve değerlendir
model_files = [f for f in os.listdir(models_dir) if f.endswith('.model')]
model_results = []

for model_file in model_files:
    model_path = os.path.join(models_dir, model_file)
    try:
        model = Word2Vec.load(model_path)

        # Model parametrelerini çıkar
        model_info = {
            'filename': model_file,
            'is_lemmatized': 'lemmatized' in model_file,
            'model_type': 'CBOW' if 'cbow' in model_file else 'Skip-gram',
            'window_size': int(model_file.split('window')[1].split('_')[0]),
            'vector_size': int(model_file.split('dim')[1].split('.')[0]),
            'vocabulary_size': len(model.wv.index_to_key),
        }

        # Her hedef kelime için en benzer 5 kelimeyi bul
        similar_words = {}
        for word in target_words:
            if word in model.wv:
                similar = model.wv.most_similar(word, topn=5)
                similar_words[word] = similar
            else:
                similar_words[word] = []

        model_info['similar_words'] = similar_words
        model_results.append(model_info)
        print(f"Model değerlendirildi: {model_file}")
    except Exception as e:
        print(f"Hata: {model_file} yüklenemedi: {e}")

# Sonuçları DataFrame'e dönüştür
results_df = pd.DataFrame(model_results)

# Sonuçları CSV olarak kaydet
results_path = os.path.join(data_processed_dir, "word2vec_model_evaluation.csv")
results_df[['filename', 'is_lemmatized', 'model_type', 'window_size', 'vector_size', 'vocabulary_size']].to_csv(
    results_path, index=False)
print(f"Model değerlendirme sonuçları kaydedildi: {results_path}")


# Örnek olarak bir model seçip görselleştirme
def visualize_word_vectors(model_file, target_words, filename):
    model_path = os.path.join(models_dir, model_file)
    model = Word2Vec.load(model_path)

    # Hedef kelimeler ve benzer kelimelerden oluşan liste
    words_to_plot = set()
    for word in target_words:
        if word in model.wv:
            words_to_plot.add(word)
            similars = [pair[0] for pair in model.wv.most_similar(word, topn=3)]
            words_to_plot.update(similars)

    # Listedeki kelimeler için vektörleri al
    word_vectors = np.array([model.wv[word] for word in words_to_plot if word in model.wv])
    words = [word for word in words_to_plot if word in model.wv]

    # Vektör sayısını kontrol et
    if len(word_vectors) < 2:
        print(f"Uyarı: {model_file} için yeterli vektör bulunamadı, görselleştirme atlanıyor.")
        return

    # t-SNE için perplexity değerini ayarla (en az 2 veya vektör sayısının 1 eksiği olacak şekilde)
    perplexity = min(30, max(2, len(word_vectors) - 1))

    # t-SNE ile 2 boyuta indirgeme
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    vectors_2d = tsne.fit_transform(word_vectors)

    # Görselleştirme
    plt.figure(figsize=(12, 10))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)

    # Kelime etiketlerini ekle
    for i, word in enumerate(words):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                     xytext=(5, 2), textcoords='offset points',
                     fontsize=10, alpha=0.8)

    plt.title(f"Word2Vec Model Görselleştirmesi: {model_file}")
    plt.tight_layout()
    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Word2Vec görselleştirme kaydedildi: {output_path}")


# Lemmatized ve stemmed modeller için görselleştirme
if model_results:
    lemmatized_models = [m for m in model_results if m['is_lemmatized']]
    stemmed_models = [m for m in model_results if not m['is_lemmatized']]

    if lemmatized_models:
        # En yüksek boyutlu modeli seç
        best_lemmatized = max(lemmatized_models, key=lambda x: x['vector_size'])
        visualize_word_vectors(best_lemmatized['filename'], target_words, "lemmatized_word2vec_vectors.png")

    if stemmed_models:
        best_stemmed = max(stemmed_models, key=lambda x: x['vector_size'])
        visualize_word_vectors(best_stemmed['filename'], target_words, "stemmed_word2vec_vectors.png")

# Model sözlük boyutlarının karşılaştırması
plt.figure(figsize=(10, 6))
lemmatized_sizes = [m['vocabulary_size'] for m in model_results if m['is_lemmatized']]
stemmed_sizes = [m['vocabulary_size'] for m in model_results if not m['is_lemmatized']]

if lemmatized_sizes and stemmed_sizes:
    bars = plt.bar(['Lemmatized', 'Stemmed'], [np.mean(lemmatized_sizes), np.mean(stemmed_sizes)])
    plt.title('Ortalama Sözlük Boyutları Karşılaştırması')
    plt.ylabel('Ortalama Kelime Sayısı')
    plt.grid(axis='y', alpha=0.3)

    # Sayısal değerleri çubukların üzerine ekle
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
                 f'{int(height)}', ha='center', va='bottom')

    plt.savefig(os.path.join(plots_dir, "vocabulary_size_comparison.png"))
    plt.close()
    print("Sözlük boyutları karşılaştırma grafiği kaydedildi.")


# Her model için benzer kelime tablosu oluştur
def create_similarity_table(model_info):
    filename = model_info['filename']
    model_type = model_info['model_type']
    window = model_info['window_size']
    dim = model_info['vector_size']

    table_data = []
    for target_word, similar_words in model_info['similar_words'].items():
        if similar_words:
            similar_str = ", ".join([f"{word}({score:.2f})" for word, score in similar_words])
            table_data.append([target_word, similar_str])

    # Benzer kelimeler tablosunu kaydet
    df = pd.DataFrame(table_data, columns=['Target Word', 'Most Similar Words'])
    table_filename = f"similar_words_{filename.replace('.model', '')}.csv"
    df.to_csv(os.path.join(data_processed_dir, table_filename), index=False)
    print(f"Benzer kelimeler tablosu kaydedildi: {table_filename}")


# Her model için benzer kelimeler tablosu oluştur
for model_info in model_results:
    create_similarity_table(model_info)
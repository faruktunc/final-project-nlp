# word2vec_visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.cm as cm
import argparse


def load_model(model_path):
    """Verilen yoldan word2vec modelini yükle"""
    print(f"Model yükleniyor: {model_path}")
    model = Word2Vec.load(model_path)
    return model


def get_most_frequent_words(model, n=100):
    """Modelden en sık kullanılan n kelimeyi getir"""
    vocab = model.wv.index_to_key[:n]  # Gensim 4.x için
    return vocab


def reduce_dimensions(model, words):
    """Kelime vektörlerini t-SNE ile 2 boyuta indirge"""
    print("Kelime vektörleri çıkarılıyor...")
    word_vectors = np.array([model.wv[word] for word in words])

    print("t-SNE ile boyut indirgeme yapılıyor...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    reduced_vectors = tsne.fit_transform(word_vectors)

    return reduced_vectors


def visualize_embeddings(words, vectors, title="Word2Vec Kelime Vektörleri", n_annotate=50):
    """İndirgenmiş vektörleri görselleştir ve kelimeleri etiketle"""
    plt.figure(figsize=(16, 16))
    plt.scatter(vectors[:, 0], vectors[:, 1], c=np.arange(len(vectors)), cmap=cm.rainbow, alpha=0.7)

    # Sadece n_annotate kadar kelimeyi etiketle (çok kalabalık olmaması için)
    for i, word in enumerate(words[:n_annotate]):
        plt.annotate(word,
                     xy=(vectors[i, 0], vectors[i, 1]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     fontsize=10)

    plt.title(title)
    plt.xlabel("Boyut 1")
    plt.ylabel("Boyut 2")
    plt.tight_layout()

    # Sonuçları kaydet
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Görselleştirme kaydedildi: {output_path}")

    # Grafik göster
    plt.show()


def compare_models(model_paths, top_n=100, n_annotate=30):
    """Birden fazla modeli yükle ve karşılaştır"""
    for model_path in model_paths:
        # Modeli yükle
        model = load_model(model_path)
        model_name = os.path.basename(model_path).replace('.model', '')

        # En sık kullanılan kelimeleri al
        words = get_most_frequent_words(model, top_n)

        # Vektörleri indirgeyerek 2 boyutlu hale getir
        reduced_vectors = reduce_dimensions(model, words)

        # Görselleştir
        visualize_embeddings(words, reduced_vectors, title=f"{model_name} - Top {top_n} Words", n_annotate=n_annotate)


def analyze_specific_words(model_path, specific_words, n_similar=10):
    """Belirli kelimelerin benzer kelimelerini analiz et ve görselleştir"""
    model = load_model(model_path)
    model_name = os.path.basename(model_path).replace('.model', '')

    # Her kelime için benzer kelimeleri bul
    similar_words = []
    for word in specific_words:
        if word in model.wv:
            # Benzer kelimeleri al
            similars = [item[0] for item in model.wv.most_similar(word, topn=n_similar)]
            # Orijinal kelimeyi ve benzerlerini listeye ekle
            similar_words.append(word)
            similar_words.extend(similars)
        else:
            print(f"'{word}' kelimesi modelin kelime dağarcığında bulunmuyor.")

    # Tekrar eden kelimeleri çıkar
    similar_words = list(set(similar_words))

    if similar_words:
        # Bu kelimelerin vektörlerini indirgeyerek 2 boyutlu hale getir
        reduced_vectors = reduce_dimensions(model, similar_words)

        # Görselleştir
        visualize_embeddings(similar_words, reduced_vectors,
                             title=f"{model_name} - Specific Words Analysis",
                             n_annotate=len(similar_words))


def main():
    parser = argparse.ArgumentParser(description='Word2Vec modeli görselleştirme aracı')
    parser.add_argument('--model', type=str, help='Görselleştirilecek model yolu')
    parser.add_argument('--models_dir', type=str, help='Tüm modellerin bulunduğu dizin')
    parser.add_argument('--top_n', type=int, default=100, help='Görselleştirilecek kelime sayısı')
    parser.add_argument('--words', type=str, nargs='+', help='Analiz edilecek özel kelimeler')

    args = parser.parse_args()

    if args.model and args.words:
        # Belirli kelimeleri analiz et
        analyze_specific_words(args.model, args.words)
    elif args.model:
        # Tek bir modeli görselleştir
        model = load_model(args.model)
        model_name = os.path.basename(args.model).replace('.model', '')
        words = get_most_frequent_words(model, args.top_n)
        reduced_vectors = reduce_dimensions(model, words)
        visualize_embeddings(words, reduced_vectors, title=f"{model_name} - Top {args.top_n} Words")
    elif args.models_dir:
        # Dizindeki tüm modelleri karşılaştır
        model_paths = [os.path.join(args.models_dir, f) for f in os.listdir(args.models_dir) if f.endswith('.model')]
        compare_models(model_paths, args.top_n)
    else:
        print("Lütfen --model veya --models_dir parametrelerinden birini belirtin.")


if __name__ == "__main__":
    main()
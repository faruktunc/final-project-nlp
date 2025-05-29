import os
import pandas as pd
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')

# === Dosya yolları (yeni proje dizin yapısına göre düzenlendi) ===
base_dir = os.path.dirname(os.path.abspath(__file__))  # scripts klasörü
project_root = os.path.dirname(base_dir)  # ana proje klasörü
models_dir = os.path.join(project_root, "models")
plots_dir = os.path.join(project_root, "plots")
processed_data_dir = os.path.join(project_root, "data", "processed")

# Plots klasörünü oluştur (zaten var ama emin olmak için)
os.makedirs(plots_dir, exist_ok=True)

# === Verileri Yükle ===
print("Veriler yükleniyor...")
try:
    lemmatized = pd.read_csv(os.path.join(processed_data_dir, "lemmatized_real_estate.csv"))
    stemmed = pd.read_csv(os.path.join(processed_data_dir, "stemmed_real_estate.csv"))
    print(f"Lemmatized veri yüklendi: {len(lemmatized)} satır")
    print(f"Stemmed veri yüklendi: {len(stemmed)} satır")
except FileNotFoundError as e:
    print(f"HATA: Veri dosyası bulunamadı: {e}")
    print(
        "Lütfen data/processed klasöründe lemmatized_real_estate.csv ve stemmed_real_estate.csv dosyalarının olduğundan emin olun.")
    exit(1)

# === Sütun adlarını belirle ===
print(f"Lemmatized sütunlar: {lemmatized.columns.tolist()}")
print(f"Stemmed sütunlar: {stemmed.columns.tolist()}")

# Veri yapısına göre sütun adlarını belirle
if 'lemmatized_text' in lemmatized.columns:
    lemma_col = 'lemmatized_text'
elif 'processed_text' in lemmatized.columns:
    lemma_col = 'processed_text'
else:
    lemma_col = lemmatized.columns[-1]  # Son sütunu al

if 'stemmed_text' in stemmed.columns:
    stem_col = 'stemmed_text'
elif 'processed_text' in stemmed.columns:
    stem_col = 'processed_text'
else:
    stem_col = stemmed.columns[-1]  # Son sütunu al

print(f"Kullanılan sütunlar - Lemma: {lemma_col}, Stem: {stem_col}")

# === Giriş Metni Belirle ===
input_text_lemma = "block"
input_text_stem = "block"

print(f"\n=== GİRİŞ METNİ ===")
print(f"Lemmatized: {input_text_lemma}")
print(f"Stemmed: {input_text_stem}")


# === Yardımcı Fonksiyonlar ===
def sentence_vector(model, sentence):
    if pd.isna(sentence) or sentence is None:
        return np.zeros(model.vector_size)

    sentence = str(sentence).strip()
    if not sentence:
        return np.zeros(model.vector_size)

    vectors = [model.wv[word] for word in word_tokenize(sentence) if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


def jaccard_similarity(set1, set2):
    intersection = len(set(set1).intersection(set2))
    union = len(set(set1).union(set2))
    return intersection / union if union > 0 else 0


# === TF-IDF Benzerlikleri ===
def tfidf_similarity(tfidf_model_csv_path, df, label, text_col, input_text):
    try:
        print(f"TF-IDF dosyası aranıyor: {tfidf_model_csv_path}")
        tfidf_df = pd.read_csv(tfidf_model_csv_path, index_col=0)
        df_clean = df.dropna(subset=[text_col]).copy()
        df_clean[text_col] = df_clean[text_col].astype(str)

        print(f"TF-IDF matrisi şekli: {tfidf_df.shape}")
        print(f"Temiz veri seti şekli: {df_clean.shape}")

        # Giriş metninin indeksini bul - daha esnek arama
        input_indices = df_clean.index[df_clean[text_col].str.contains(input_text, case=False, na=False)].tolist()
        if not input_indices:
            # Tam eşleşme bulamazsa, kelimeyi içeren metinleri ara
            input_indices = df_clean.index[
                df_clean[text_col].str.contains(f'\\b{input_text}\\b', case=False, na=False, regex=True)].tolist()

        if not input_indices:
            print(f"UYARI: '{input_text}' metni {label} veri setinde bulunamadı!")
            print(f"İlk 5 satırı kontrol edelim:")
            print(df_clean[text_col].head().tolist())
            return [], [], []

        input_index = input_indices[0]
        print(f"Bulunan indeks: {input_index}")

        # TF-IDF matrisinden ilgili satırı al
        if input_index >= len(tfidf_df):
            print(f"HATA: İndeks {input_index} TF-IDF matrisinde yok (max: {len(tfidf_df) - 1})")
            return [], [], []

        input_vec = tfidf_df.iloc[input_index].values.reshape(1, -1)
        tfidf_matrix = tfidf_df.values
        sims = cosine_similarity(input_vec, tfidf_matrix).flatten()

        # En benzer 5'i al
        top5_idx = sims.argsort()[-5:][::-1]

        print(f"\n=== TF-IDF ({label}) EN BENZER 5 SONUÇ ===")
        results = []
        for i, idx in enumerate(top5_idx):
            if idx < len(df_clean):
                text = df_clean[text_col].iloc[idx]
                score = sims[idx]
                print(f"{i + 1}. Skor: {score:.4f} | Metin: {text[:100]}...")
                results.append((idx, text, score))

        return [r[0] for r in results], [r[2] for r in results], [r[1] for r in results]

    except Exception as e:
        print(f"HATA - TF-IDF {label}: {e}")
        return [], [], []


# === Word2Vec Benzerlikleri ===
def w2v_similarity(model_path, df, label, text_col, input_text):
    try:
        print(f"Word2Vec modeli yükleniyor: {model_path}")
        if not os.path.exists(model_path):
            print(f"HATA: Model dosyası bulunamadı: {model_path}")
            return [], [], []

        model = Word2Vec.load(model_path)
        df_clean = df.dropna(subset=[text_col]).copy()
        df_clean[text_col] = df_clean[text_col].astype(str)

        input_vec = sentence_vector(model, input_text)
        vectors = [sentence_vector(model, sent) for sent in df_clean[text_col]]
        sims = cosine_similarity([input_vec], vectors).flatten()

        # En benzer 5'i al
        top5_idx = sims.argsort()[-5:][::-1]

        print(f"\n=== Word2Vec ({label}) EN BENZER 5 SONUÇ ===")
        results = []
        for i, idx in enumerate(top5_idx):
            text = df_clean[text_col].iloc[idx]
            score = sims[idx]
            print(f"{i + 1}. Skor: {score:.4f} | Metin: {text[:100]}...")
            results.append((idx, text, score))

        return [r[0] for r in results], [r[2] for r in results], [r[1] for r in results]

    except Exception as e:
        print(f"HATA - Word2Vec {label}: {e}")
        return [], [], []


# === Manuel Anlamsal Değerlendirme Fonksiyonu ===
def get_semantic_scores(model_name, texts, input_text):
    """
    Bu fonksiyon gerçek kullanımda kullanıcıdan manuel puan almalı.
    Şimdilik otomatik puanlar veriyoruz.
    """
    print(f"\n=== ANLAMSAL DEĞERLENDİRME: {model_name} ===")
    print(f"Giriş metni: {input_text}")
    print("Her benzer metin için 1-5 arası puan verin:")
    print("1: Çok alakasız, 2: Kısmen ilgili, 3: Ortalama, 4: Anlamlı benzer, 5: Çok güçlü benzerlik")

    scores = []
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. Metin: {text[:150]}...")

        # Gerçek kullanımda bu satır yerine şu kullanılmalı:
        # score = int(input(f"Puan (1-5): "))

        # Şimdilik otomatik puanlama (benzer kelime sayısına göre)
        input_words = set(input_text.lower().split())
        text_words = set(text.lower().split())
        overlap = len(input_words.intersection(text_words))
        total_words = len(input_words.union(text_words))

        if total_words > 0:
            similarity_ratio = overlap / total_words
            if similarity_ratio > 0.7:
                score = 5
            elif similarity_ratio > 0.5:
                score = 4
            elif similarity_ratio > 0.3:
                score = 3
            elif similarity_ratio > 0.1:
                score = 2
            else:
                score = 1
        else:
            score = 1

        print(f"Verilen puan: {score}")
        scores.append(score)

    return scores


# === Ana Hesaplama ===
print("\n" + "=" * 80)
print("BENZERLİK HESAPLAMALARI BAŞLIYOR")
print("=" * 80)

results = {}

# TF-IDF modelleri
print("\n### TF-IDF MODELLERİ ###")
tfidf_lemma_indices, tfidf_lemma_scores, tfidf_lemma_texts = tfidf_similarity(
    os.path.join(processed_data_dir, "tfidf_lemmatized.csv"),
    lemmatized, "lemmatized", lemma_col, input_text_lemma
)
results["tfidf_lemmatized"] = (tfidf_lemma_indices, tfidf_lemma_scores, tfidf_lemma_texts)

tfidf_stem_indices, tfidf_stem_scores, tfidf_stem_texts = tfidf_similarity(
    os.path.join(processed_data_dir, "tfidf_stemmed.csv"),
    stemmed, "stemmed", stem_col, input_text_stem
)
results["tfidf_stemmed"] = (tfidf_stem_indices, tfidf_stem_scores, tfidf_stem_texts)

# Word2Vec modelleri - Mevcut model isimlerine göre düzenlendi
print("\n### WORD2VEC MODELLERİ ###")
configs = [
    ("cbow", "window2", "dim100"),
    ("cbow", "window2", "dim300"),
    ("cbow", "window4", "dim100"),
    ("cbow", "window4", "dim300"),
    ("skipgram", "window2", "dim100"),
    ("skipgram", "window2", "dim300"),
    ("skipgram", "window4", "dim100"),
    ("skipgram", "window4", "dim300")
]

for method, window, dim in configs:
    # Lemmatized model - dosya ismini models klasöründeki gerçek isimlerle eşleştir
    name_lemma = f"lemmatized_{method}_{window}_{dim}"
    path_lemma = os.path.join(models_dir, f"{name_lemma}.model")
    indices, scores, texts = w2v_similarity(path_lemma, lemmatized, name_lemma, lemma_col, input_text_lemma)
    results[name_lemma] = (indices, scores, texts)

    # Stemmed model - dosya ismini models klasöründeki gerçek isimlerle eşleştir
    name_stem = f"stemmed_{method}_{window}_{dim}"
    path_stem = os.path.join(models_dir, f"{name_stem}.model")
    indices, scores, texts = w2v_similarity(path_stem, stemmed, name_stem, stem_col, input_text_stem)
    results[name_stem] = (indices, scores, texts)

# === ANLAMSAL DEĞERLENDİRME ===
print("\n" + "=" * 80)
print("ANLAMSAL DEĞERLENDİRME")
print("=" * 80)

semantic_results = {}
for model_name, (indices, scores, texts) in results.items():
    if texts:  # Eğer sonuç varsa
        input_text = input_text_lemma if 'lemmatized' in model_name else input_text_stem
        semantic_scores = get_semantic_scores(model_name, texts, input_text)
        avg_score = np.mean(semantic_scores)
        semantic_results[model_name] = {
            'scores': semantic_scores,
            'average': avg_score
        }
        print(f"\n{model_name} - Ortalama Puan: {avg_score:.2f}")

# === ANLAMSAL DEĞERLENDİRME TABLOSU ===
print("\n" + "=" * 80)
print("ANLAMSAL DEĞERLENDİRME TABLOSU")
print("=" * 80)

semantic_df_data = []
for model_name, result in semantic_results.items():
    row = [model_name] + result['scores'] + [result['average']]
    semantic_df_data.append(row)

if semantic_df_data:
    semantic_df = pd.DataFrame(semantic_df_data,
                               columns=['Model', 'Metin1', 'Metin2', 'Metin3', 'Metin4', 'Metin5', 'Ortalama'])
    print(semantic_df.to_string(index=False))

    # En başarılı modeller
    semantic_df_sorted = semantic_df.sort_values('Ortalama', ascending=False)
    print(f"\nEN BAŞARILI 5 MODEL (Anlamsal Değerlendirme):")
    print(semantic_df_sorted[['Model', 'Ortalama']].head().to_string(index=False))

    # === JACCARD BENZERLİK MATRİSİ ===
    print("\n" + "=" * 80)
    print("JACCARD BENZERLİK MATRİSİ HESAPLANIYOR")
    print("=" * 80)

    model_names = list(results.keys())
    n_models = len(model_names)
    jaccard_matrix = np.zeros((n_models, n_models))

    print(f"Toplam model sayısı: {n_models}")

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            indices1 = results[name1][0]
            indices2 = results[name2][0]
            if indices1 and indices2:
                jaccard_score = jaccard_similarity(indices1, indices2)
                jaccard_matrix[i][j] = jaccard_score
            else:
                jaccard_matrix[i][j] = 0

    # === Jaccard Matris Görselleştirme ===
    plt.figure(figsize=(16, 14))
    df_jaccard = pd.DataFrame(jaccard_matrix, index=model_names, columns=model_names)

    # Heatmap çiz
    sns.heatmap(df_jaccard, annot=True, fmt=".2f", cmap="coolwarm",
                cbar_kws={'label': 'Jaccard Benzerlik Skoru'})
    plt.title(f"Jaccard Benzerlik Matrisi ({n_models}x{n_models})", fontsize=16, fontweight='bold')
    plt.xlabel("Modeller", fontsize=12)
    plt.ylabel("Modeller", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"jaccard_matrix_{n_models}x{n_models}.png"), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n[Jaccard benzerlik matrisi oluşturuldu: {plots_dir}/jaccard_matrix_{n_models}x{n_models}.png]")

    # === JACCARD MATRİSİ ANALİZİ ===
    print("\n" + "=" * 80)
    print("JACCARD MATRİSİ ANALİZİ")
    print("=" * 80)

    # Köşegen dışındaki en yüksek benzerlikler
    jaccard_no_diag = df_jaccard.copy()
    np.fill_diagonal(jaccard_no_diag.values, 0)  # Köşegeni sıfırla

    # En benzer model çiftlerini bul
    max_similarity = 0
    most_similar_pairs = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            similarity = jaccard_matrix[i][j]
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pairs = [(model_names[i], model_names[j], similarity)]
            elif similarity == max_similarity:
                most_similar_pairs.append((model_names[i], model_names[j], similarity))

    print("EN BENZER MODEL ÇİFTLERİ:")
    for model1, model2, sim in most_similar_pairs[:5]:
        print(f"{model1} <-> {model2}: {sim:.3f}")

    # Ortalama benzerlikler
    avg_similarities = []
    for i, model in enumerate(model_names):
        # Kendi dışındaki diğer modellerle ortalama benzerlik
        other_sims = [jaccard_matrix[i][j] for j in range(len(model_names)) if i != j]
        avg_sim = np.mean(other_sims) if other_sims else 0
        avg_similarities.append((model, avg_sim))

    avg_similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\nEN TUTARLI SONUÇ ÜRETEN MODELLER (Ortalama Jaccard Skoru):")
    for model, avg_sim in avg_similarities[:5]:
        print(f"{model}: {avg_sim:.3f}")

    # === ÖZET RAPOR ===
    print("\n" + "=" * 80)
    print("ÖZET RAPOR")
    print("=" * 80)

    print("\n1. ANLAMSAL DEĞERLENDİRME SONUÇLARI:")
    print(
        f"   En başarılı model: {semantic_df_sorted.iloc[0]['Model']} (Ortalama: {semantic_df_sorted.iloc[0]['Ortalama']:.2f})")
    print(
        f"   En düşük performans: {semantic_df_sorted.iloc[-1]['Model']} (Ortalama: {semantic_df_sorted.iloc[-1]['Ortalama']:.2f})")

    # TF-IDF vs Word2Vec karşılaştırması
    tfidf_scores = [semantic_results[m]['average'] for m in semantic_results.keys() if 'tfidf' in m]
    w2v_scores = [semantic_results[m]['average'] for m in semantic_results.keys() if 'tfidf' not in m]

    if tfidf_scores and w2v_scores:
        avg_tfidf = np.mean(tfidf_scores)
        avg_w2v = np.mean(w2v_scores)
        print(f"\n2. TF-IDF vs WORD2VEC KARŞILAŞTIRMASI:")
        print(f"   TF-IDF Ortalama: {avg_tfidf:.2f}")
        print(f"   Word2Vec Ortalama: {avg_w2v:.2f}")
        winner = "TF-IDF" if avg_tfidf > avg_w2v else "Word2Vec"
        print(f"   Başarılı: {winner}")

    print(f"\n3. SIRALAMA TUTARLILIĞI:")
    print(f"   En tutarlı model: {avg_similarities[0][0]} (Jaccard: {avg_similarities[0][1]:.3f})")
    print(
        f"   En benzer model çifti: {most_similar_pairs[0][0]} <-> {most_similar_pairs[0][1]} ({most_similar_pairs[0][2]:.3f})")

    # === SONUÇ TABLOLARINI KAYDET ===
    semantic_df.to_csv(os.path.join(plots_dir, "anlamsal_degerlendirme.csv"), index=False)
    df_jaccard.to_csv(os.path.join(plots_dir, "jaccard_matrix.csv"))

    print(f"\n[Sonuç tabloları kaydedildi: {plots_dir}/]")

else:
    print("UYARI: Hiçbir modelden sonuç alınamadı!")

print("\n" + "=" * 80)
print("ANALİZ TAMAMLANDI!")
print("=" * 80)
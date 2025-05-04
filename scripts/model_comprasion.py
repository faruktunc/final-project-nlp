"""
Model Comparison Script: Lemmatized vs Stemmed Word2Vec Models
For Real Estate Listing Similarity Analysis

This script compares the performance of lemmatized and stemmed Word2Vec models
for finding similar real estate listings.
"""
from similarity_filter_code import RealEstateListingSimilarityFilter
import pandas as pd
import time
from tabulate import tabulate
import os

# Proje dizini
BASE_DIR = "C:\\Users\\omerf\\nlp"
DATA_PATH = os.path.join(BASE_DIR,"data","processed", "processed_real_estate.csv")

# Test parametreleri
SIMILARITY_THRESHOLD = 0.85  # Benzerlik eşiği
DUPLICATE_GROUP_THRESHOLD = 0.92  # Duplikasyon grup eşiği
TEST_INDICES = [5, 10, 15]  # Test edilecek ilan indeksleri
TEST_QUERIES = [
    "A quiet house away from the city where I can have my own private spaces",
    "Modern apartment near the city center with good transportation options",
    "Family home with garden and multiple bedrooms in a safe neighborhood"
]

# Test edilecek modeller
lemmatized_models = [
    "lemmatized_cbow_window2_dim100.model",
    "lemmatized_cbow_window4_dim300.model",
    "lemmatized_skipgram_window2_dim300.model",
    "lemmatized_skipgram_window4_dim300.model"
]

stemmed_models = [
    "stemmed_cbow_window2_dim100.model",
    "stemmed_cbow_window4_dim300.model",
    "stemmed_skipgram_window2_dim300.model",
    "stemmed_skipgram_window4_dim300.model"
]

# Sonuçları depolamak için
results = []


def test_model(model_name, model_type):
    """Belirtilen modeli test eder ve performans metriklerini döndürür"""
    print(f"\n{'-' * 80}")
    print(f"Test ediliyor: {model_name} ({model_type})")
    print(f"{'-' * 80}")

    model_path = os.path.join(BASE_DIR, "models", model_name)

    # Filtreleme sistemini oluştur
    start_time = time.time()
    filter_system = RealEstateListingSimilarityFilter(
        data_path=DATA_PATH,
        model_path=model_path,
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    load_time = time.time() - start_time
    print(f"Model yükleme süresi: {load_time:.2f} saniye")

    # Test 1: İndeks tabanlı benzerlik araması
    index_results = []
    index_time = 0
    for idx in TEST_INDICES:
        start_time = time.time()
        query_info, similar_listings = filter_system.find_similar_listings(query_listing_idx=idx, top_n=5)
        query_time = time.time() - start_time
        index_time += query_time

        print(f"\nİndeks {idx} için sonuçlar:")
        print(f"Sorgu İlanı: {query_info['location']} - {query_info['price']}")
        match_count = len(similar_listings) if not similar_listings.empty else 0
        index_results.append(match_count)

        if not similar_listings.empty:
            for i, (_, row) in enumerate(similar_listings.head(3).iterrows()):
                print(f"{i + 1}. Benzerlik: {row['similarity_score']:.2f}")
                print(f"   Konum: {row['location']}")

    avg_index_matches = sum(index_results) / len(index_results) if index_results else 0
    avg_index_time = index_time / len(TEST_INDICES) if TEST_INDICES else 0

    # Test 2: Metin tabanlı benzerlik araması
    text_results = []
    text_time = 0
    for query in TEST_QUERIES:
        start_time = time.time()
        query_info, similar_listings = filter_system.find_similar_listings(query_description=query, top_n=5)
        query_time = time.time() - start_time
        text_time += query_time

        print(f"\nSorgu için sonuçlar: '{query[:50]}...'")
        print(f"Tokenler: {query_info['tokens'][:10]}...")
        match_count = len(similar_listings) if not similar_listings.empty else 0
        text_results.append(match_count)

        if not similar_listings.empty:
            for i, (_, row) in enumerate(similar_listings.head(3).iterrows()):
                print(f"{i + 1}. Benzerlik: {row['similarity_score']:.2f}")
                print(f"   Konum: {row['location']}")

    avg_text_matches = sum(text_results) / len(text_results) if text_results else 0
    avg_text_time = text_time / len(TEST_QUERIES) if TEST_QUERIES else 0

    # Test 3: Duplikasyon filtreleme
    start_time = time.time()
    groups = filter_system.filter_duplicate_listings(group_threshold=DUPLICATE_GROUP_THRESHOLD)
    duplicate_time = time.time() - start_time
    duplicate_groups = len(groups)

    print(f"\nDuplikasyon filtreleme sonuçları:")
    print(f"Toplam {duplicate_groups} benzer ilan grubu bulundu.")
    if groups:
        for i, group in enumerate(groups[:2]):  # İlk 2 grubu göster
            print(f"\nGrup {i + 1} ({group['count']} ilan):")
            for j, listing in enumerate(group['listings'][:2]):  # Her gruptan ilk 2 ilanı göster
                print(f"  {j + 1}. {listing['location']} - {listing['price']}")

    # Sonuçları kaydet
    return {
        'model_name': model_name,
        'model_type': model_type,
        'load_time': load_time,
        'avg_index_matches': avg_index_matches,
        'avg_index_time': avg_index_time,
        'avg_text_matches': avg_text_matches,
        'avg_text_time': avg_text_time,
        'duplicate_groups': duplicate_groups,
        'duplicate_time': duplicate_time,
        'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
    }


def main():
    """Ana test yürütme fonksiyonu"""
    print("=" * 80)
    print("EMLAK İLANLARI İÇİN BENZERLİK MODELİ KARŞILAŞTIRMA TESTİ")
    print("=" * 80)
    print(f"Veri dosyası: {DATA_PATH}")
    print(f"Benzerlik eşiği: {SIMILARITY_THRESHOLD}")
    print(f"Duplikasyon grup eşiği: {DUPLICATE_GROUP_THRESHOLD}")

    # Lemmatized modelleri test et
    for model in lemmatized_models:
        results.append(test_model(model, "Lemmatized"))

    # Stemmed modelleri test et
    for model in stemmed_models:
        results.append(test_model(model, "Stemmed"))

    # Sonuçları tablo olarak göster
    print("\n" + "=" * 120)
    print("SONUÇLAR ÖZET TABLOSU")
    print("=" * 120)

    headers = [
        "Model", "Tip", "Boyut (MB)", "Yükleme (s)",
        "İndeks Eşleşme", "İndeks Süre (s)",
        "Metin Eşleşme", "Metin Süre (s)",
        "Duplikasyon Grupları", "Duplikasyon Süre (s)"
    ]

    table_data = []
    for r in results:
        table_data.append([
            r['model_name'].replace("_", " ").replace(".model", ""),
            r['model_type'],
            f"{r['model_size_mb']:.1f}",
            f"{r['load_time']:.2f}",
            f"{r['avg_index_matches']:.1f}",
            f"{r['avg_index_time']:.3f}",
            f"{r['avg_text_matches']:.1f}",
            f"{r['avg_text_time']:.3f}",
            r['duplicate_groups'],
            f"{r['duplicate_time']:.2f}"
        ])

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Lemmatized vs Stemmed karşılaştırma analizi
    print("\n" + "=" * 120)
    print("LEMMATIZED VS STEMMED KARŞILAŞTIRMA ANALİZİ")
    print("=" * 120)

    lem_results = [r for r in results if r['model_type'] == "Lemmatized"]
    stem_results = [r for r in results if r['model_type'] == "Stemmed"]

    # Ortalama değerleri hesapla
    lem_avg = {
        'size': sum([r['model_size_mb'] for r in lem_results]) / len(lem_results),
        'load_time': sum([r['load_time'] for r in lem_results]) / len(lem_results),
        'index_matches': sum([r['avg_index_matches'] for r in lem_results]) / len(lem_results),
        'index_time': sum([r['avg_index_time'] for r in lem_results]) / len(lem_results),
        'text_matches': sum([r['avg_text_matches'] for r in lem_results]) / len(lem_results),
        'text_time': sum([r['avg_text_time'] for r in lem_results]) / len(lem_results),
        'duplicate_groups': sum([r['duplicate_groups'] for r in lem_results]) / len(lem_results),
        'duplicate_time': sum([r['duplicate_time'] for r in lem_results]) / len(lem_results)
    }

    stem_avg = {
        'size': sum([r['model_size_mb'] for r in stem_results]) / len(stem_results),
        'load_time': sum([r['load_time'] for r in stem_results]) / len(stem_results),
        'index_matches': sum([r['avg_index_matches'] for r in stem_results]) / len(stem_results),
        'index_time': sum([r['avg_index_time'] for r in stem_results]) / len(stem_results),
        'text_matches': sum([r['avg_text_matches'] for r in stem_results]) / len(stem_results),
        'text_time': sum([r['avg_text_time'] for r in stem_results]) / len(stem_results),
        'duplicate_groups': sum([r['duplicate_groups'] for r in stem_results]) / len(stem_results),
        'duplicate_time': sum([r['duplicate_time'] for r in stem_results]) / len(stem_results)
    }

    # Karşılaştırma tablosu
    comp_headers = ["Metrik", "Lemmatized", "Stemmed", "Fark (%)", "Daha İyi Olan"]
    comp_data = []

    metrics = [
        ("Model Boyutu (MB)", 'size', "Küçük"),
        ("Yükleme Süresi (s)", 'load_time', "Küçük"),
        ("Ortalama İndeks Eşleşme", 'index_matches', "Büyük"),
        ("Ortalama İndeks Sorgu Süresi (s)", 'index_time', "Küçük"),
        ("Ortalama Metin Eşleşme", 'text_matches', "Büyük"),
        ("Ortalama Metin Sorgu Süresi (s)", 'text_time', "Küçük"),
        ("Duplikasyon Grupları", 'duplicate_groups', "Büyük"),
        ("Duplikasyon Süre (s)", 'duplicate_time', "Küçük")
    ]

    for name, key, better in metrics:
        lem_val = lem_avg[key]
        stem_val = stem_avg[key]

        if better == "Küçük":
            diff_pct = ((stem_val - lem_val) / lem_val * 100) if lem_val > 0 else 0
            winner = "Lemmatized" if lem_val < stem_val else "Stemmed" if stem_val < lem_val else "Eşit"
        else:  # "Büyük"
            diff_pct = ((lem_val - stem_val) / stem_val * 100) if stem_val > 0 else 0
            winner = "Lemmatized" if lem_val > stem_val else "Stemmed" if stem_val > lem_val else "Eşit"

        if key in ['duplicate_groups']:
            comp_data.append([
                name,
                f"{lem_val:.1f}",
                f"{stem_val:.1f}",
                f"{abs(diff_pct):.1f}%",
                winner
            ])
        else:
            comp_data.append([
                name,
                f"{lem_val:.3f}",
                f"{stem_val:.3f}",
                f"{abs(diff_pct):.1f}%",
                winner
            ])

    print(tabulate(comp_data, headers=comp_headers, tablefmt="grid"))

    # En iyi modeller analizi
    print("\n" + "=" * 120)
    print("EN İYİ MODELLER")
    print("=" * 120)

    # İndeks sorgusu için en iyi model
    best_index_model = max(results, key=lambda x: x['avg_index_matches'])
    print(f"İndeks sorgusu için en iyi model: {best_index_model['model_name']} ({best_index_model['model_type']})")
    print(f"  Ortalama eşleşme: {best_index_model['avg_index_matches']:.1f}")
    print(f"  Ortalama sorgu süresi: {best_index_model['avg_index_time']:.3f} saniye")

    # Metin sorgusu için en iyi model
    best_text_model = max(results, key=lambda x: x['avg_text_matches'])
    print(f"\nMetin sorgusu için en iyi model: {best_text_model['model_name']} ({best_text_model['model_type']})")
    print(f"  Ortalama eşleşme: {best_text_model['avg_text_matches']:.1f}")
    print(f"  Ortalama sorgu süresi: {best_text_model['avg_text_time']:.3f} saniye")

    # Duplikasyon tespiti için en iyi model
    best_dup_model = max(results, key=lambda x: x['duplicate_groups'])
    print(f"\nDuplikasyon tespiti için en iyi model: {best_dup_model['model_name']} ({best_dup_model['model_type']})")
    print(f"  Tespit edilen grup sayısı: {best_dup_model['duplicate_groups']}")
    print(f"  İşlem süresi: {best_dup_model['duplicate_time']:.2f} saniye")

    # Performans/boyut oranı en iyi model
    for r in results:
        r['perf_size_ratio'] = (r['avg_index_matches'] + r['avg_text_matches']) / r['model_size_mb']

    best_ratio_model = max(results, key=lambda x: x['perf_size_ratio'])
    print(f"\nPerformans/boyut oranı en iyi model: {best_ratio_model['model_name']} ({best_ratio_model['model_type']})")
    print(f"  Performans/boyut oranı: {best_ratio_model['perf_size_ratio']:.3f}")
    print(f"  Model boyutu: {best_ratio_model['model_size_mb']:.1f} MB")


if __name__ == "__main__":
    main()
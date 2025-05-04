# Örnek kullanım dosyası: similarity_filter_example.py
from scripts.similarity_filter_code import RealEstateListingSimilarityFilter

# Proje dizini: C:\Users\omerf\nlp
DATA_PATH = "../data/processed/processed_real_estate.csv"
MODEL_PATH = "../models/lemmatized_skipgram_window4_dim300.model"  # En iyi performans veren modellerden biri

# Filter sistemi oluştur
filter_system = RealEstateListingSimilarityFilter(
    data_path=DATA_PATH,
    model_path=MODEL_PATH,
    similarity_threshold=0.85  # Benzerlik eşiği (daha yüksek = daha katı filtreleme)
)

# Örnek 1: Belirli bir indeksteki ilanın benzerlerini bul
print("\nÖrnek 1: İndeks 5 için benzer ilanlar:")
query_info, similar_listings = filter_system.find_similar_listings(query_listing_idx=5, top_n=3)
print(f"Sorgu İlanı: {query_info['location']} - {query_info['price']}")
if not similar_listings.empty:
    for i, (_, row) in enumerate(similar_listings.iterrows()):
        print(f"{i+1}. Benzerlik: {row['similarity_score']:.2f}")
        print(f"   Konum: {row['location']}")
        print(f"   Fiyat: {row['price']}")
        print(f"   URL: {row['url']}")

# Örnek 2: Metin sorgusu ile benzer ilanları bul
print("\nÖrnek 2: Metin sorgusu ile benzer ilanlar:")
query_text = "A quiet house away from the city where I can have my own private spaces"
query_info, similar_listings = filter_system.find_similar_listings(query_description=query_text, top_n=3)
print(f"Sorgu Metni: {query_text}")
print(f"Tokenler: {query_info['tokens']}")
if not similar_listings.empty:
    for i, (_, row) in enumerate(similar_listings.iterrows()):
        print(f"{i+1}. Benzerlik: {row['similarity_score']:.2f}")
        print(f"   Konum: {row['location']}")
        print(f"   Fiyat: {row['price']}")
        print(f"   URL: {row['url']}")

# Örnek 3: Benzer ilanları grupla (olası duplikasyonları bul)
print("\nÖrnek 3: Benzer ilan grupları (potansiyel duplikasyonlar):")
groups = filter_system.filter_duplicate_listings(group_threshold=0.92)
print(f"Toplam {len(groups)} benzer ilan grubu bulundu.")
for i, group in enumerate(groups[:3]):  # İlk 3 grubu göster
    print(f"\nGrup {i+1} ({group['count']} ilan):")
    for j, listing in enumerate(group['listings'][:2]):  # Her gruptan ilk 2 ilanı göster
        print(f"  {j+1}. {listing['location']} - {listing['price']}")
        print(f"     URL: {listing['url']}")
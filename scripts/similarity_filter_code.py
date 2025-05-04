import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import ast
import argparse


class RealEstateListingSimilarityFilter:
    def __init__(self, data_path, model_path, similarity_threshold=0.9):
        """
        İlan benzerlik filtresi sınıfı.

        Args:
            data_path: Temizlenmiş emlak verilerinin bulunduğu CSV dosyası
            model_path: Eğitilmiş Word2Vec modelinin yolu
            similarity_threshold: Benzerlik eşik değeri (0-1 arası)
        """
        self.similarity_threshold = similarity_threshold
        self.df = pd.read_csv(data_path)
        self.model = Word2Vec.load(model_path)
        print(f"Model yüklendi: {model_path}")
        print(f"Toplam {len(self.df)} ilan bulundu.")

        # Token stringlerini liste olarak parse etme
        self.df['tokens'] = self.df['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    def get_listing_vector(self, tokens):
        """
        Bir ilanın token'larından vektör temsili oluşturur.

        Args:
            tokens: Kelimelerin listesi

        Returns:
            İlanın vektör temsili
        """
        vectors = []
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])

        if not vectors:
            return np.zeros(self.model.vector_size)

        # Tüm kelime vektörlerinin ortalaması
        return np.mean(vectors, axis=0)

    def create_listing_vectors(self):
        """Tüm ilanların vektör temsillerini oluşturur."""
        self.df['vector'] = self.df['tokens'].apply(self.get_listing_vector)

    def find_similar_listings(self, query_listing_idx=None, query_description=None, query_tokens=None, top_n=5):
        """
        Verilen ilana en benzer ilanları bulur.

        Args:
            query_listing_idx: Sorgu yapılacak ilanın indeksi
            query_description: Sorgu metni (tokens yoksa)
            query_tokens: Sorgu için token listesi
            top_n: Dönülecek benzer ilan sayısı

        Returns:
            Benzer ilanların listesi ve benzerlik skorları
        """
        # Vektörler oluşturulmadıysa oluştur
        if 'vector' not in self.df.columns:
            self.create_listing_vectors()

        # Sorgu vektörünü belirle
        if query_listing_idx is not None:
            query_vector = self.df.iloc[query_listing_idx]['vector']
            query_info = self.df.iloc[query_listing_idx]
        elif query_tokens is not None:
            query_vector = self.get_listing_vector(query_tokens)
            query_info = {"tokens": query_tokens}
        elif query_description is not None:
            # Metni tokenize et
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer

            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)

            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))

            # Metni temizle ve tokenize et
            text = query_description.lower()
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(token) for token in tokens if
                      token.isalpha() and token.lower() not in stop_words]

            query_vector = self.get_listing_vector(tokens)
            query_info = {"description": query_description, "tokens": tokens}
        else:
            raise ValueError("İlan indeksi, açıklama veya token listesi verilmelidir.")

        # Tüm vektörleri liste olarak al
        all_vectors = np.array(self.df['vector'].tolist())

        # Benzerlik hesapla
        similarity_scores = cosine_similarity([query_vector], all_vectors)[0]

        # Sonuçları indeks ve skorlarıyla birlikte al
        similarity_results = [(idx, score) for idx, score in enumerate(similarity_scores)]

        # Eğer query_listing_idx verildiyse kendisini sonuçlardan çıkar
        if query_listing_idx is not None:
            similarity_results = [(idx, score) for idx, score in similarity_results if idx != query_listing_idx]

        # Benzerliklere göre sırala (büyükten küçüğe)
        similarity_results.sort(key=lambda x: x[1], reverse=True)

        # Threshold üzerindeki sonuçları filtrele
        filtered_results = [(idx, score) for idx, score in similarity_results if score >= self.similarity_threshold]

        # En iyi top_n sonucu al
        top_results = filtered_results[:top_n]

        # Sonuçları DataFrame olarak döndür
        result_listings = []
        for idx, score in top_results:
            listing = self.df.iloc[idx].copy()
            listing['similarity_score'] = score
            result_listings.append(listing)

        if result_listings:
            result_df = pd.DataFrame(result_listings)
            return query_info, result_df
        else:
            return query_info, pd.DataFrame()

    def filter_duplicate_listings(self, group_threshold=0.9):
        """
        Benzer ilanları gruplar ve muhtemel kopyaları tespit eder.

        Args:
            group_threshold: Gruplamak için benzerlik eşik değeri

        Returns:
            Gruplanmış ilanlar listesi
        """
        # Vektörler oluşturulmadıysa oluştur
        if 'vector' not in self.df.columns:
            self.create_listing_vectors()

        # Benzerlik matrisini hesapla
        all_vectors = np.array(self.df['vector'].tolist())
        similarity_matrix = cosine_similarity(all_vectors)

        # İşlenmiş ilanları takip et
        processed = set()
        groups = []

        # Tüm ilanları dolaş
        for i in range(len(self.df)):
            if i in processed:
                continue

            # Yeni bir grup oluştur
            group = [i]
            processed.add(i)

            # Bu ilana benzer diğer ilanları bul
            for j in range(len(self.df)):
                if j != i and j not in processed and similarity_matrix[i, j] >= group_threshold:
                    group.append(j)
                    processed.add(j)

            # Eğer grupta birden fazla ilan varsa kaydet
            if len(group) > 1:
                groups.append({
                    'group_id': len(groups),
                    'listings': [self.df.iloc[idx].to_dict() for idx in group],
                    'listing_indices': group,
                    'count': len(group)
                })

        return groups



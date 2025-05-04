# Emlak NLP Projesi

## Proje Genel Bakış
Bu proje, emlak ilanları verilerini analiz etmek için Doğal Dil İşleme (NLP) tekniklerini uygulamaktadır. Proje, metin ön işleme, Zipf yasası analizi, TF-IDF vektörleştirme ve emlak açıklamaları üzerinde Word2Vec modellerinin eğitimini içermektedir.
### Hariç bırakılan dosyalar
100 MB boyutunu geçtiği için aşağıdaki dosyalar hariç bırakılmıştır bu dosyaları elde etmek için [TF-IDF vektörleştirme](#5-tf-idf-vektörleştirme) adımına gidiniz. 
```
data/processed/tfidf_lemmatized.csv
data/processed/tfidf_stemmed.csv
```
## Veri Seti Amacı ve Uygulamaları
Bu projede kullanılan emlak veri seti şunlar için kullanılabilir:
- Emlak açıklamalarındaki yaygın kalıpları ve trendleri analiz etme
- Mülk ilanlarının dilsel özelliklerini anlama
- Emlak platformları için anlamsal arama yetenekleri geliştirme
- Mülk açıklamalarına dayalı öneri sistemleri oluşturma
- Metin açıklamalarına dayalı benzer mülkleri tanımlama
- Mülk ilanlarından önemli özellikleri ve olanakları çıkarma

## Kurulum ve Ayarlar

### Ön Koşullar
Sisteminizde Python'un kurulu olduğundan emin olun.

### Ortam Kurulumu
1. Sanal ortam oluşturun:
```bash
python -m venv venv
```

2. Sanal ortamı etkinleştirin:
- Windows:
```bash
venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

3. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

Gerekli kütüphaneler şunlardır:
- nltk: Metin işleme ve tokenizasyon için
- gensim: Word2Vec model uygulaması için
- pandas: Veri manipülasyonu için
- numpy: Sayısal işlemler için
- matplotlib: Görselleştirme için
- scikit-learn: TF-IDF vektörleştirme için

## Adım Adım Model Oluşturma

### 1. Veri Toplama
Proje, emlak ilanlarının web scraping işlemi ile başlar. Veriler https://www.realestate.com.au web sitesinden toplanmıştır:
```bash
python scripts/crawler.py
```

### 2. Veri Birleştirme
Birden fazla JSON dosyasını tek bir veri setinde birleştirin:
```bash
python scripts/jsonCombiner.py
```

### 3. Veri Ön İşleme
Metin verilerini temizleyin ve ön işleyin:
```bash
python scripts/preprocess_real_estate_data.py
```
Bu adım şunları içerir:
- Metin temizleme
- Tokenizasyon
- Durak kelime (stop word) kaldırma
- Lemmatizasyon ve kök bulma (stemming)

### 4. Zipf Yasası Analizi
Kelime frekans dağılımlarını analiz edin:
```bash
python scripts/zipf_analysis.py
```

### 5. TF-IDF Vektörleştirme
Metnin TF-IDF temsillerini oluşturun:
```bash
python scripts/tfidf_vectorizer.py
```

### 6. Word2Vec Model Eğitimi
Farklı parametrelerle çeşitli Word2Vec modellerini eğitin:
```bash
python scripts/train_word2vecmodels.py
```

Eğitim süreci şunları içerir:
- CBOW ve Skip-gram mimarileri
- Farklı pencere boyutları (2 ve 4)
- Çeşitli vektör boyutları (100 ve 300)
- Hem lemmatize edilmiş hem de stemming uygulanmış versiyonlar

### 7. Model Değerlendirme
Eğitilmiş modelleri değerlendirin:
```bash
python scripts/word2vec_evaluation.py
```

### 8. Benzerlik Analizi
Benzerlik fonksiyonlarını ve filtrelemeyi test edin:
```bash
python scripts/similarity_test_code.py
python scripts/similarity_filter_code.py
```

## Proje Yapısı
```
.
├── data/
│   ├── raw/           # Ham JSON veri dosyaları
│   └── processed/     # İşlenmiş CSV dosyaları
├── models/            # Eğitilmiş Word2Vec modelleri
├── plots/             # Oluşturulan grafikler ve görselleştirmeler
├── scripts/           # Python scriptleri
├── venv/              # Sanal ortam
└── visualizations/    # Model Görselleştirmeleri
```




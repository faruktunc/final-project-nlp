import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import time
import random
import json

# ————— AYARLAR —————
base_list_url = "https://www.realestate.com.au/international/tr"
query_string = "?sort=date+desc"
base_url = "https://www.realestate.com.au"
ua = UserAgent()

# Başlangıç/bitiş sayfaları:
START_PAGE = 61   # örnek: 3'ten başla
END_PAGE   = 80 # örnek: 10'da bitir

all_pages_data = {}

for page in range(START_PAGE, END_PAGE + 1):
    # URL oluştur
    if page == 1:
        list_url = f"{base_list_url}{query_string}"
    else:
        list_url = f"{base_list_url}/p{page}{query_string}"

    print(f"\nSayfa {page} çekiliyor: {list_url}")
    headers = {'User-Agent': ua.random}

    try:
        res = requests.get(list_url, headers=headers, timeout=15)
        res.raise_for_status()
    except requests.RequestException as e:
        print(f"Sayfa {page} alınırken hata: {e}. Atlanıyor.")
        continue

    soup = BeautifulSoup(res.text, 'html.parser')
    ilan_bloklari = soup.find_all('div', class_='sc-1dun5hk-0')

    if not ilan_bloklari:
        print(f"Sayfa {page} için ilan bulunamadı, atlanıyor.")
        continue

    page_key = f"page_{page}"
    all_pages_data[page_key] = []

    for i, blok in enumerate(ilan_bloklari, 1):
        try:
            # İlan linki
            a = blok.find('a', href=True)
            href = a['href'] if a else None
            if not href:
                continue
            ilan_url = href if href.startswith('http') else base_url + href

            # Fiyat/konum
            fiyat_tag = blok.find('div', class_='displayConsumerPrice')
            fiyat = fiyat_tag.get_text(strip=True) if fiyat_tag else ""
            konum_tag = blok.find('div', class_='address')
            konum = konum_tag.get_text(strip=True) if konum_tag else ""

            # Detay sayfası isteği
            time.sleep(random.uniform(1.5, 4.0))
            det = requests.get(ilan_url, headers={'User-Agent': ua.random}, timeout=15)
            det.raise_for_status()
            detail_soup = BeautifulSoup(det.text, 'html.parser')

            # Açıklama çekme
            desc_tag = detail_soup.select_one('pre.property-description')
            if desc_tag:
                aciklama = desc_tag.get_text(strip=True)
            else:
                alt = detail_soup.find('div', class_='property-description__content')
                aciklama = alt.get_text("\n", strip=True) if alt else ""

            all_pages_data[page_key].append({
                "url": ilan_url,
                "price": fiyat,
                "location": konum,
                "description": aciklama
            })
            print(f"  İlan {i} kaydedildi.")

        except Exception as e:
            print(f"  İlan {i} işlenirken hata: {e}. Atlanıyor.")
            continue

# Dosya adını başlangıç/bitiş ile oluştur
output_filename = f"tr_scraped_data_{START_PAGE}_{END_PAGE}.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(all_pages_data, f, ensure_ascii=False, indent=2)

print(f"\nİşlem tamamlandı. Veriler '{output_filename}' dosyasına kaydedildi.")

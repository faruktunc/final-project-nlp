import os
import json
import csv

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data", "raw")
output_path = os.path.join(base_dir, "..", "data", "processed", "real_estate_data_2.csv")

json_files = [
    "scraped_data.json",
    "scraped_data_7_40.json",
    "scraped_data_40_70.json",
    "scraped_data_70_100.json",
    "scraped_data_100_130.json",
    "scraped_data_130_160.json",
    "tr_scraped_data_1_2.json",
    "tr_scraped_data_3_20.json",
    "tr_scraped_data_21_40.json",
    "tr_scraped_data_41_60.json",
    "tr_scraped_data_61_80.json",
]

all_entries = []

for filename in json_files:
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        print(f" Dosya yok: {file_path}")
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for page_key in data:
            for item in data[page_key]:
                entry = {
                    "url": item.get("url", ""),
                    "price": item.get("price", ""),
                    "location": item.get("location", ""),
                    "description": item.get("description", "").replace("\n", " ").strip(),
                }
                all_entries.append(entry)

# CSV'ye yaz
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["url", "price", "location", "description"])
    writer.writeheader()
    writer.writerows(all_entries)

print(f" {len(all_entries)} ilan başarıyla yazıldı: {output_path}")

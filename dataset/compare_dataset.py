import pandas as pd
import os
import re

CSV_PATH = "dataset/city_quality/city_traffic.csv"
FOLDER_PATH = "dataset/city_visuals/osm_city_data"


def normalize_city(name):
    name = name.split(",")[0].strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


# =========================
# CSV CITIES
# =========================

df = pd.read_csv(CSV_PATH)

csv_cities = {
    normalize_city(city)
    for city in df["City"].dropna()
}


# =========================
# NON-EMPTY CITY FOLDERS
# =========================

folder_cities = set()

for folder in os.listdir(FOLDER_PATH):
    folder_path = os.path.join(FOLDER_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    # Check whether folder contains anything
    if os.listdir(folder_path):
        folder_cities.add(normalize_city(folder))


# =========================
# COMPARISON
# =========================

intersection = sorted(csv_cities & folder_cities)
csv_minus_folders = sorted(csv_cities - folder_cities)


# =========================
# OUTPUT
# =========================

print("=" * 50)
print(f"CSV cities:              {len(csv_cities)}")
print(f"Non-empty city folders:  {len(folder_cities)}")
print(f"Intersection:             {len(intersection)}")
print(f"CSV - Folders:            {len(csv_minus_folders)}")
print("=" * 50)


print("\nINTERSECTION")
print("-" * 50)

for city in intersection:
    print(city)


print("\nCSV - FOLDERS (missing or empty)")
print("-" * 50)

for city in csv_minus_folders:
    print(city)
# inat_scraper.py
import requests
import pandas as pd
from pathlib import Path
import time

SAVE_DIR = Path("inat_images")
SAVE_DIR.mkdir(exist_ok=True)

def get_inat_observations(taxon_id=47126, max_obs=500):
    all_obs = []
    page = 1
    per_page = 50
    while len(all_obs) < max_obs:
        url = "https://api.inaturalist.org/v1/observations"
        params = {
            "taxon_id": taxon_id,
            "photos": "true",
            "quality_grade": "research",
            "per_page": per_page,
            "page": page
        }
        response = requests.get(url, params=params)
        data = response.json()
        if not data['results']:
            break
        for obs in data['results']:
            if obs.get("photos"):
                all_obs.append({
                    "id": obs["id"],
                    "species": obs["taxon"]["name"] if obs.get("taxon") else "unknown",
                    "photo_url": obs["photos"][0]["url"].replace("square", "original"),
                    "lat": obs["geojson"]["coordinates"][1] if obs.get("geojson") else None,
                    "lon": obs["geojson"]["coordinates"][0] if obs.get("geojson") else None,
                    "date": obs["observed_on"]
                })
        page += 1
        time.sleep(1)
    return all_obs[:max_obs]

def download_image(url, save_path):
    try:
        img = requests.get(url, timeout=10)
        with open(save_path, 'wb') as f:
            f.write(img.content)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    data = get_inat_observations(max_obs=300)
    df = pd.DataFrame(data)
    df.to_csv("inat_metadata.csv", index=False)

    for _, row in df.iterrows():
        image_path = SAVE_DIR / f"{row['id']}.jpg"
        if not image_path.exists():
            download_image(row['photo_url'], image_path)

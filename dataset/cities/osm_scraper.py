"""
OSM city scraper (test batch): pulls MAIN ROADS ONLY (no alleys/service/
residential streets) per city, saves the road network as GraphML and a
rendered raster PNG.

Data source: OpenStreetMap, via the osmnx library (wraps Overpass + Nominatim).

Install:
    pip install osmnx matplotlib

Run:
    python osm_city_scraper.py

Output structure:
    osm_city_data/<city>/<city>_roads.graphml
    osm_city_data/<city>/<city>_roads.png
"""

import os
import time
import osmnx as ox
import matplotlib.pyplot as plt

ox.settings.log_console = True
ox.settings.use_cache = True       # caches raw responses locally, avoids re-hitting API on reruns
ox.settings.timeout = 300          # bigger queries need more time on Overpass

# Swap to a mirror if overpass-api.de is slow/overloaded (uncomment):
# ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api"

from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "osm_city_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Only keep major roads (motorway/trunk/primary/secondary + their link
# ramps). Excludes tertiary, residential, service, alleys, footways,
# tracks, etc. Add "tertiary", "tertiary_link" below to include connector
# roads too.
MAIN_ROAD_TAGS = [
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
]
ROAD_FILTER = f'["highway"~"^({"|".join(MAIN_ROAD_TAGS)})$"]'

# megacities whose OSM admin boundary balloons past the urban core
# (Tokyo's prefecture includes remote islands, Jakarta's includes Kepulauan
# Seribu, Lagos/Mumbai boundaries sprawl) get an explicit bbox around the
# built-up area instead of the full place polygon.
# bbox format: (west, south, east, north)
CITY_CONFIGS = [
    # {"name": "Jakarta",       "bbox": (106.68, -6.37, 106.97, -6.08)},
    # {"name": "Tokyo",         "bbox": (139.56, 35.53, 139.92, 35.82)},
    # {"name": "New York City", "place": "New York City, USA"},
    # {"name": "London",        "place": "Greater London, United Kingdom"},
    # {"name": "Paris",         "place": "Paris, France"},
    # {"name": "Mumbai",        "bbox": (72.77, 18.89, 72.99, 19.30)},
    # {"name": "Lagos",         "bbox": (3.15, 6.39, 3.55, 6.70)},
    # {"name": "Sao Paulo",     "place": "Sao Paulo, Brazil"},
    # {"name": "Berlin",        "place": "Berlin, Germany"},
    # {"name": "Singapore",     "place": "Singapore"},
    # ADD MORE CITIES HERE
]


def safe_name(city: str) -> str:
    return city.strip().lower().replace(" ", "_")


def fetch_graph(config):
    if "bbox" in config:
        return ox.graph_from_bbox(bbox=config["bbox"], custom_filter=ROAD_FILTER)
    return ox.graph_from_place(config["place"], custom_filter=ROAD_FILTER)


def scrape_city(config: dict) -> None:
    city = config["name"]
    name = safe_name(city)
    city_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(city_dir, exist_ok=True)

    print(f"[{city}] fetching main road network...")
    try:
        graph = fetch_graph(config)
        ox.save_graphml(graph, os.path.join(city_dir, f"{name}_roads.graphml"))
    except Exception as e:
        print(f"  road network failed: {e}")
        return

    fig, ax = ox.plot_graph(
        graph, show=False, close=False, node_size=0,
        edge_color="white", edge_linewidth=0.8, bgcolor="black",
    )
    fig.savefig(os.path.join(city_dir, f"{name}_roads.png"), dpi=200, facecolor="black")
    plt.close(fig)
    print(f"[{city}] saved -> {city_dir}")


if __name__ == "__main__":
    for cfg in CITY_CONFIGS:
        scrape_city(cfg)
        time.sleep(1)  # be polite to Overpass/Nominatim, avoid rate limiting
    print("Done.")
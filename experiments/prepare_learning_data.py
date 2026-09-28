"""Create comparable 5 km road images and graph tensors for learned models.

Requires NumPy and Pillow. Run fixed_window.py first, then this script with the
project virtual environment: .venv/Scripts/python experiments/prepare_learning_data.py
"""

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

from project import (MIN_EDGES, MIN_NODES, MODEL_CACHE, RASTER_PIXELS,
                     WINDOW_METRES, artifact, documented_bounds, graph_for_city,
                     model_input, model_manifest)
from road_baseline import (CITY_ALIASES, GRAPHML, ROAD_CLASSES,
                           km_between, load_places, road_class, slug)


WINDOW_KM = WINDOW_METRES / 1000
PIXELS = RASTER_PIXELS


def collect(path, center_lat, center_lon, requested_bounds=None):
    if requested_bounds is None:
        half_lat = WINDOW_KM / 2 / 111.32
        half_lon = WINDOW_KM / 2 / (111.32 * math.cos(math.radians(center_lat)))
        south, north = center_lat - half_lat, center_lat + half_lat
        west, east = center_lon - half_lon, center_lon + half_lon
    else:
        west, south, east, north = requested_bounds
    keys, nodes, edges = {}, {}, []
    graph_element = None
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            if elem.tag == GRAPHML + "graph":
                graph_element = elem
            continue
        if elem.tag == GRAPHML + "key":
            keys[elem.attrib["id"]] = elem.attrib.get("attr.name", "")
        elif elem.tag in (GRAPHML + "node", GRAPHML + "edge"):
            data = {keys.get(child.attrib.get("key", ""), ""): child.text or ""
                    for child in elem if child.tag == GRAPHML + "data"}
            if elem.tag == GRAPHML + "node":
                try:
                    lat, lon = float(data["y"]), float(data["x"])
                    if south <= lat <= north and west <= lon <= east:
                        nodes[elem.attrib["id"]] = (lon, lat)
                except (KeyError, ValueError):
                    pass
            else:
                source, target = elem.attrib.get("source"), elem.attrib.get("target")
                if source in nodes and target in nodes:
                    try:
                        length = float(data.get("length", "0"))
                    except ValueError:
                        length = 0.0
                    edges.append((source, target, data.get("highway", ""),
                                  data.get("oneway", "").lower() in ("true", "1"),
                                  max(0.0, length), data.get("geometry", "")))
            if graph_element is not None:
                graph_element.clear()
    return nodes, edges, (west, south, east, north)


def render_and_encode(nodes, edges, bounds):
    node_ids = sorted(nodes)
    node_index = {value: index for index, value in enumerate(node_ids)}
    west, south, east, north = bounds
    def pixel(lon, lat):
        return ((lon - west) / (east - west) * (PIXELS - 1),
                (north - lat) / (north - south) * (PIXELS - 1))
    images = [Image.new("L", (PIXELS, PIXELS), 0) for _ in ROAD_CLASSES]
    draws = [ImageDraw.Draw(image) for image in images]
    degree = defaultdict(set)
    graph_edges, edge_attrs = [], []
    for source, target, highway, oneway, length, geometry in edges:
        kind = road_class(highway)
        # Preserve the historical fourth-channel fallback for unclassified edges.
        channel = ROAD_CLASSES.index(kind) if kind in ROAD_CLASSES else ROAD_CLASSES.index("secondary")
        if geometry.startswith("LINESTRING (") and geometry.endswith(")"):
            points = []
            for pair in geometry[12:-1].split(","):
                try:
                    lon, lat = (float(value) for value in pair.split()[:2])
                    points.append(pixel(lon, lat))
                except ValueError:
                    pass
        else:
            points = [pixel(*nodes[source]), pixel(*nodes[target])]
        if len(points) >= 2:
            draws[channel].line(points, fill=255, width=1)
        source_index, target_index = node_index[source], node_index[target]
        degree[source].add(target)
        degree[target].add(source)
        graph_edges.append((source_index, target_index))
        edge_attrs.append([*(float(kind == road_kind) for road_kind in ROAD_CLASSES),
                           float(oneway), math.log1p(length / 1000)])
    x = np.array([[2 * (lon - west) / (east - west) - 1,
                   2 * (lat - south) / (north - south) - 1,
                   math.log1p(len(degree[node]))]
                  for node in node_ids for lon, lat in [nodes[node]]], dtype=np.float32)
    return (np.stack([np.asarray(image) for image in images]).astype(np.uint8), x,
            np.array(graph_edges, dtype=np.int64).T,
            np.array(edge_attrs, dtype=np.float32))


def main():
    with artifact("fixed_window_features").open(encoding="utf-8", newline="") as handle:
        eligible = list(csv.DictReader(handle))
    with artifact("road_features").open(encoding="utf-8", newline="") as handle:
        full = {row["city"]: row for row in csv.DictReader(handle)}
    places = load_places()
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    manifest = []
    used_paths = set()
    for index, row in enumerate(eligible, start=1):
        name = slug(row["city"].split(",")[0])
        candidates = places[CITY_ALIASES.get(name, name)]
        graph_lat = float(full[row["city"]]["graph_center_lat"])
        graph_lon = float(full[row["city"]]["graph_center_lon"])
        center = min(candidates, key=lambda p: km_between(graph_lat, graph_lon,
                                                           float(p["lat"]), float(p["lng"])))
        path = graph_for_city(row["city"])
        bounds = documented_bounds(path, float(center["lat"]), float(center["lng"]))
        print(f"Preparing {index}/{len(eligible)}: {row['city']}", flush=True)
        nodes, edges, bounds = collect(path, float(center["lat"]), float(center["lng"]),
                                      bounds)
        image, x, edge_index, edge_attr = render_and_encode(nodes, edges, bounds)
        if len(edge_attr) < MIN_EDGES or len(x) < MIN_NODES:
            raise ValueError(f"Crop unexpectedly empty: {row['city']}")
        output_path = model_input(slug(row["city"]))
        if output_path in used_paths:
            raise ValueError(f"Duplicate normalized city name in model inputs: {row['city']}")
        used_paths.add(output_path)
        np.savez_compressed(output_path, image=image, x=x, edge_index=edge_index,
                            edge_attr=edge_attr, bounds=np.asarray(bounds, dtype=np.float64))
        manifest.append({"city": row["city"], "country": row["country"],
                         "target": float(row["target"]), "file": output_path.name,
                         "nodes": len(x), "edges": edge_index.shape[1]})
    model_manifest().write_text(json.dumps(manifest, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"Prepared {len(manifest)} city images and graphs at {MODEL_CACHE}")


if __name__ == "__main__":
    main()

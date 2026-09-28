"""Compare fixed 5 km city-center road crops with the original full networks.

Run after road_baseline.py: python experiments/fixed_window.py
"""

import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from project import (MIN_EDGES, MIN_NODES, WINDOW_METRES, artifact,
                     documented_bounds, graph_for_city)
from road_baseline import (CITY_ALIASES, FEATURE_NAMES, GRAPHML,
                           ROAD_CLASSES, folds, km_between, load_places, metrics,
                           ridge_predict, road_class, slug)


WINDOW_KM = WINDOW_METRES / 1000
CROP_FEATURES = ("log_nodes", "log_edges", "log_directed_road_km",
                 "log_directed_road_km_per_25_km2", "mean_segment_m",
                 "dead_end_node_fraction", "intersection_node_fraction",
                 "oneway_edge_fraction", "motorway_length_fraction",
                 "trunk_length_fraction", "primary_length_fraction")


def crop_features(path, center_lat, center_lon, requested_bounds=None):
    if requested_bounds is None:
        half_lat = (WINDOW_KM / 2) / 111.32
        half_lon = (WINDOW_KM / 2) / (111.32 * math.cos(math.radians(center_lat)))
        south, north = center_lat - half_lat, center_lat + half_lat
        west, east = center_lon - half_lon, center_lon + half_lon
    else:
        west, south, east, north = requested_bounds
    min_lat = min_lon = float("inf")
    max_lat = max_lon = float("-inf")
    keys, selected, neighbors = {}, set(), defaultdict(set)
    edges = one_way = 0
    length_total = 0.0
    length_by_class = Counter()
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
                    min_lat, max_lat = min(min_lat, lat), max(max_lat, lat)
                    min_lon, max_lon = min(min_lon, lon), max(max_lon, lon)
                    if south <= lat <= north and west <= lon <= east:
                        selected.add(elem.attrib["id"])
                except (KeyError, ValueError):
                    pass
            else:
                source, target = elem.attrib.get("source"), elem.attrib.get("target")
                if source in selected and target in selected:
                    edges += 1
                    neighbors[source].add(target)
                    neighbors[target].add(source)
                    try:
                        length = float(data.get("length", "0"))
                    except ValueError:
                        length = 0.0
                    if math.isfinite(length) and length >= 0:
                        length_total += length
                        length_by_class[road_class(data.get("highway", ""))] += length
                    one_way += data.get("oneway", "").lower() in ("true", "1")
            if graph_element is not None:
                graph_element.clear()
    bbox_covers_window = (requested_bounds is not None or
                          (min_lat <= south and max_lat >= north and
                           min_lon <= west and max_lon >= east))
    if not bbox_covers_window or len(selected) < MIN_NODES or edges < MIN_EDGES or length_total <= 0:
        return None, {"bbox_covers_window": bbox_covers_window,
                      "nodes": len(selected), "edges": edges}
    active_nodes = sum(bool(neighbors[node]) for node in selected)
    dead_ends = sum(len(neighbors[node]) == 1 for node in selected)
    intersections = sum(len(neighbors[node]) >= 3 for node in selected)
    road_km = length_total / 1000
    features = [math.log1p(active_nodes), math.log1p(edges), math.log1p(road_km),
                math.log1p(road_km / (WINDOW_KM ** 2)), length_total / edges,
                dead_ends / len(selected), intersections / len(selected), one_way / edges,
                *(length_by_class[kind] / length_total for kind in ROAD_CLASSES[:3])]
    return features, {"bbox_covers_window": True, "nodes": len(selected), "edges": edges}


def assess(rows, feature_key, grouping):
    actual, predictions = [], []
    for test_indices in folds(rows, grouping):
        test_set = set(test_indices)
        train = [row for index, row in enumerate(rows) if index not in test_set]
        test = [rows[index] for index in test_indices]
        estimates = ridge_predict([row[feature_key] for row in train],
                                  [row["target"] for row in train],
                                  [row[feature_key] for row in test],
                                  feature_count=len(train[0][feature_key]))
        actual.extend(row["target"] for row in test)
        predictions.extend(estimates)
    return metrics(actual, predictions)


def main():
    with artifact("road_features").open(encoding="utf-8", newline="") as handle:
        original = list(csv.DictReader(handle))
    places = load_places()
    rows, excluded = [], []
    documented_requests = 0
    for index, row in enumerate(original, start=1):
        city_slug = slug(row["city"].split(",")[0])
        candidates = places[CITY_ALIASES.get(city_slug, city_slug)]
        graph_lat, graph_lon = float(row["graph_center_lat"]), float(row["graph_center_lon"])
        center = min(candidates, key=lambda p: km_between(graph_lat, graph_lon,
                                                           float(p["lat"]), float(p["lng"])))
        path = graph_for_city(row["city"])
        bounds = documented_bounds(path, float(center["lat"]), float(center["lng"]))
        if bounds is not None:
            documented_requests += 1
        print(f"Cropping graph {index}/{len(original)}: {row['city']}", flush=True)
        crop, info = crop_features(path, float(center["lat"]), float(center["lng"]),
                                   bounds)
        if crop is None:
            excluded.append({"city": row["city"], **info})
            continue
        rows.append({"city": row["city"], "country": row["country"],
                     "target": float(row["target"]), "full": [float(row[name]) for name in FEATURE_NAMES],
                     "crop": crop, "crop_nodes": info["nodes"], "crop_edges": info["edges"]})
    if len(rows) < 30:
        raise RuntimeError(f"Only {len(rows)} cities have usable fixed windows")
    report = {"window_km": WINDOW_KM, "cities": len(rows), "excluded": excluded,
              "graphs_with_documented_query_bounds": documented_requests,
              "feature_names": CROP_FEATURES, "splits": {},
              "caveat": "A graph bounding box or documented query extent does not prove that OSM mapped every road inside it."}
    for grouping in ("city", "country"):
        report["splits"][grouping] = {"original_full_network": assess(rows, "full", grouping),
                                     "fixed_window": assess(rows, "crop", grouping)}
    artifact("fixed_window_report").write_text(json.dumps(report, indent=2) + "\n",
                                                    encoding="utf-8")
    with artifact("fixed_window_features").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["city", "country", "target", "crop_nodes", "crop_edges", *CROP_FEATURES])
        for row in rows:
            writer.writerow([row["city"], row["country"], row["target"], row["crop_nodes"],
                             row["crop_edges"], *row["crop"]])
    print(json.dumps({"cities": len(rows), "excluded_count": len(excluded),
                      "splits": report["splits"]}, indent=2))


if __name__ == "__main__":
    main()

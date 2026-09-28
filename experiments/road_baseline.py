"""First city-level traffic baseline using only Python's standard library.

Run from any directory: python experiments/road_baseline.py

Each city supplies one label. The script resolves names against graph locations
and a city gazetteer, extracts simple features from OSMnx GraphML, and compares a training-mean
predictor with fixed-strength ridge regression. It reports both city-held-out
and country-held-out five-fold results; neither split tunes hyperparameters.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from project import (LABELS, PLACES, QUALITY, ROAD_CLASSES, artifact, graph_paths,
                     slug, source_city_name)


GRAPHML = "{http://graphml.graphdrawing.org/xmlns}"
CITY_ALIASES = {"cebu": "cebu_city"}
COUNTRY_ALIASES = {"czech_republic": "czechia", "hong_kong_china": "hong_kong",
                   "south_korea": "korea_south"}
FEATURE_NAMES = (
    "log_nodes", "log_edges", "log_directed_road_km", "log_bbox_km2",
    "log_directed_road_km_per_bbox_km2", "mean_segment_m",
    "dead_end_node_fraction", "intersection_node_fraction",
    "oneway_edge_fraction", "motorway_length_fraction",
    "trunk_length_fraction", "primary_length_fraction",
)


def canonical_country(value: str) -> str:
    name = slug(value)
    return COUNTRY_ALIASES.get(name, name)


def km_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    a, b = math.radians(lat1), math.radians(lat2)
    dlat, dlon = b - a, math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(h)))


def load_places() -> dict[str, list[dict]]:
    by_name: dict[str, list[dict]] = defaultdict(list)
    with PLACES.open(encoding="utf-8-sig", newline="") as handle:
        for place in csv.DictReader(handle):
            for name in {slug(place["city"]), slug(place["city_ascii"])}:
                by_name[name].append(place)
    return by_name


def resolve_label(city_slug: str, lat: float, lon: float,
                  labels: list[dict], places: dict[str, list[dict]]) -> tuple[dict | None, dict]:
    candidates = places.get(CITY_ALIASES.get(slug(city_slug), slug(city_slug)), [])
    if not candidates:
        return None, {"reason": "no reference city"}
    nearest = min(candidates, key=lambda p: km_between(lat, lon, float(p["lat"]), float(p["lng"])))
    distance = km_between(lat, lon, float(nearest["lat"]), float(nearest["lng"]))
    details = {"reference_city": nearest["city"], "reference_country": nearest["country"],
               "distance_km": round(distance, 1)}
    if distance > QUALITY["maximum_join_distance_km"]:
        return None, {**details, "reason": "graph center too far from reference city"}
    matches = [label for label in labels
               if canonical_country(label["City"].split(",")[-1]) ==
               canonical_country(nearest["country"])]
    if len(matches) != 1:
        return None, {**details, "reason": f"{len(matches)} labels match reference country",
                      "candidate_labels": [label["City"] for label in labels]}
    return matches[0], {**details, "reason": "matched"}


def road_class(value: str) -> str:
    # OSMnx may serialize a list of highway classes as a Python-like string.
    for kind in ROAD_CLASSES:
        if re.search(rf"(?<![a-z]){kind}(?:_link)?(?![a-z_])", value):
            return kind
    return "other"


def graph_features(path: Path) -> tuple[list[float], tuple[float, float], tuple[float, float]]:
    keys: dict[str, str] = {}
    nodes = edges = dead_ends = intersections = oneway_edges = 0
    min_lat = min_lon = float("inf")
    max_lat = max_lon = float("-inf")
    total_length = 0.0
    lengths_by_class: Counter[str] = Counter()
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
                nodes += 1
                try:
                    lat, lon = float(data["y"]), float(data["x"])
                    min_lat, max_lat = min(min_lat, lat), max(max_lat, lat)
                    min_lon, max_lon = min(min_lon, lon), max(max_lon, lon)
                except (KeyError, ValueError):
                    pass
                try:
                    street_count = int(data.get("street_count", "0"))
                    dead_ends += street_count == 1
                    intersections += street_count >= 3
                except ValueError:
                    pass
            else:
                edges += 1
                try:
                    length = float(data.get("length", "0"))
                except ValueError:
                    length = 0.0
                if math.isfinite(length) and length >= 0:
                    total_length += length
                    lengths_by_class[road_class(data.get("highway", ""))] += length
                oneway_edges += data.get("oneway", "").lower() in ("true", "1")
            # Clear processed graph elements so large city networks fit in memory.
            if graph_element is not None:
                graph_element.clear()
    if not nodes or not edges or not total_length or not math.isfinite(min_lat):
        raise ValueError(f"Empty or incomplete road graph: {path}")

    mid_lat = math.radians((min_lat + max_lat) / 2)
    height_km = (max_lat - min_lat) * 111.32
    width_km = (max_lon - min_lon) * 111.32 * math.cos(mid_lat)
    bbox_km2 = max(height_km * width_km, 0.001)
    road_km = total_length / 1000
    features = [
        math.log1p(nodes), math.log1p(edges), math.log1p(road_km),
        math.log1p(bbox_km2), math.log1p(road_km / bbox_km2),
        total_length / edges, dead_ends / nodes, intersections / nodes,
        oneway_edges / edges,
        *(lengths_by_class[kind] / total_length for kind in ROAD_CLASSES[:3]),
    ]
    return features, ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2), (width_km, height_km)


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for the small ridge system."""
    a = [row[:] + [value] for row, value in zip(matrix, vector)]
    n = len(vector)
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        a[col], a[pivot] = a[pivot], a[col]
        if abs(a[col][col]) < 1e-12:
            raise ValueError("Singular ridge system")
        scale = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= scale
        for row in range(col + 1, n):
            scale = a[row][col]
            for j in range(col, n + 1):
                a[row][j] -= scale * a[col][j]
    result = [0.0] * n
    for row in range(n - 1, -1, -1):
        result[row] = a[row][n] - sum(a[row][j] * result[j] for j in range(row + 1, n))
    return result


def ridge_predict(train_x: list[list[float]], train_y: list[float],
                  test_x: list[list[float]], alpha: float = 10.0,
                  feature_count: int | None = None) -> list[float]:
    n_features = feature_count or len(FEATURE_NAMES)
    means = [statistics.mean(row[j] for row in train_x) for j in range(n_features)]
    scales = [statistics.pstdev(row[j] for row in train_x) or 1.0
              for j in range(n_features)]
    x = [[(row[j] - means[j]) / scales[j] for j in range(n_features)]
         for row in train_x]
    target_mean = statistics.mean(train_y)
    gram = [[sum(row[j] * row[k] for row in x) + (alpha if j == k else 0.0)
             for k in range(n_features)] for j in range(n_features)]
    rhs = [sum(row[j] * (y - target_mean) for row, y in zip(x, train_y))
           for j in range(n_features)]
    coefficients = solve(gram, rhs)
    return [target_mean + sum(coefficients[j] * (row[j] - means[j]) / scales[j]
                              for j in range(n_features)) for row in test_x]


def folds(rows: list[dict], grouping: str) -> list[list[int]]:
    result = [[] for _ in range(5)]
    if grouping == "city":
        indices = list(range(len(rows)))
        random.Random(42).shuffle(indices)
        for position, index in enumerate(indices):
            result[position % 5].append(index)
    else:
        by_country: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            by_country[row["country"]].append(index)
        # Largest groups first gives reasonably balanced country-held-out folds.
        for _, indices in sorted(by_country.items(), key=lambda item: (-len(item[1]), item[0])):
            result[min(range(5), key=lambda fold: len(result[fold]))].extend(indices)
    return result


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    errors = [y - p for y, p in zip(actual, predicted)]
    total = sum((y - statistics.mean(actual)) ** 2 for y in actual)
    return {
        "mae": round(statistics.mean(abs(error) for error in errors), 3),
        "rmse": round(math.sqrt(statistics.mean(error * error for error in errors)), 3),
        "r2": round(1 - sum(error * error for error in errors) / total, 3),
    }


def evaluate(rows: list[dict], grouping: str) -> tuple[dict, list[dict]]:
    predictions = []
    for fold_number, test_indices in enumerate(folds(rows, grouping), start=1):
        test_set = set(test_indices)
        train = [row for index, row in enumerate(rows) if index not in test_set]
        test = [rows[index] for index in test_indices]
        baseline = statistics.mean(row["target"] for row in train)
        ridge = ridge_predict([row["features"] for row in train],
                              [row["target"] for row in train],
                              [row["features"] for row in test])
        for row, estimate in zip(test, ridge):
            predictions.append({"split": grouping, "fold": fold_number,
                                "city": row["city"], "country": row["country"],
                                "actual": row["target"], "mean_prediction": baseline,
                                "ridge_prediction": estimate})
    actual = [row["actual"] for row in predictions]
    return ({"mean": metrics(actual, [row["mean_prediction"] for row in predictions]),
             "ridge": metrics(actual, [row["ridge_prediction"] for row in predictions])},
            predictions)


def main() -> None:
    with LABELS.open(encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))
    by_slug: dict[str, list[dict]] = defaultdict(list)
    for label in labels:
        by_slug[slug(label["City"].split(",")[0])].append(label)
    places = load_places()
    paths = graph_paths()
    rows = []
    excluded = []
    for position, path in enumerate(paths, start=1):
        city_slug = slug(source_city_name(path))
        print(f"Reading graph {position}/{len(paths)}: {city_slug}", flush=True)
        try:
            features, (latitude, longitude), (width_km, height_km) = graph_features(path)
            label, join_details = resolve_label(city_slug, latitude, longitude,
                                                by_slug.get(slug(city_slug), []), places)
            if label is None:
                excluded.append({"folder": city_slug, **join_details})
                continue
            target = float(label["Traffic Index"])
            if not all(map(math.isfinite, [*features, target])):
                raise ValueError("Non-finite feature or target")
        except (ET.ParseError, ValueError) as error:
            excluded.append({"folder": city_slug, "reason": str(error)})
            continue
        rows.append({"city": label["City"], "country": label["City"].split(",")[-1].strip(),
                     "target": target, "features": features, "latitude": latitude,
                     "longitude": longitude, "bbox_width_km": width_km,
                     "bbox_height_km": height_km, "join": join_details})
    if len(rows) < 20:
        raise RuntimeError(f"Only {len(rows)} usable cities; inspect the joins and GraphML")

    city_results, city_predictions = evaluate(rows, "city")
    country_results, country_predictions = evaluate(rows, "country")
    report = {"target": "Traffic Index", "cities": len(rows), "features": FEATURE_NAMES,
              "method": "5-fold out-of-fold predictions; ridge alpha=10 fixed; train-only scaling",
              "splits": {"city": city_results, "country": country_results},
              "excluded_graphs": excluded,
              "caveats": [f"Worldcities coordinates validate names and countries only to {QUALITY['maximum_join_distance_km']} km; inspect outliers.",
                          "Bounding-box area is a rough network-extent proxy, not administrative area.",
                          "Observed association is not a causal effect of roads on traffic."]}
    artifact("road_baseline_report").write_text(json.dumps(report, indent=2) + "\n",
                                                        encoding="utf-8")
    with artifact("road_baseline_predictions").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(city_predictions[0]))
        writer.writeheader()
        writer.writerows(city_predictions + country_predictions)
    with artifact("road_features").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["city", "country", "target", "graph_center_lat", "graph_center_lon",
                         "reference_city", "reference_country", "reference_distance_km",
                         "bbox_width_km", "bbox_height_km", *FEATURE_NAMES])
        for row in rows:
            writer.writerow([row["city"], row["country"], row["target"], row["latitude"],
                             row["longitude"], row["join"]["reference_city"],
                             row["join"]["reference_country"], row["join"]["distance_km"],
                             row["bbox_width_km"], row["bbox_height_km"],
                             *row["features"]])
    print(json.dumps({"cities": len(rows), "splits": report["splits"],
                      "excluded_graphs": excluded}, indent=2))


if __name__ == "__main__":
    main()

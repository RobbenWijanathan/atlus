"""Audit road-graph/traffic-label joins against worldcities.csv coordinates.

Run after road_baseline.py. This is an audit, not a definitive geocoder: similar
names, alternative spellings, and different metropolitan extents need review.
"""

import csv
import json
from collections import defaultdict

from project import LABELS, PLACES, QUALITY, artifact, graph_paths, source_city_name
from road_baseline import (CITY_ALIASES,
                           canonical_country, graph_features, km_between, slug)


def main():
    with PLACES.open(encoding="utf-8-sig", newline="") as f:
        reference = list(csv.DictReader(f))
    places = defaultdict(list)
    for place in reference:
        for name in {slug(place["city"]), slug(place["city_ascii"])}:
            places[name].append(place)
    with LABELS.open(encoding="utf-8-sig", newline="") as f:
        labels = list(csv.DictReader(f))
    by_name = defaultdict(list)
    for label in labels:
        by_name[slug(label["City"].split(",")[0])].append(label)
    with artifact("road_features").open(encoding="utf-8", newline="") as f:
        features = {slug(row["city"].split(",")[0]): row for row in csv.DictReader(f)}

    audit = []
    for path in graph_paths():
        name = slug(source_city_name(path))
        if name in features:
            row = features[name]
            lat, lon = float(row["graph_center_lat"]), float(row["graph_center_lon"])
        else:
            _, (lat, lon), _ = graph_features(path)
        candidates = places.get(CITY_ALIASES.get(name, name), [])
        nearest = min(candidates, key=lambda p: km_between(lat, lon, float(p["lat"]),
                                                           float(p["lng"]))) if candidates else None
        distance = (km_between(lat, lon, float(nearest["lat"]), float(nearest["lng"]))
                    if nearest else None)
        country = nearest["country"] if nearest else ""
        label_candidates = by_name.get(name, [])
        # Exact normalized country comparison is conservative; mismatches require review.
        country_matches = [row for row in label_candidates
                           if canonical_country(row["City"].split(",")[-1]) ==
                           canonical_country(country)]
        if nearest is None:
            status = "no_reference_city"
        elif distance > QUALITY["maximum_join_distance_km"]:
            status = "far_from_reference_city"
        elif len(country_matches) == 1:
            status = "country_and_location_match"
        elif not label_candidates:
            status = "no_traffic_label"
        else:
            status = "country_mismatch_or_ambiguous"
        audit.append({"folder": path.parent.name, "graph_center_lat": round(lat, 5),
                      "graph_center_lon": round(lon, 5), "reference_city": nearest["city"] if nearest else "",
                      "reference_admin": nearest["admin_name"] if nearest else "",
                      "reference_country": country, "distance_km": round(distance, 1) if distance is not None else None,
                      "traffic_labels": [r["City"] for r in label_candidates], "status": status})

    artifact("join_audit").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    counts = defaultdict(int)
    for row in audit:
        counts[row["status"]] += 1
    print(dict(counts))
    for row in audit:
        if row["status"] != "country_and_location_match":
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

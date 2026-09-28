"""Summarize what current graph extents prove about the intended 5 km windows.

Run after fixed_window.py and prepare_learning_data.py. An empty image grid
cell is a review flag, never proof that OSM data is missing.
"""

import csv
import json

import numpy as np

from project import MODEL_CACHE, QUALITY, artifact, model_manifest


def main():
    with artifact("road_features").open(encoding="utf-8", newline="") as handle:
        all_cities = list(csv.DictReader(handle))
    with artifact("fixed_window_features").open(encoding="utf-8", newline="") as handle:
        accepted = {row["city"]: row for row in csv.DictReader(handle)}
    excluded_report = json.loads(artifact("fixed_window_report").read_text(encoding="utf-8"))
    excluded = {row["city"]: row for row in excluded_report["excluded"]}
    manifest = json.loads(model_manifest().read_text(encoding="utf-8"))
    cache_by_city = {row["city"]: row for row in manifest}
    records = []
    for row in all_cities:
        city = row["city"]
        record = {"city": city, "country": row["country"],
                  "reference_distance_km": float(row["reference_distance_km"]),
                  "old_bbox_width_km": float(row["bbox_width_km"]),
                  "old_bbox_height_km": float(row["bbox_height_km"])}
        if city in accepted:
            record["old_bbox_surrounds_window"] = True
            record["old_window_nodes"] = int(accepted[city]["crop_nodes"])
            record["old_window_edges"] = int(accepted[city]["crop_edges"])
            with np.load(MODEL_CACHE / cache_by_city[city]["file"]) as data:
                road_mask = data["image"].max(axis=0) > 0
            cell_size = road_mask.shape[0] // QUALITY["image_grid_cells_per_side"]
            empty_cells = sum(not road_mask[y:y + cell_size, x:x + cell_size].any()
                              for y in range(0, road_mask.shape[0], cell_size)
                              for x in range(0, road_mask.shape[1], cell_size))
            record["empty_4x4_image_cells"] = empty_cells
            record["status"] = ("accepted_with_sparse_warning"
                                if empty_cells >= QUALITY["sparse_grid_empty_cells"] else "accepted")
        else:
            failure = excluded[city]
            record["old_bbox_surrounds_window"] = failure["bbox_covers_window"]
            record["old_window_nodes"] = failure["nodes"]
            record["old_window_edges"] = failure["edges"]
            record["empty_4x4_image_cells"] = None
            record["status"] = ("bbox_does_not_surround_window" if not failure["bbox_covers_window"]
                                else "too_few_center_roads")
        records.append(record)
    counts = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    summary = {"cities": len(records), "status_counts": counts,
               "note": "Bounding-box failure is evidence of incomplete geographic extent."
                       " Sparse images can reflect water, geography, major-road filtering,"
                       " original query boundaries or missing OSM data; they require review.",
               "review_first": [record["city"] for record in records
                                if record["status"] in ("bbox_does_not_surround_window",
                                                        "too_few_center_roads")]}
    artifact("existing_coverage_summary").write_text(json.dumps(summary, indent=2) + "\n",
                                                           encoding="utf-8")
    with artifact("existing_coverage_audit").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

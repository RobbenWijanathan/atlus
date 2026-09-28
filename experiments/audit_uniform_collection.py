"""Summarize the separate uniform-window dataset after any collection run."""

import csv
import json

from project import (MIN_EDGES, MIN_NODES, QUALITY, WINDOW_METRES, artifact,
                     slug, uniform_graph_dir)


def main() -> None:
    with artifact("road_features").open(encoding="utf-8", newline="") as handle:
        original = list(csv.DictReader(handle))
    with artifact("existing_coverage_audit").open(encoding="utf-8", newline="") as handle:
        old = {row["city"]: row for row in csv.DictReader(handle)}
    records = []
    manifest = []
    for row in original:
        city = row["city"]
        folder = uniform_graph_dir(slug(city))
        graph_path = folder / "uniform_window__network.graphml"
        meta_path = folder / "uniform_window__metadata.json"
        record = {"city": city, "country": row["country"],
                  "old_status": old[city]["status"],
                  "old_window_nodes": old[city]["old_window_nodes"],
                  "old_window_edges": old[city]["old_window_edges"],
                  "old_graph_to_reference_km": row["reference_distance_km"]}
        if graph_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["city"] != city or meta["window_metres"] != WINDOW_METRES:
                raise ValueError(f"Unexpected metadata for {city}")
            manifest.append(meta)
            record.update({"status": "collected", "new_nodes": meta["nodes"],
                           "new_edges": meta["edges"],
                           "reference_distance_km": meta["original_graph_to_reference_km"],
                           "collected_utc": meta["collected_utc"]})
            reasons = []
            if meta["nodes"] < MIN_NODES or meta["edges"] < MIN_EDGES:
                reasons.append("fewer_than_20_major_road_nodes_or_edges")
            if float(row["reference_distance_km"]) > QUALITY["reference_midpoint_review_km"]:
                reasons.append("reference_far_from_old_graph_midpoint")
            record["review_reason"] = ";".join(reasons)
        else:
            incomplete = graph_path.exists() or meta_path.exists()
            record.update({"status": "incomplete_cache" if incomplete else "missing",
                           "new_nodes": "", "new_edges": "",
                           "reference_distance_km": "", "collected_utc": "",
                           "review_reason": ("incomplete_graph_metadata_pair" if incomplete
                                             else "not_collected") +
                           (";reference_far_from_old_graph_midpoint"
                            if float(row["reference_distance_km"]) > QUALITY["reference_midpoint_review_km"]
                            else "")})
        records.append(record)
    fieldnames = list(records[0])
    with artifact("uniform_collection_audit_csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    run_report = artifact("uniform_collection_report")
    report = json.loads(run_report.read_text(encoding="utf-8")) if run_report.exists() else {}
    artifact("uniform_collection_manifest").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {"eligible_cities": len(records),
               "collected": sum(r["status"] == "collected" for r in records),
               "missing": [r["city"] for r in records if r["status"] == "missing"],
               "incomplete_cached_pairs": [r["city"] for r in records
                                           if r["status"] == "incomplete_cache"],
               "sparse_new_windows": [r["city"] for r in records
                                      if "fewer_than_20_major_road_nodes_or_edges" in r["review_reason"]],
               "reference_center_review": [r["city"] for r in records
                                           if "reference_far_from_old_graph_midpoint" in r["review_reason"]],
               "last_run_failures": [r for r in report.get("results", []) if r["status"] == "failed"],
               "last_run_requested": report.get("requested_cities"),
               "last_run_processed": len(report.get("results", [])),
               "last_run_complete": report.get("run_complete", len(report.get("results", [])) == report.get("requested_cities")),
               "last_run_successful": report.get("run_successful"),
               "last_run_stopped_reason": report.get("stopped_reason"),
               "interpretation": "Requested extraction extent is documented; sparse roads and OSM completeness need separate review."}
    artifact("uniform_collection_audit_json").write_text(json.dumps(summary, indent=2) + "\n",
                                                            encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

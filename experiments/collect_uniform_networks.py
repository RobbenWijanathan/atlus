"""Fetch identical-size, city-centred major-road windows from OpenStreetMap.

This creates a separate, resumable dataset. It never replaces the original
graphs. A successful query documents its requested extent, not OSM completeness.

Examples:
  python experiments/collect_uniform_networks.py --city "Muscat, Oman" --dry-run
  python experiments/collect_uniform_networks.py --city "Muscat, Oman"
  python experiments/collect_uniform_networks.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from datetime import datetime, timezone

import osmnx as ox
from pyproj import Geod
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, Timeout as RequestsTimeout

from project import (CONFIG, HISTORY_ROOT, OSMNX_CACHE, ROAD_FILTER, ROOT,
                     WINDOW_METRES, artifact, uniform_graph_dir, write_json)
from road_baseline import CITY_ALIASES, km_between, load_places, slug


GEOD = Geod(ellps="WGS84")
COLLECTION = CONFIG["collection"]
if (COLLECTION["maximum_attempts_per_city"] < 1 or
        COLLECTION["maximum_consecutive_transient_failures"] < 1 or
        COLLECTION["retry_initial_delay_seconds"] < 0 or
        COLLECTION["retry_max_delay_seconds"] < COLLECTION["retry_initial_delay_seconds"]):
    raise ValueError("Invalid collection retry settings in config.json")


def transient_failure(error: Exception) -> bool:
    """Retry transport failures and overloaded servers, not empty road queries."""
    if isinstance(error, PermissionError) or "[WinError 10013]" in str(error):
        return False
    if isinstance(error, (RequestsConnectionError, RequestsTimeout,
                          ConnectionError, TimeoutError)):
        return True
    if isinstance(error, HTTPError) and error.response is not None:
        return error.response.status_code in (429, 500, 502, 503, 504)
    return False


def retry_delay(attempt: int) -> float:
    return min(float(COLLECTION["retry_max_delay_seconds"]),
               float(COLLECTION["retry_initial_delay_seconds"]) * 2 ** (attempt - 1))


def record_attempt(result: dict, item: dict, attempt: int, transient: bool,
                   network_request: bool, started_utc: str) -> None:
    with artifact("uniform_collection_attempts").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**result, "attempt": attempt,
                                 "transient_failure": transient,
                                 "network_request": network_request,
                                 "started_utc": started_utc,
                                 "completed_utc": datetime.now(timezone.utc).isoformat(),
                                 "overpass_url": item["overpass_url"]}) + "\n")


def selected_rows(names: list[str], all_cities: bool) -> list[dict]:
    with artifact("road_features").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if all_cities:
        matches = rows
    else:
        requested = set(names)
        matches = [row for row in rows if row["city"] in requested]
        missing = requested - {row["city"] for row in matches}
        if missing:
            raise ValueError(f"Unknown city labels: {sorted(missing)}")
    if not matches:
        raise ValueError("Provide --city at least once or --all")
    slugs = [slug(row["city"]) for row in matches]
    if len(slugs) != len(set(slugs)):
        raise ValueError("Eligible city labels collide after normalization")
    return matches


def plan(row: dict, places: dict) -> dict:
    city_slug = slug(row["city"].split(",")[0])
    candidates = places[CITY_ALIASES.get(city_slug, city_slug)]
    graph_lat = float(row["graph_center_lat"])
    graph_lon = float(row["graph_center_lon"])
    center = min(candidates, key=lambda place: km_between(
        graph_lat, graph_lon, float(place["lat"]), float(place["lng"])))
    lat, lon = float(center["lat"]), float(center["lng"])
    west, _, _ = GEOD.fwd(lon, lat, 270, WINDOW_METRES / 2)
    east, _, _ = GEOD.fwd(lon, lat, 90, WINDOW_METRES / 2)
    _, south, _ = GEOD.fwd(lon, lat, 180, WINDOW_METRES / 2)
    _, north, _ = GEOD.fwd(lon, lat, 0, WINDOW_METRES / 2)
    if west >= east:
        raise ValueError(f"Antimeridian-crossing window requires special handling: {row['city']}")
    return {"city": row["city"], "country": row["country"],
            "reference_city": center["city"], "reference_country": center["country"],
            "center_lat": lat, "center_lon": lon,
            "original_graph_to_reference_km": round(km_between(graph_lat, graph_lon, lat, lon), 3),
            "bbox_west_south_east_north": [west, south, east, north],
            "window_metres": WINDOW_METRES, "highway_filter": ROAD_FILTER,
            "network_type": CONFIG["collection"]["network_type"],
            "retain_all": CONFIG["collection"]["retain_all"],
            "truncate_by_edge": CONFIG["collection"]["truncate_by_edge"],
            "simplify": CONFIG["collection"]["simplify"],
            "osmnx_version": ox.__version__}


def fetch(item: dict, overwrite: bool) -> dict:
    stem = slug(item["city"])
    folder = uniform_graph_dir(stem)
    graph_path = folder / "uniform_window__network.graphml"
    metadata_path = folder / "uniform_window__metadata.json"
    if graph_path.exists() != metadata_path.exists() and not overwrite:
        raise ValueError(f"Incomplete cached graph/metadata pair for {item['city']}; review before --overwrite")
    if graph_path.exists() and metadata_path.exists() and not overwrite:
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        query_keys = ("city", "country", "reference_city", "reference_country",
                      "center_lat", "center_lon", "bbox_west_south_east_north",
                      "window_metres", "highway_filter", "network_type",
                      "retain_all", "truncate_by_edge", "simplify")
        if all(saved.get(key) == item[key] for key in query_keys):
            return {"city": item["city"], "status": "cached", "nodes": saved["nodes"],
                    "edges": saved["edges"], "graph": str(graph_path.relative_to(ROOT))}
        raise ValueError(f"Existing query parameters differ for {item['city']}; review before --overwrite")
    graph = ox.graph_from_bbox(tuple(item["bbox_west_south_east_north"]),
                               network_type=item["network_type"],
                               custom_filter=item["highway_filter"],
                               simplify=item["simplify"],
                               retain_all=item["retain_all"],
                               truncate_by_edge=item["truncate_by_edge"])
    node_count, edge_count = graph.number_of_nodes(), graph.number_of_edges()
    if not node_count or not edge_count:
        raise ValueError("No major-road nodes or edges returned")
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "uniform_window__network.graphml.partial"
    try:
        ox.io.save_graphml(graph, filepath=temporary)
        if overwrite and (graph_path.exists() or metadata_path.exists()):
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            history = HISTORY_ROOT / "uniform_windows" / stem / timestamp
            history.mkdir(parents=True, exist_ok=False)
            for old_path in (graph_path, metadata_path):
                if old_path.exists():
                    shutil.copy2(old_path, history / old_path.name)
        os.replace(temporary, graph_path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {**item, "nodes": node_count, "edges": edge_count,
                "collected_utc": datetime.now(timezone.utc).isoformat(),
                "data_source": "OpenStreetMap via OSMnx/Overpass",
                "interpretation": "The requested bounding box is known; OSM completeness is unverified."}
    write_json(metadata_path, metadata)
    return {"city": item["city"], "status": "downloaded", "nodes": node_count,
            "edges": edge_count, "graph": str(graph_path.relative_to(ROOT))}


def collect_item(item: dict, overwrite: bool, minimum_delay: float,
                 last_query: float | None) -> tuple[dict, float | None, bool]:
    """Collect one city with bounded retries; return result, query time, outage flag."""
    graph_path = uniform_graph_dir(slug(item["city"])) / "uniform_window__network.graphml"
    needs_query = overwrite or not graph_path.exists()
    maximum = int(COLLECTION["maximum_attempts_per_city"]) if needs_query else 1
    for attempt in range(1, maximum + 1):
        if needs_query:
            if last_query is not None:
                time.sleep(max(0, minimum_delay - (time.monotonic() - last_query)))
            last_query = time.monotonic()
        started_utc = datetime.now(timezone.utc).isoformat()
        try:
            result = fetch(item, overwrite)
        except Exception as error:
            transient = needs_query and transient_failure(error)
            result = {"city": item["city"], "status": "failed",
                      "error": f"{type(error).__name__}: {error}"}
        else:
            transient = False
        if result["status"] != "cached":
            record_attempt(result, item, attempt, transient, needs_query, started_utc)
        if result["status"] != "failed" or not transient or attempt == maximum:
            return result, last_query, transient
        delay = retry_delay(attempt)
        print(f"Retrying {item['city']} after {delay:g}s: {result['error']}", flush=True)
        time.sleep(delay)
    raise AssertionError("Unreachable retry state")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", action="append", default=[], help="Exact city label in road_features.csv")
    parser.add_argument("--all", action="store_true", help="Collect all eligible cities")
    parser.add_argument("--review-first", action="store_true",
                        help="Collect cities flagged by the old-window audit")
    parser.add_argument("--dry-run", action="store_true", help="Show requests without network access")
    parser.add_argument("--overwrite", action="store_true", help="Replace this separate dataset's saved city")
    parser.add_argument("--delay", type=float,
                        default=float(CONFIG["collection"]["minimum_delay_seconds"]),
                        help="Minimum seconds between queries")
    parser.add_argument("--overpass-url", default=CONFIG["collection"]["overpass_url"],
                        help="Overpass API base URL, such as https://overpass.osm.jp/api")
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be nonnegative")
    requested = list(args.city)
    if args.review_first:
        summary = json.loads(artifact("existing_coverage_summary").read_text(encoding="utf-8"))
        requested.extend(summary["review_first"])
    rows = selected_rows(requested, args.all)
    places = load_places()
    planned = [{**plan(row, places), "overpass_url": args.overpass_url} for row in rows]
    if args.dry_run:
        print(json.dumps(planned, indent=2))
        return
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(OSMNX_CACHE)
    ox.settings.requests_timeout = int(CONFIG["collection"]["requests_timeout_seconds"])
    ox.settings.overpass_url = args.overpass_url
    results = []
    last_query = None
    consecutive_transient_failures = 0
    stopped_reason = None

    def save_report(complete: bool) -> None:
        successful = sum(row["status"] in ("downloaded", "cached") for row in results)
        write_json(artifact("uniform_collection_report"), {
            "window_metres": WINDOW_METRES,
            "requested_cities": len(planned),
            "processed_cities": len(results),
            "run_complete": complete,
            "successful_cities": successful,
            "run_successful": complete and successful == len(results),
            "overpass_url": args.overpass_url,
            "stopped_reason": stopped_reason,
            "results": results,
        })

    save_report(False)
    for index, item in enumerate(planned, start=1):
        print(f"[{index}/{len(planned)}] {item['city']}", flush=True)
        try:
            result, last_query, transient = collect_item(item, args.overwrite, args.delay, last_query)
        except KeyboardInterrupt:
            stopped_reason = "interrupted"
            save_report(False)
            raise
        results.append(result)
        save_report(False)
        print(json.dumps(result), flush=True)
        if result["status"] == "cached":
            continue
        consecutive_transient_failures = (consecutive_transient_failures + 1
                                          if transient else 0)
        if consecutive_transient_failures >= COLLECTION["maximum_consecutive_transient_failures"]:
            stopped_reason = f"{consecutive_transient_failures} consecutive transient city failures"
            print(f"Stopping collection: {stopped_reason}", flush=True)
            break
    save_report(stopped_reason is None)


if __name__ == "__main__":
    main()

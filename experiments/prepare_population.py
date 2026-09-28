"""Download GHSL population tiles and align residential density with road windows.

One city remains one observation. Source archives and derived inputs use the
configured artifact root; rerunning resumes verified source downloads.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import time
import zipfile
from contextlib import ExitStack

import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from project import (CONFIG, POPULATION_CACHE, RASTER_PIXELS, WINDOW_METRES,
                     artifact, documented_bounds, graph_for_city,
                     population_input, population_tile, write_json)
from road_baseline import CITY_ALIASES, km_between, load_places, slug


POP = CONFIG["population"]
DATASET = f"GHS_POP_E{POP['epoch']}_GLOBE_R2023A_{POP['resolution']}"
SOURCE_VERSION = f"{DATASET}_V1_0"
FEATURE_NAMES = ("log_mean_residents_per_km2", "log_peak_residents_per_km2",
                 "fraction_area_above_density_threshold")


def tile_indices(bounds):
    west, south, east, north = bounds
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(f"Unsupported population bounds: {bounds}")
    # GHSL's actual GeoTIFF grid is offset from integer degree boundaries.
    # The origin below is measured from the official source raster metadata.
    top = POP["tile_north_edge_lat"]
    left = POP["tile_west_edge_lon"]
    step = POP["tile_size_degrees"]
    rows = range(math.floor((top - north) / step) + 1,
                 math.floor((top - math.nextafter(south, math.inf)) / step) + 2)
    cols = range(math.floor((west - left) / step) + 1,
                 math.floor((math.nextafter(east, -math.inf) - left) / step) + 2)
    return [(row, col) for row in rows for col in cols]


def source_url(row, col):
    return (f"{POP['base_url']}/{DATASET}/V1-0/tiles/"
            f"{SOURCE_VERSION}_R{row}_C{col}.zip")


def valid_archive(path):
    try:
        with zipfile.ZipFile(path) as archive:
            tiffs = [name for name in archive.namelist() if name.lower().endswith(".tif")]
            return len(tiffs) == 1 and archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def ensure_tile(row, col):
    path = population_tile(row, col)
    if path.exists() and valid_archive(path):
        return path
    if path.exists():
        raise ValueError(f"Corrupt cached source tile; inspect before replacing: {path}")
    url = source_url(row, col)
    partial = path.with_name(path.name + ".partial")
    for attempt in range(1, POP["maximum_attempts"] + 1):
        try:
            with requests.get(url, stream=True, timeout=POP["requests_timeout_seconds"]) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            output.write(chunk)
            if not valid_archive(partial):
                raise ValueError(f"Invalid downloaded GHSL archive: {url}")
            os.replace(partial, path)
            return path
        except (requests.RequestException, ValueError, OSError):
            if partial.exists():
                partial.unlink()
            if attempt == POP["maximum_attempts"]:
                raise
            time.sleep(min(2 ** attempt, 10))


def source_dataset(archive_path, stack):
    with zipfile.ZipFile(archive_path) as archive:
        name = next(name for name in archive.namelist() if name.lower().endswith(".tif"))
        memory = stack.enter_context(MemoryFile(archive.read(name)))
    dataset = stack.enter_context(memory.open())
    if dataset.crs != rasterio.CRS.from_epsg(4326) or dataset.count != 1:
        raise ValueError(f"Unexpected GHSL raster format: {archive_path}")
    return dataset


def density_grid(bounds):
    """Return density in people/km² and audited source counts for a window."""
    size = RASTER_PIXELS
    destination = np.full((size, size), np.nan, dtype=np.float32)
    target_transform = from_bounds(*bounds, size, size)
    window_population = 0.0
    source_cells = 0
    source_peak = 0.0
    covered_cell_weight = 0.0
    dense_cell_weight = 0.0
    sources = []
    for row, col in tile_indices(bounds):
        path = ensure_tile(row, col)
        sources.append({"tile": f"R{row}_C{col}", "url": source_url(row, col),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        with ExitStack() as stack:
            dataset = source_dataset(path, stack)
            west, south, east, north = bounds
            window = rasterio.windows.from_bounds(west, south, east, north,
                                                   transform=dataset.transform)
            first_col = math.floor(window.col_off) - 2
            first_row = math.floor(window.row_off) - 2
            last_col = math.ceil(window.col_off + window.width) + 2
            last_row = math.ceil(window.row_off + window.height) + 2
            window = Window(first_col, first_row, last_col - first_col,
                            last_row - first_row).intersection(
                                Window(0, 0, dataset.width, dataset.height))
            values = dataset.read(1, window=window).astype(np.float32)
            transform = dataset.window_transform(window)
            lat_north = transform.f + np.arange(values.shape[0]) * transform.e
            lat_south = lat_north + transform.e
            lon_west = transform.c + np.arange(values.shape[1]) * transform.a
            lon_east = lon_west + transform.a
            latitudes = (lat_north + lat_south) / 2
            cell_area = (111.32 ** 2 * abs(transform.a * transform.e) *
                         np.cos(np.deg2rad(latitudes)))[:, None]
            valid = np.isfinite(values) & (values >= 0)
            density = np.where(valid, values / cell_area, np.nan).astype(np.float32)
            latitude_fraction = (np.maximum(0, np.minimum(lat_north, north) -
                                            np.maximum(lat_south, south)) / abs(transform.e))
            longitude_fraction = (np.maximum(0, np.minimum(lon_east, east) -
                                             np.maximum(lon_west, west)) / abs(transform.a))
            weights = latitude_fraction[:, None] * longitude_fraction[None, :]
            included = (weights > 0) & valid
            selected = density[included]
            window_population += float((values[included] * weights[included]).sum())
            source_cells += len(selected)
            covered_cell_weight += float(weights[included].sum())
            if len(selected):
                source_peak = max(source_peak, float(selected.max()))
                dense_cell_weight += float(weights[included][
                    selected >= POP["high_density_threshold_per_km2"]].sum())
            reproject(source=density, destination=destination,
                      src_transform=transform, src_crs=dataset.crs,
                      src_nodata=np.nan, dst_transform=target_transform,
                      dst_crs=dataset.crs, dst_nodata=np.nan,
                      init_dest_nodata=False, resampling=Resampling.bilinear)
    coverage = float(np.isfinite(destination).mean())
    if coverage < 0.95 or source_cells < 5:
        raise ValueError(f"Insufficient GHSL coverage: {coverage:.3f}, {source_cells} cells")
    destination = np.nan_to_num(destination, nan=0, posinf=0, neginf=0)
    destination = np.maximum(destination, 0)
    features = [math.log1p(window_population / ((WINDOW_METRES / 1000) ** 2)),
                math.log1p(source_peak), dense_cell_weight / covered_cell_weight]
    image = (np.log1p(destination) /
             math.log1p(POP["image_density_reference_per_km2"])).astype(np.float32)
    return image, features, {"coverage_fraction": coverage, "source_cells": source_cells,
                             "window_population_estimate": round(window_population, 2),
                             "source_tiles": sources}


def city_bounds(row, places):
    name = slug(row["city"].split(",")[0])
    candidates = places[CITY_ALIASES.get(name, name)]
    graph_lat, graph_lon = float(row["graph_center_lat"]), float(row["graph_center_lon"])
    center = min(candidates, key=lambda place: km_between(graph_lat, graph_lon,
                 float(place["lat"]), float(place["lng"])))
    lat, lon = float(center["lat"]), float(center["lng"])
    requested = documented_bounds(graph_for_city(row["city"]), lat, lon)
    if requested is not None:
        return requested
    half_lat = WINDOW_METRES / 2000 / 111.32
    half_lon = WINDOW_METRES / 2000 / (111.32 * math.cos(math.radians(lat)))
    return [lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat]


def main():
    with artifact("fixed_window_features").open(encoding="utf-8", newline="") as handle:
        eligible = list(csv.DictReader(handle))
    with artifact("road_features").open(encoding="utf-8", newline="") as handle:
        roads = {row["city"]: row for row in csv.DictReader(handle)}
    places = load_places()
    records, audit, failed = [], [], []
    for index, city in enumerate(eligible, 1):
        print(f"Population {index}/{len(eligible)}: {city['city']}", flush=True)
        try:
            bounds = city_bounds(roads[city["city"]], places)
            image, features, details = density_grid(bounds)
            np.savez_compressed(population_input(slug(city["city"])), density=image,
                                bounds=np.asarray(bounds, dtype=np.float64),
                                epoch=np.asarray(POP["epoch"]),
                                resolution=np.asarray(POP["resolution"]))
            records.append({"city": city["city"], "country": city["country"],
                            "target": city["target"],
                            **dict(zip(FEATURE_NAMES, features))})
            audit.append({"city": city["city"], "bounds": bounds, **details})
        except (ValueError, requests.RequestException, OSError) as exc:
            failed.append({"city": city["city"], "error": str(exc)})
    with artifact("population_features").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["city", "country", "target", *FEATURE_NAMES])
        writer.writeheader()
        writer.writerows(records)
    report = {"source": POP["source"], "epoch": POP["epoch"],
              "resolution": POP["resolution"], "population_unit": "residents per source cell",
              "density_unit": "residents per square kilometre", "input_cities": len(eligible),
              "high_density_threshold_per_km2": POP["high_density_threshold_per_km2"],
              "prepared_cities": len(records), "failed": failed, "audit": audit,
              "caveat": "Residential population is estimated; the traffic label's date and geographic footprint are unknown. Existing 5 km road windows have unverified road completeness."}
    write_json(artifact("population_report"), report)
    if failed:
        raise RuntimeError(f"Population inputs incomplete: {len(failed)} of {len(eligible)} cities; see report")


if __name__ == "__main__":
    main()

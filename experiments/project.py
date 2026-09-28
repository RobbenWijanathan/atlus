"""Shared configuration and artifact locations for the Atlus pipeline.

Generated filenames use ``stage__role`` or ``stage__variant__role``. All
generated files live under one configured artifact root.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = Path(os.environ.get("ATLUS_CONFIG", Path(__file__).with_name("config.json"))).resolve()
CONFIG = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
QUALITY = CONFIG["quality_checks"]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def slug(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


GRAPHS = project_path(CONFIG["inputs"]["graphs"])
LABELS = project_path(CONFIG["inputs"]["traffic_labels"])
PLACES = project_path(CONFIG["inputs"]["city_reference"])
ARTIFACT_ROOT = project_path(CONFIG["artifacts_root"])
if not ARTIFACT_ROOT.resolve().is_relative_to(ROOT.resolve()):
    raise ValueError("artifacts_root must stay within the project workspace")
MODEL_CACHE = ARTIFACT_ROOT / "cache/model_inputs"
POPULATION_CACHE = ARTIFACT_ROOT / "cache/population"
UNIFORM_CACHE = ARTIFACT_ROOT / "cache/uniform_windows"
OSMNX_CACHE = ARTIFACT_ROOT / "cache/osmnx"
HISTORY_ROOT = ARTIFACT_ROOT / "history"
WINDOW_METRES = int(CONFIG["road_window"]["width_metres"])
RASTER_PIXELS = int(CONFIG["road_window"]["raster_pixels"])
MIN_NODES = int(CONFIG["road_window"]["minimum_nodes"])
MIN_EDGES = int(CONFIG["road_window"]["minimum_edges"])
HIGHWAY_TAGS = tuple(CONFIG["road_window"]["highway_tags"])
ROAD_CLASSES = tuple(tag for tag in HIGHWAY_TAGS if not tag.endswith("_link"))
if WINDOW_METRES <= 0 or RASTER_PIXELS < 16 or MIN_NODES < 1 or MIN_EDGES < 1:
    raise ValueError("Invalid road_window settings in config.json")
if not ROAD_CLASSES or len(set(HIGHWAY_TAGS)) != len(HIGHWAY_TAGS):
    raise ValueError("Invalid or duplicate highway_tags in config.json")
if ROAD_CLASSES[:3] != ("motorway", "trunk", "primary"):
    raise ValueError("Current statistical features require motorway, trunk, primary in that order")
if ROAD_CLASSES != ("motorway", "trunk", "primary", "secondary"):
    raise ValueError("Current four-channel models require motorway, trunk, primary, secondary")
if (QUALITY["image_grid_cells_per_side"] < 1 or
        RASTER_PIXELS % QUALITY["image_grid_cells_per_side"] != 0):
    raise ValueError("image_grid_cells_per_side must divide raster_pixels")
ROAD_FILTER = '["highway"~"^(' + "|".join(HIGHWAY_TAGS) + ')$"]'


_ARTIFACTS = {
    "road_features": ("tables", "road_baseline__features.csv"),
    "road_baseline_report": ("reports", "road_baseline__report.json"),
    "road_baseline_predictions": ("predictions", "road_baseline__predictions.csv"),
    "join_audit": ("reports", "join__audit.json"),
    "sample_size_report": ("reports", "sample_size__report.json"),
    "fixed_window_features": ("tables", "fixed_window__features.csv"),
    "fixed_window_report": ("reports", "fixed_window__report.json"),
    "road_crop_preview": ("figures", "road_crop__jakarta.png"),
    "learned_model_report": ("reports", "learned_fixed_epochs__report.json"),
    "learned_model_predictions": ("predictions", "learned_fixed_epochs__predictions.csv"),
    "model_comparison": ("reports", "model_comparison__report.json"),
    "small_rotation_audit": ("reports", "small_rotation__audit.json"),
    "small_rotation_preview": ("figures", "small_rotation__jakarta.png"),
    "existing_coverage_audit": ("tables", "existing_coverage__audit.csv"),
    "existing_coverage_summary": ("reports", "existing_coverage__summary.json"),
    "uniform_collection_report": ("reports", "uniform_collection__run.json"),
    "uniform_collection_audit_csv": ("tables", "uniform_collection__audit.csv"),
    "uniform_collection_audit_json": ("reports", "uniform_collection__audit.json"),
    "uniform_collection_manifest": ("reports", "uniform_collection__manifest.json"),
    "uniform_collection_attempts": ("reports", "uniform_collection__attempts.jsonl"),
    "pipeline_runs": ("reports", "pipeline__runs.jsonl"),
    "population_features": ("tables", "population__features.csv"),
    "population_report": ("reports", "population__report.json"),
    "population_evaluation_report": ("reports", "population_evaluation__report.json"),
    "population_predictions": ("predictions", "population__predictions.csv"),
}


def artifact(key: str, *, variant: str | None = None) -> Path:
    """Return a registered generated-file path, creating its parent folder."""
    if key in ("nested_report", "nested_predictions"):
        if variant is None:
            variant = "plain"
        if not re.fullmatch(r"[a-z][a-z0-9_]*", variant):
            raise ValueError(f"Invalid artifact variant: {variant}")
        bucket = "reports" if key == "nested_report" else "predictions"
        role = "report.json" if key == "nested_report" else "predictions.csv"
        path = ARTIFACT_ROOT / bucket / f"learned_nested__{variant}__{role}"
    else:
        if variant is not None:
            raise ValueError(f"Artifact {key} does not take a variant")
        bucket, name = _ARTIFACTS[key]
        path = ARTIFACT_ROOT / bucket / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def model_manifest() -> Path:
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    return MODEL_CACHE / "model_inputs__manifest.json"


def population_input(name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"Invalid population-input name: {name}")
    path = POPULATION_CACHE / "inputs" / f"population__{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def population_tile(row: int, col: int) -> Path:
    if not (1 <= row <= 18 and 1 <= col <= 36):
        raise ValueError("Invalid GHSL tile index")
    name = (f"ghs_pop_e{CONFIG['population']['epoch']}_globe_r2023a_"
            f"{CONFIG['population']['resolution']}_v1_0_r{row}_c{col}.zip")
    path = POPULATION_CACHE / "source" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = POPULATION_CACHE / "source" / f"ghsl_pop_e{CONFIG['population']['epoch']}_r{row}_c{col}.zip"
    if not path.exists() and legacy.exists() and CONFIG["population"]["resolution"] == "4326_30ss":
        legacy.replace(path)
    return path


def model_input(name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"Invalid model-input name: {name}")
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    return MODEL_CACHE / f"model_inputs__{name}.npz"


def uniform_graph_dir(city_slug: str) -> Path:
    if not re.fullmatch(r"[a-z0-9_]+", city_slug):
        raise ValueError(f"Invalid city slug: {city_slug}")
    return UNIFORM_CACHE / city_slug


@lru_cache(maxsize=1)
def graph_paths() -> tuple[Path, ...]:
    """Find one GraphML source per city directory, independent of filename."""
    if not GRAPHS.exists():
        raise FileNotFoundError(f"Configured graph directory does not exist: {GRAPHS}")
    paths = []
    for folder in sorted(path for path in GRAPHS.iterdir() if path.is_dir()):
        files = sorted(folder.glob("*.graphml"))
        if len(files) > 1:
            raise ValueError(f"Ambiguous GraphML sources in {folder}")
        paths.extend(files)
    return tuple(paths)


def source_city_name(path: Path) -> str:
    """Use explicit metadata when present; otherwise use the source folder."""
    metadata = graph_metadata(path)
    if metadata is not None:
        return metadata["city"].split(",")[0]
    return path.parent.name


def graph_metadata(path: Path) -> dict | None:
    metadata = path.parent / "uniform_window__metadata.json"
    return json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else None


def documented_bounds(path: Path, center_lat: float, center_lon: float) -> list[float] | None:
    """Return a matching recorded query box, if this graph has provenance."""
    metadata = graph_metadata(path)
    if metadata is None:
        return None
    if (metadata["window_metres"] != WINDOW_METRES or
            abs(metadata["center_lat"] - center_lat) > 1e-6 or
            abs(metadata["center_lon"] - center_lon) > 1e-6):
        raise ValueError(f"Query metadata conflicts with configured window or center: {path}")
    return metadata["bbox_west_south_east_north"]


def graph_for_city(city_label: str) -> Path:
    """Resolve a saved city label to exactly one graph source."""
    full = slug(city_label)
    short = slug(city_label.split(",")[0])
    exact = [path for path in graph_paths() if slug(path.parent.name) == full]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"Multiple exact graph folders for {city_label}")
    matches = [path for path in graph_paths() if slug(source_city_name(path)) == short]
    if len(matches) != 1:
        raise ValueError(f"Expected one graph for {city_label}, found {len(matches)}")
    return matches[0]


def figure(stage: str, name: str) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", stage) or not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("Figure stage and name must be snake_case")
    path = ARTIFACT_ROOT / "figures" / f"{stage}__{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value: object) -> None:
    """Atomically replace a JSON artifact after its complete write succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


# Legacy names are used only for the idempotent migration in pipeline.py.
LEGACY_FILES = {
    "road_features.csv": "road_features",
    "road_baseline_report.json": "road_baseline_report",
    "road_baseline_predictions.csv": "road_baseline_predictions",
    "join_audit.json": "join_audit",
    "sample_size_report.json": "sample_size_report",
    "fixed_window_features.csv": "fixed_window_features",
    "fixed_window_report.json": "fixed_window_report",
    "jakarta_crop_preview.png": "road_crop_preview",
    "learned_model_report.json": "learned_model_report",
    "learned_model_predictions.csv": "learned_model_predictions",
    "model_comparison.json": "model_comparison",
    "small_rotation_audit.json": "small_rotation_audit",
    "jakarta_small_rotation_preview.png": "small_rotation_preview",
    "existing_coverage_audit.csv": "existing_coverage_audit",
    "existing_coverage_summary.json": "existing_coverage_summary",
    "uniform_collection_report.json": "uniform_collection_report",
    "uniform_collection_audit.csv": "uniform_collection_audit_csv",
    "uniform_collection_audit.json": "uniform_collection_audit_json",
    "uniform_collection_manifest.json": "uniform_collection_manifest",
    "learned_model_nested_report.json": ("nested_report", "plain"),
    "learned_model_nested_predictions.csv": ("nested_predictions", "plain"),
    "learned_model_nested_augmented_cnn_report.json": ("nested_report", "right_angle_flip"),
    "learned_model_nested_augmented_cnn_predictions.csv": ("nested_predictions", "right_angle_flip"),
    "learned_model_nested_small_angle_cnn_report.json": ("nested_report", "small_angle"),
    "learned_model_nested_small_angle_cnn_predictions.csv": ("nested_predictions", "small_angle"),
    "learned_model_nested_small_angle_flip_cnn_report.json": ("nested_report", "small_angle_flip"),
    "learned_model_nested_small_angle_flip_cnn_predictions.csv": ("nested_predictions", "small_angle_flip"),
}

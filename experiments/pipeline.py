"""Single entry point for migration, preparation, evaluation, and collection.

Use ``python experiments/pipeline.py list`` to inspect stages and
``python experiments/pipeline.py run --target coverage --dry-run`` to preview.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

from project import (ARTIFACT_ROOT, CONFIG, CONFIG_FILE, HISTORY_ROOT, MODEL_CACHE, OSMNX_CACHE,
                     ROOT, UNIFORM_CACHE, LEGACY_FILES, artifact, model_input,
                     model_manifest, slug, uniform_graph_dir)


STAGES = {
    "baseline": ((), "road_baseline.py", []),
    "join_audit": (("baseline",), "audit_joins.py", []),
    "sample_size": (("baseline",), "check_sample_size.py", []),
    "fixed_window": (("baseline",), "fixed_window.py", []),
    "prepare_models": (("fixed_window",), "prepare_learning_data.py", []),
    "train_fixed": (("prepare_models",), "train_learned.py",
                    ["--splits", *CONFIG["evaluation"]["fixed_split_units"],
                     "--models", *CONFIG["evaluation"]["fixed_models"],
                     "--epochs", str(CONFIG["evaluation"]["fixed_epochs"])]),
    "train_nested_plain": (("prepare_models",), "train_with_inner_validation.py", []),
    "train_nested_right_angle_flip": (("prepare_models",), "train_with_inner_validation.py", ["--augmentation", "right_angle_flip", "--models", "cnn"]),
    "train_nested_small_angle": (("prepare_models",), "train_with_inner_validation.py", ["--augmentation", "small_angle", "--models", "cnn"]),
    "train_nested_small_angle_flip": (("prepare_models",), "train_with_inner_validation.py", ["--augmentation", "small_angle_flip", "--models", "cnn"]),
    "rotation_audit": (("prepare_models",), "audit_small_rotation.py", []),
    "model_comparison": (("train_nested_plain", "train_nested_right_angle_flip",
                          "train_nested_small_angle", "train_nested_small_angle_flip"),
                         "compare_models.py", []),
    "existing_coverage": (("prepare_models",), "audit_existing_coverage.py", []),
    "collect_uniform": (("existing_coverage",), "collect_uniform_networks.py", []),
    "audit_uniform": (("baseline", "existing_coverage"), "audit_uniform_collection.py", []),
    "prepare_population": (("fixed_window",), "prepare_population.py", []),
    "evaluate_population": (("prepare_population",), "evaluate_population.py", []),
    "train_nested_population": (("prepare_models", "prepare_population"),
                                "train_with_inner_validation.py", ["--population"]),
}

TARGETS = {
    "baseline": ("baseline", "join_audit", "sample_size"),
    "coverage": ("existing_coverage",),
    "evaluation": ("train_fixed", "model_comparison", "rotation_audit", "existing_coverage"),
    "collection": ("collect_uniform", "audit_uniform"),
    "collection_audit": ("audit_uniform",),
    "population": ("evaluate_population", "train_nested_population"),
}

STAGE_OUTPUTS = {
    "baseline": ("road_features", "road_baseline_report", "road_baseline_predictions"),
    "join_audit": ("join_audit",),
    "sample_size": ("sample_size_report",),
    "fixed_window": ("fixed_window_features", "fixed_window_report"),
    "train_fixed": ("learned_model_report", "learned_model_predictions"),
    "train_nested_plain": (("nested_report", "plain"), ("nested_predictions", "plain")),
    "train_nested_right_angle_flip": (("nested_report", "right_angle_flip"),
                                      ("nested_predictions", "right_angle_flip")),
    "train_nested_small_angle": (("nested_report", "small_angle"),
                                 ("nested_predictions", "small_angle")),
    "train_nested_small_angle_flip": (("nested_report", "small_angle_flip"),
                                      ("nested_predictions", "small_angle_flip")),
    "rotation_audit": ("small_rotation_audit",),
    "model_comparison": ("model_comparison",),
    "existing_coverage": ("existing_coverage_audit", "existing_coverage_summary"),
    "collect_uniform": ("uniform_collection_report",),
    "audit_uniform": ("uniform_collection_audit_csv", "uniform_collection_audit_json",
                      "uniform_collection_manifest"),
    "prepare_population": ("population_features", "population_report"),
    "evaluate_population": ("population_evaluation_report", "population_predictions"),
    "train_nested_population": (("nested_report", "population"),
                                ("nested_predictions", "population")),
}


def ordered_stages(target: str) -> list[str]:
    ordered, visited = [], set()

    def visit(stage: str) -> None:
        if stage in visited:
            return
        for dependency in STAGES[stage][0]:
            visit(dependency)
        visited.add(stage)
        ordered.append(stage)

    for stage in TARGETS.get(target, (target,)):
        visit(stage)
    return ordered


def within_workspace(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Migration path is outside the workspace: {resolved}")
    return resolved


def same_file(source: Path, destination: Path) -> bool:
    if source.stat().st_size != destination.stat().st_size:
        return False
    def digest(path: Path) -> str:
        hash_value = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hash_value.update(chunk)
        return hash_value.hexdigest()
    return digest(source) == digest(destination)


def migrate_file(source: Path, destination: Path) -> bool:
    """Move one legacy artifact safely; return whether a file was moved."""
    source, destination = within_workspace(source), within_workspace(destination)
    if not source.exists():
        return False
    if not source.is_file():
        raise ValueError(f"Expected a file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not same_file(source, destination):
            raise ValueError(f"Conflicting artifacts; review before migration: {source} and {destination}")
        source.unlink()
        return False
    shutil.move(str(source), str(destination))
    return True


def prune_empty(directory: Path) -> None:
    directory = within_workspace(directory)
    if not directory.exists():
        return
    for child in sorted((p for p in directory.rglob("*") if p.is_dir()),
                        key=lambda p: len(p.parts), reverse=True):
        if not any(child.iterdir()):
            within_workspace(child).rmdir()
    if not any(directory.iterdir()):
        directory.rmdir()


def migrate() -> dict:
    """Move known historical results and caches to the configured artifact root."""
    old_results = within_workspace(ROOT / "experiments/results")
    old_cache = within_workspace(ROOT / "experiments/cache")
    if old_results.exists():
        unknown = sorted(p.name for p in old_results.iterdir() if p.is_file()
                         and p.name not in LEGACY_FILES)
        if unknown:
            raise ValueError(f"Unregistered legacy result files: {unknown}")
    moved = {"results": 0, "model_inputs": 0, "uniform_windows": 0, "osmnx_cache": 0}
    if old_results.exists():
        for old_name, key in LEGACY_FILES.items():
            destination = (artifact(key[0], variant=key[1])
                           if isinstance(key, tuple) else artifact(key))
            moved["results"] += migrate_file(old_results / old_name, destination)
    if old_cache.exists():
        old_manifest = old_cache / "manifest.json"
        manifest = json.loads(old_manifest.read_text(encoding="utf-8")) if old_manifest.exists() else None
        old_to_new = {}
        if manifest is not None:
            for row in manifest:
                new_name = model_input(slug(row["city"])).name
                old_to_new[row["file"]] = new_name
        for source in sorted(old_cache.rglob("*")):
            if not source.is_file() or source == old_manifest:
                continue
            relative = source.relative_to(old_cache)
            if relative.parts[0] == "osmnx":
                destination = OSMNX_CACHE.joinpath(*relative.parts[1:])
                bucket = "osmnx_cache"
            elif relative.parts[0] == "uniform5km":
                city = relative.parts[1]
                filename = {"roads.graphml": "uniform_window__network.graphml",
                            "metadata.json": "uniform_window__metadata.json"}.get(relative.name, relative.name)
                destination = uniform_graph_dir(city).joinpath(*relative.parts[2:-1], filename)
                bucket = "uniform_windows"
            elif len(relative.parts) == 1 and source.suffix == ".npz":
                destination = MODEL_CACHE / old_to_new.get(source.name, f"model_inputs__{source.stem}.npz")
                bucket = "model_inputs"
            else:
                raise ValueError(f"Unrecognized legacy cache file: {source}")
            moved[bucket] += migrate_file(source, destination)
        if manifest is not None:
            for row in manifest:
                row["file"] = old_to_new[row["file"]]
            destination = model_manifest()
            if destination.exists():
                current = json.loads(destination.read_text(encoding="utf-8"))
                if current != manifest:
                    raise ValueError(f"Conflicting model-input manifests: {old_manifest} and {destination}")
                old_manifest.unlink()
            else:
                destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                old_manifest.unlink()
            moved["model_inputs"] += 1
    run_report = artifact("uniform_collection_report")
    if run_report.exists():
        report = json.loads(run_report.read_text(encoding="utf-8"))
        changed = False
        for result in report.get("results", []):
            if "graph" in result:
                new_path = uniform_graph_dir(slug(result["city"])) / "uniform_window__network.graphml"
                relative = str(new_path.relative_to(ROOT))
                if result["graph"] != relative:
                    result["graph"] = relative
                    changed = True
        if changed:
            run_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    prune_empty(old_results)
    prune_empty(old_cache)
    return moved


def snapshot_outputs(stage: str, run_id: str) -> int:
    """Keep the prior report version before a stage replaces its outputs."""
    copied = 0
    for key in STAGE_OUTPUTS.get(stage, ()):
        source = (artifact(key[0], variant=key[1]) if isinstance(key, tuple)
                  else artifact(key))
        if not source.exists():
            continue
        relative = source.relative_to(ARTIFACT_ROOT)
        destination = within_workspace(HISTORY_ROOT / "pipeline_runs" / run_id / relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List targets and stages")
    commands.add_parser("migrate", help="Move historical artifacts into the central artifact root")
    run = commands.add_parser("run", help="Run a target and all its dependencies")
    run.add_argument("--target", choices=tuple(TARGETS) + tuple(STAGES), required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--collection-scope", choices=("review-first", "all"), default="review-first")
    run.add_argument("--city", action="append", default=[],
                     help="Collect only these exact city labels; repeat for a controlled batch")
    run.add_argument("--collection-overwrite", action="store_true",
                     help="Re-query existing uniform windows after changing collection settings")
    run.add_argument("--overpass-url", help="Override the configured Overpass API base URL")
    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps({"targets": TARGETS, "stages": {name: value[0] for name, value in STAGES.items()}}, indent=2))
        return
    if args.command == "migrate":
        print(json.dumps(migrate(), indent=2))
        return
    stages = ordered_stages(args.target)
    if "collect_uniform" not in stages and (args.collection_overwrite or args.overpass_url or
                                            args.collection_scope != "review-first" or args.city):
        parser.error("Collection options require a target that runs collect_uniform")
    if args.city and args.collection_scope != "review-first":
        parser.error("Use --city alone or --collection-scope all")
    if args.dry_run:
        print(json.dumps({"config": str(CONFIG_FILE), "artifacts_root": str(ARTIFACT_ROOT),
                          "stages": stages, "collection_scope": args.collection_scope,
                          "cities": args.city,
                          "collection_overwrite": args.collection_overwrite,
                          "overpass_url": args.overpass_url or CONFIG["collection"]["overpass_url"]}, indent=2))
        return
    migrate()
    environment = os.environ.copy()
    environment["ATLUS_CONFIG"] = str(CONFIG_FILE)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    config_hash = hashlib.sha256(CONFIG_FILE.read_bytes()).hexdigest()
    def log_event(event: str, **details) -> None:
        record = {"run_id": run_id, "time_utc": datetime.now(timezone.utc).isoformat(),
                  "target": args.target, "event": event,
                  "config_sha256": config_hash, **details}
        with artifact("pipeline_runs").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    log_event("started", stages=stages)
    for stage in stages:
        _, script, parameters = STAGES[stage]
        command = [sys.executable, str(Path(__file__).with_name(script)), *parameters]
        if stage == "collect_uniform":
            if args.city:
                for city in args.city:
                    command += ["--city", city]
            else:
                command.append("--all" if args.collection_scope == "all" else "--review-first")
            if args.collection_overwrite:
                command.append("--overwrite")
            if args.overpass_url:
                command += ["--overpass-url", args.overpass_url]
        copied = snapshot_outputs(stage, run_id)
        if copied:
            log_event("prior_outputs_saved", stage=stage, files=copied)
        print(f"[{stage}] {' '.join(command)}", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
        except KeyboardInterrupt:
            log_event("interrupted", stage=stage)
            raise
        except subprocess.CalledProcessError as exc:
            log_event("failed", stage=stage, exit_code=exc.returncode)
            raise
        log_event("stage_complete", stage=stage)
    if "collect_uniform" in stages:
        report = json.loads(artifact("uniform_collection_report").read_text(encoding="utf-8"))
        if not report.get("run_successful", False):
            failures = [row["city"] for row in report["results"] if row["status"] == "failed"]
            log_event("incomplete", failed_cities=failures)
            raise RuntimeError(f"Uniform collection incomplete for {len(failures)} requested cities; see collection report")
    log_event("completed")


if __name__ == "__main__":
    main()

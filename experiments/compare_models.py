"""Compare country-held-out predictions on the same eligible cities.

Country bootstrap intervals describe uncertainty in paired MAE differences.
They are exploratory: models and preprocessing were developed on this dataset.
"""

import csv
import json
import math
import random
import statistics
from collections import defaultdict

from project import artifact
from road_baseline import FEATURE_NAMES, folds, metrics, ridge_predict


def load_predictions(path, rows, expected_models):
    """Reject stale, partial or duplicated held-out predictions."""
    expected = {row["city"]: row for row in rows}
    expected_folds = {rows[index]["city"]: fold
                      for fold, indices in enumerate(folds(rows, "country"), start=1)
                      for index in indices}
    predictions = {name: {} for name in expected_models}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model, city = row["model"], row["city"]
            if model not in predictions or city not in expected:
                raise ValueError(f"Unexpected model or city in {path}: {model}, {city}")
            if city in predictions[model]:
                raise ValueError(f"Duplicate prediction in {path}: {model}, {city}")
            actual = float(row["actual"])
            predicted = float(row["predicted"])
            if (row["country"] != expected[city]["country"] or
                    not math.isclose(actual, expected[city]["target"], abs_tol=1e-6) or
                    int(row["fold"]) != expected_folds[city] or
                    not math.isfinite(predicted)):
                raise ValueError(f"Prediction provenance differs from current evaluation rows in {path}: {city}")
            predictions[model][city] = predicted
    for model, by_city in predictions.items():
        missing = expected.keys() - by_city.keys()
        if missing:
            raise ValueError(f"Missing {len(missing)} {model} predictions in {path}")
    return predictions


def main():
    with artifact("road_features").open(encoding="utf-8", newline="") as handle:
        full = {row["city"]: row for row in csv.DictReader(handle)}
    with artifact("fixed_window_features").open(encoding="utf-8", newline="") as handle:
        cropped = list(csv.DictReader(handle))
    rows = [{"city": row["city"], "country": row["country"],
             "target": float(row["target"]),
             "extent": [float(full[row["city"]][name]) for name in FEATURE_NAMES[:4]]}
            for row in cropped]
    prediction_by_city = {"mean": {}, "extent_ridge": {}}
    for test_indices in folds(rows, "country"):
        held_out = set(test_indices)
        train = [row for index, row in enumerate(rows) if index not in held_out]
        test = [rows[index] for index in test_indices]
        baseline = statistics.mean(row["target"] for row in train)
        extent = ridge_predict([row["extent"] for row in train],
                               [row["target"] for row in train],
                               [row["extent"] for row in test], feature_count=4)
        for row, estimate in zip(test, extent):
            prediction_by_city["mean"][row["city"]] = baseline
            prediction_by_city["extent_ridge"][row["city"]] = estimate
    prediction_by_city.update(load_predictions(
        artifact("nested_predictions", variant="plain"), rows, ("cnn", "gnn")))
    augmented = artifact("nested_predictions", variant="right_angle_flip")
    if augmented.exists():
        prediction_by_city["cnn_rotflip"] = load_predictions(augmented, rows, ("cnn",))["cnn"]
    for mode in ("small_angle", "small_angle_flip"):
        path = artifact("nested_predictions", variant=mode)
        if path.exists():
            prediction_by_city[f"cnn_{mode}"] = load_predictions(path, rows, ("cnn",))["cnn"]
    names = list(prediction_by_city)
    report = {"cities": len(rows), "held_out_unit": "country", "models": {}, "differences": {},
              "bootstrap": "5000 resamples of countries with replacement; percentile 95% intervals",
              "caveat": "Exploratory interval conditional on this dataset, trained models and preprocessing."}
    for name in names:
        report["models"][name] = metrics([row["target"] for row in rows],
                                         [prediction_by_city[name][row["city"]] for row in rows])
    by_country = defaultdict(list)
    for row in rows:
        by_country[row["country"]].append(row)
    countries = list(by_country)
    rng = random.Random(20260928)
    comparisons = [(candidate, reference) for candidate in ("gnn", "cnn")
                   for reference in ("mean", "extent_ridge")]
    if "cnn_rotflip" in names:
        comparisons.extend(("cnn_rotflip", reference)
                           for reference in ("cnn", "mean", "extent_ridge"))
    for mode in ("small_angle", "small_angle_flip"):
        name = f"cnn_{mode}"
        if name in names:
            comparisons.extend((name, reference)
                               for reference in ("cnn", "cnn_rotflip", "mean"))
    for candidate, reference in comparisons:
            errors = []
            for _ in range(5000):
                sampled = [row for _ in countries
                           for country in [rng.choice(countries)] for row in by_country[country]]
                difference = statistics.mean(
                    abs(row["target"] - prediction_by_city[candidate][row["city"]]) -
                    abs(row["target"] - prediction_by_city[reference][row["city"]])
                    for row in sampled)
                errors.append(difference)
            errors.sort()
            report["differences"][f"{candidate}_minus_{reference}_mae"] = {
                "observed": round(report["models"][candidate]["mae"] -
                                  report["models"][reference]["mae"], 3),
                "lower_95": round(errors[125], 3), "upper_95": round(errors[4874], 3)}
    artifact("model_comparison").write_text(json.dumps(report, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Coverage ablation and training-size diagnostic for the traffic baseline.

Run after road_baseline.py: python experiments/check_sample_size.py
"""

import csv
import json
import random
import statistics

from project import artifact
from road_baseline import FEATURE_NAMES, folds, metrics, ridge_predict


def load_rows():
    with artifact("road_features").open(encoding="utf-8", newline="") as handle:
        return [{"city": row["city"], "country": row["country"],
                 "target": float(row["target"]),
                 "features": [float(row[name]) for name in FEATURE_NAMES]}
                for row in csv.DictReader(handle)]


def evaluate(rows, grouping, columns, train_size=None, seed=42):
    actual, predicted = [], []
    for fold_number, test_indices in enumerate(folds(rows, grouping)):
        test_set = set(test_indices)
        train_indices = [index for index in range(len(rows)) if index not in test_set]
        if train_size is not None:
            train_indices = random.Random(seed + fold_number).sample(train_indices,
                                                                      min(train_size, len(train_indices)))
        train = [rows[index] for index in train_indices]
        test = [rows[index] for index in test_indices]
        estimates = ridge_predict([[row["features"][j] for j in columns] for row in train],
                                  [row["target"] for row in train],
                                  [[row["features"][j] for j in columns] for row in test],
                                  feature_count=len(columns))
        actual.extend(row["target"] for row in test)
        predicted.extend(estimates)
    return metrics(actual, predicted)


def main():
    rows = load_rows()
    variants = {"all_features": list(range(len(FEATURE_NAMES))),
                "extent_only": [0, 1, 2, 3],
                "normalized_structure": list(range(4, len(FEATURE_NAMES)))}
    report = {"cities": len(rows), "features": FEATURE_NAMES, "ablations": {},
              "learning_curve": {},
              "notes": ["Fixed model and fixed held-out folds; these are diagnostics, not tuned test scores.",
                        "City counts are training cities, not road tiles or graph nodes.",
                        "Country folds test transfer to unseen countries, not causal effects."]}
    for split in ("city", "country"):
        report["ablations"][split] = {name: evaluate(rows, split, columns)
                                       for name, columns in variants.items()}
        report["learning_curve"][split] = {}
        for count in (25, 50, 100, 150):
            trials = [evaluate(rows, split, variants["all_features"], count, seed)
                      for seed in (11, 22, 33, 44, 55)]
            report["learning_curve"][split][str(count)] = {
                "mae_mean": round(statistics.mean(t["mae"] for t in trials), 3),
                "mae_min": min(t["mae"] for t in trials),
                "mae_max": max(t["mae"] for t in trials)}
    path = artifact("sample_size_report")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

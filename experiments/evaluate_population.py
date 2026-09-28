"""Compare road, population-density, and combined city predictors on identical folds."""

import csv
import json
import statistics

from fixed_window import CROP_FEATURES
from prepare_population import FEATURE_NAMES as POPULATION_FEATURES
from project import CONFIG, WINDOW_METRES, artifact
from road_baseline import FEATURE_NAMES as FULL_ROAD_FEATURES, folds, metrics, ridge_predict


def main():
    with artifact("fixed_window_features").open(encoding="utf-8", newline="") as handle:
        roads = list(csv.DictReader(handle))
    with artifact("population_features").open(encoding="utf-8", newline="") as handle:
        population = {row["city"]: row for row in csv.DictReader(handle)}
    with artifact("road_features").open(encoding="utf-8", newline="") as handle:
        full_roads = {row["city"]: row for row in csv.DictReader(handle)}
    if set(population) != {row["city"] for row in roads}:
        raise ValueError("Population and fixed-window cities differ; run population preparation")
    rows = []
    for road in roads:
        pop = population[road["city"]]
        if road["country"] != pop["country"] or road["target"] != pop["target"]:
            raise ValueError(f"Population city metadata differs: {road['city']}")
        road_x = [float(road[name]) for name in CROP_FEATURES]
        pop_x = [float(pop[name]) for name in POPULATION_FEATURES]
        extent_x = [float(full_roads[road["city"]][name]) for name in FULL_ROAD_FEATURES[:4]]
        rows.append({"city": road["city"], "country": road["country"],
                     "target": float(road["target"]), "road": road_x,
                     "population": pop_x, "combined": road_x + pop_x,
                     "extent": extent_x, "extent_population": extent_x + pop_x})
    variants = ("mean", "population", "road", "combined", "extent", "extent_population")
    predictions = []
    scores = {}
    for grouping in ("country", "city"):
        for fold_number, test_indices in enumerate(folds(rows, grouping), 1):
            test_set = set(test_indices)
            train = [row for index, row in enumerate(rows) if index not in test_set]
            test = [rows[index] for index in test_indices]
            estimates = {"mean": [statistics.mean(row["target"] for row in train)] * len(test)}
            for name in variants[1:]:
                estimates[name] = ridge_predict(
                    [row[name] for row in train], [row["target"] for row in train],
                    [row[name] for row in test], feature_count=len(train[0][name]))
            for name in variants:
                for row, predicted in zip(test, estimates[name]):
                    predictions.append({"split": grouping, "fold": fold_number,
                                        "model": name, "city": row["city"],
                                        "country": row["country"], "actual": row["target"],
                                        "predicted": predicted})
        scores[grouping] = {}
        for name in variants:
            selected = [row for row in predictions if row["split"] == grouping and
                        row["model"] == name]
            scores[grouping][name] = metrics([row["actual"] for row in selected],
                                              [row["predicted"] for row in selected])
    with artifact("population_predictions").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    report = {"cities": len(rows), "population_source": CONFIG["population"]["source"],
              "population_epoch": CONFIG["population"]["epoch"],
              "population_resolution": CONFIG["population"]["resolution"],
              "window_km": WINDOW_METRES / 1000, "road_features": CROP_FEATURES,
              "extent_features": FULL_ROAD_FEATURES[:4],
              "population_features": POPULATION_FEATURES,
              "model": "Ridge regression; alpha=10; training-fold standardization",
              "splits": scores,
              "limitations": "The Traffic Index date and spatial footprint are unknown; residential density is not daytime travel demand; fixed-window road completeness is unverified."}
    artifact("population_evaluation_report").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()

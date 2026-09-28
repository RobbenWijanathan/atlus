"""Select CNN/GNN epoch counts using only each outer training fold.

This keeps the country-held-out evaluation separate from epoch selection.
Requires the cache from prepare_learning_data.py.
"""

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict

import torch
from torch import nn

from project import artifact
from road_baseline import folds, metrics
from train_learned import (RoadGNN, SmallCNN, batches, load_data, predict,
                           train_fold)


def inner_split(rows, train_indices, grouping, seed):
    rng = random.Random(seed)
    if grouping == "country":
        countries = defaultdict(list)
        for index in train_indices:
            countries[rows[index]["country"]].append(index)
        names = list(countries)
        rng.shuffle(names)
        validation = []
        for name in names:
            validation.extend(countries[name])
            if len(validation) >= len(train_indices) // 5:
                break
    else:
        shuffled = train_indices[:]
        rng.shuffle(shuffled)
        validation = shuffled[:max(1, len(shuffled) // 5)]
    validation_set = set(validation)
    training = [index for index in train_indices if index not in validation_set]
    return training, validation


def choose_epochs(rows, kind, train_indices, grouping, seed, max_epochs=40,
                  patience=7, augmentation="none", population=False):
    inner_train, validation = inner_split(rows, train_indices, grouping, seed)
    torch.manual_seed(seed)
    model = SmallCNN(population) if kind == "cnn" else RoadGNN(population)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    target_mean = statistics.mean(rows[index]["target"] for index in inner_train)
    target_scale = statistics.pstdev(rows[index]["target"] for index in inner_train) or 1.0
    best_error, best_epoch = float("inf"), 1
    for epoch in range(1, max_epochs + 1):
        model.train()
        for indices in batches(inner_train, 16, seed + epoch, True):
            batch = [rows[index] for index in indices]
            targets = torch.tensor([(row["target"] - target_mean) / target_scale
                                    for row in batch], dtype=torch.float32)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(predict(model, kind, batch,
                                                  augmentation=augmentation), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        predicted = []
        with torch.no_grad():
            for indices in batches(validation, 16, seed, False):
                predicted.extend((predict(model, kind, [rows[index] for index in indices]) *
                                  target_scale + target_mean).tolist())
        actual = [rows[index]["target"] for index in validation]
        error = statistics.mean(abs(y - p) for y, p in zip(actual, predicted))
        if error < best_error:
            best_error, best_epoch = error, epoch
        if epoch - best_epoch >= patience:
            break
    return best_epoch, round(best_error, 3), len(validation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment-images", action="store_true",
                        help="Apply whole-image right-angle rotations and reflections during CNN training only")
    parser.add_argument("--augmentation", choices=("none", "right_angle_flip",
                                                   "small_angle", "small_angle_flip"),
                        help="Image training augmentation; small angle means uniform -15 to +15 degrees")
    parser.add_argument("--models", nargs="+", choices=("cnn", "gnn"), default=["cnn", "gnn"])
    parser.add_argument("--population", action="store_true",
                        help="Add aligned GHSL residential-density image/node features")
    args = parser.parse_args()
    if args.augment_images and args.augmentation is not None:
        parser.error("Choose either --augment-images or --augmentation")
    image_augmentation = args.augmentation or ("right_angle_flip" if args.augment_images
                                                else "none")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rows = load_data(population=args.population)
    report = {"cities": len(rows), "selection": "inner validation MAE; up to 40 epochs; patience 7",
              "outer_split": "country", "models": {},
              "image_augmentation": image_augmentation,
              "population_density": args.population,
              "caveat": "Single inner validation split and single seed per fold; performance remains uncertain."}
    outputs = []
    country_folds = folds(rows, "country")
    for kind in args.models:
        selected, predictions = [], []
        for fold_number, test_indices in enumerate(country_folds, start=1):
            test_set = set(test_indices)
            train_indices = [index for index in range(len(rows)) if index not in test_set]
            seed = 2000 + fold_number
            augmentation = image_augmentation if kind == "cnn" else "none"
            epochs, validation_mae, n_validation = choose_epochs(rows, kind, train_indices,
                                                                  "country", seed,
                                                                  augmentation=augmentation,
                                                                  population=args.population)
            print(f"{kind} fold {fold_number}/5: selected {epochs} epochs on "
                  f"{n_validation} inner validation cities", flush=True)
            estimates, parameters = train_fold(rows, kind, train_indices, test_indices,
                                               seed, epochs, augmentation=augmentation,
                                               population=args.population)
            selected.append({"fold": fold_number, "epochs": epochs,
                             "inner_validation_mae": validation_mae,
                             "inner_validation_cities": n_validation})
            for index, estimate in zip(test_indices, estimates):
                item = {"model": kind, "fold": fold_number, "city": rows[index]["city"],
                        "country": rows[index]["country"], "actual": rows[index]["target"],
                        "predicted": estimate}
                predictions.append(item)
                outputs.append(item)
        report["models"][kind] = {"parameters": parameters,
                                  "selected_epochs": selected,
                                  **metrics([row["actual"] for row in predictions],
                                            [row["predicted"] for row in predictions])}
        print(f"{kind} outer result: {report['models'][kind]}", flush=True)
    # A training-fold mean on these same outer test cities is the reference.
    baseline_actual, baseline_predicted = [], []
    for test_indices in country_folds:
        test_set = set(test_indices)
        mean = statistics.mean(row["target"] for index, row in enumerate(rows)
                               if index not in test_set)
        baseline_actual.extend(rows[index]["target"] for index in test_indices)
        baseline_predicted.extend(mean for _ in test_indices)
    report["models"]["training_mean"] = metrics(baseline_actual, baseline_predicted)
    variant = "plain" if image_augmentation == "none" else image_augmentation
    if args.population:
        variant = "population" if variant == "plain" else f"population_{variant}"
    artifact("nested_report", variant=variant).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with artifact("nested_predictions", variant=variant).open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outputs[0]))
        writer.writeheader()
        writer.writerows(outputs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

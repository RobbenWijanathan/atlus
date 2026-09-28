"""Small CNN and graph-message-passing regressors on fixed 5 km road crops.

Run after prepare_learning_data.py:
  .venv/Scripts/python experiments/train_learned.py --splits country city

The city is the supervision unit. No map patch receives an invented city label.
Hyperparameters and epochs are fixed before held-out evaluation.
"""

import argparse
import csv
import json
import math
import statistics

import numpy as np
import torch
from torch import nn

from project import (CONFIG, MODEL_CACHE, RASTER_PIXELS, ROAD_CLASSES, WINDOW_METRES,
                     artifact, model_manifest, population_input, slug)
from road_baseline import folds, metrics


class SmallCNN(nn.Module):
    def __init__(self, population=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(len(ROAD_CLASSES) + int(population), 12, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, image):
        return self.encoder(image).flatten()


class RoadGNN(nn.Module):
    def __init__(self, population=False):
        super().__init__()
        self.node_input = nn.Linear(3 + int(population), 32)
        self.edge_input = nn.Linear(len(ROAD_CLASSES) + 2, 32)
        self.update1 = nn.Linear(64, 32)
        self.update2 = nn.Linear(64, 32)
        self.readout = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, edge_index, edge_attr, batch, graph_count):
        h = torch.relu(self.node_input(x))
        edge_h = self.edge_input(edge_attr)
        source, target = edge_index
        for layer in (self.update1, self.update2):
            messages = h[source] + edge_h
            aggregate = torch.zeros_like(h)
            aggregate.index_add_(0, target, messages)
            degree = torch.zeros((len(h), 1), device=h.device)
            degree.index_add_(0, target, torch.ones((len(target), 1), device=h.device))
            h = torch.relu(layer(torch.cat((h, aggregate / degree.clamp_min(1)), dim=1)))
        pooled = []
        for graph in range(graph_count):
            nodes = h[batch == graph]
            pooled.append(torch.cat((nodes.mean(dim=0), nodes.max(dim=0).values)))
        return self.readout(torch.stack(pooled)).flatten()


def load_data(population=False):
    manifest = json.loads(model_manifest().read_text(encoding="utf-8"))
    samples = []
    for row in manifest:
        with np.load(MODEL_CACHE / row["file"]) as data:
            image = torch.from_numpy(data["image"].astype(np.float32) / 255)
            x = torch.from_numpy(data["x"].copy())
            if population:
                with np.load(population_input(slug(row["city"]))) as pop_data:
                    density = pop_data["density"].astype(np.float32)
                    if (int(pop_data["epoch"]) != CONFIG["population"]["epoch"] or
                            str(pop_data["resolution"]) != CONFIG["population"]["resolution"] or
                            "bounds" not in data or
                            not np.allclose(pop_data["bounds"], data["bounds"], atol=1e-9)):
                        raise ValueError(f"Population source or bounds differ from roads: {row['city']}")
                if density.shape != (RASTER_PIXELS, RASTER_PIXELS):
                    raise ValueError(f"Population grid shape differs for {row['city']}")
                image = torch.cat((image, torch.from_numpy(density[None, :, :])))
                columns = np.clip(np.rint((x[:, 0].numpy() + 1) / 2 * (RASTER_PIXELS - 1)),
                                  0, RASTER_PIXELS - 1).astype(int)
                pixel_rows = np.clip(np.rint((1 - x[:, 1].numpy()) / 2 * (RASTER_PIXELS - 1)),
                                     0, RASTER_PIXELS - 1).astype(int)
                x = torch.cat((x, torch.from_numpy(density[pixel_rows, columns, None])), dim=1)
            samples.append({"city": row["city"], "country": row["country"],
                            "target": row["target"],
                            "image": image, "x": x,
                            "edge_index": torch.from_numpy(data["edge_index"].copy()),
                            "edge_attr": torch.from_numpy(data["edge_attr"].copy())})
    return samples


def batches(indices, batch_size, seed, shuffle):
    ordered = indices[:]
    if shuffle:
        np.random.default_rng(seed).shuffle(ordered)
    for start in range(0, len(ordered), batch_size):
        yield ordered[start:start + batch_size]


def graph_batch(samples):
    nodes, edges, attributes, assignment = [], [], [], []
    offset = 0
    for graph_number, sample in enumerate(samples):
        nodes.append(sample["x"])
        edges.append(sample["edge_index"] + offset)
        attributes.append(sample["edge_attr"])
        assignment.append(torch.full((len(sample["x"]),), graph_number, dtype=torch.long))
        offset += len(sample["x"])
    return (torch.cat(nodes), torch.cat(edges, dim=1), torch.cat(attributes),
            torch.cat(assignment), len(samples))


def transform_images(images, augmentation):
    if augmentation == "right_angle_flip":
        transformed = []
        for image in images:
            image = torch.rot90(image, int(torch.randint(0, 4, ()).item()), dims=(-2, -1))
            if bool(torch.rand(()) < 0.5):
                image = torch.flip(image, dims=(-1,))
            transformed.append(image)
        return torch.stack(transformed)
    if augmentation in ("small_angle", "small_angle_flip"):
        angles = (torch.rand(len(images), device=images.device) * 30 - 15) * math.pi / 180
        cosines, sines = torch.cos(angles), torch.sin(angles)
        theta = torch.zeros((len(images), 2, 3), dtype=images.dtype, device=images.device)
        theta[:, 0, 0], theta[:, 0, 1] = cosines, -sines
        theta[:, 1, 0], theta[:, 1, 1] = sines, cosines
        grid = nn.functional.affine_grid(theta, images.shape, align_corners=False)
        images = nn.functional.grid_sample(images, grid, mode="bilinear",
                                           padding_mode="zeros", align_corners=False)
        if augmentation == "small_angle_flip":
            flips = torch.rand(len(images), device=images.device) < 0.5
            images[flips] = torch.flip(images[flips], dims=(-1,))
        return images
    if augmentation != "none":
        raise ValueError(f"Unknown image augmentation: {augmentation}")
    return images


def predict(model, kind, samples, augmentation="none"):
    if kind == "cnn":
        images = torch.stack([sample["image"] for sample in samples])
        return model(transform_images(images, augmentation))
    return model(*graph_batch(samples))


def train_fold(rows, kind, train_indices, test_indices, seed, epochs,
               augmentation="none", population=False):
    torch.manual_seed(seed)
    model = SmallCNN(population) if kind == "cnn" else RoadGNN(population)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    train_y = [rows[index]["target"] for index in train_indices]
    target_mean = statistics.mean(train_y)
    target_scale = statistics.pstdev(train_y) or 1.0
    for epoch in range(epochs):
        model.train()
        for indices in batches(train_indices, 16, seed + epoch, True):
            batch = [rows[index] for index in indices]
            targets = torch.tensor([(row["target"] - target_mean) / target_scale
                                    for row in batch], dtype=torch.float32)
            optimizer.zero_grad()
            estimates = predict(model, kind, batch, augmentation=augmentation)
            loss = nn.functional.mse_loss(estimates, targets)
            loss.backward()
            optimizer.step()
    model.eval()
    predictions = []
    with torch.no_grad():
        for indices in batches(test_indices, 16, seed, False):
            batch = [rows[index] for index in indices]
            estimates = predict(model, kind, batch)
            predictions.extend((estimates * target_scale + target_mean).tolist())
    return predictions, sum(parameter.numel() for parameter in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=("city", "country"), default=["country"])
    parser.add_argument("--models", nargs="+", choices=("cnn", "gnn"), default=["cnn", "gnn"])
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rows = load_data()
    report = {"cities": len(rows), "epochs": args.epochs, "batch_size": 16,
              "input": f"{WINDOW_METRES / 1000:g} km city-center crops, {RASTER_PIXELS}x{RASTER_PIXELS} road-class images and intersection graphs",
              "splits": {}, "caveat": "One seed and a fixed epoch count; scores have substantial sampling and training uncertainty."}
    predictions = []
    for grouping in args.splits:
        report["splits"][grouping] = {}
        for kind in args.models:
            model_predictions = []
            parameter_count = None
            for fold_number, test_indices in enumerate(folds(rows, grouping), start=1):
                test_set = set(test_indices)
                train_indices = [index for index in range(len(rows)) if index not in test_set]
                print(f"Training {kind}, {grouping} fold {fold_number}/5: "
                      f"{len(train_indices)} train, {len(test_indices)} test", flush=True)
                estimates, parameter_count = train_fold(rows, kind, train_indices,
                                                         test_indices, 1000 + fold_number, args.epochs)
                for index, estimate in zip(test_indices, estimates):
                    prediction = {"model": kind, "split": grouping, "fold": fold_number,
                                  "city": rows[index]["city"], "country": rows[index]["country"],
                                  "actual": rows[index]["target"], "predicted": estimate}
                    predictions.append(prediction)
                    model_predictions.append(prediction)
            report["splits"][grouping][kind] = {
                "parameters": parameter_count,
                **metrics([row["actual"] for row in model_predictions],
                          [row["predicted"] for row in model_predictions])}
            print(f"{grouping} {kind}: {report['splits'][grouping][kind]}", flush=True)
    artifact("learned_model_report").write_text(json.dumps(report, indent=2) + "\n",
                                                       encoding="utf-8")
    with artifact("learned_model_predictions").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

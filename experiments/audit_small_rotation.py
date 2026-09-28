"""Measure raster effects of a 15-degree rotation on cached road maps."""

import argparse
import json
import statistics

import numpy as np
import torch
from PIL import Image

from project import MODEL_CACHE, artifact, figure, model_manifest, slug
from train_learned import transform_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-city", help="Optional city label for a saved preview image")
    args = parser.parse_args()
    manifest = json.loads(model_manifest().read_text(encoding="utf-8"))
    torch.manual_seed(20260928)
    mass_ratios, fractional_ratios = [], []
    example = None
    for row in manifest:
        with np.load(MODEL_CACHE / row["file"]) as data:
            image = torch.from_numpy(data["image"].astype(np.float32) / 255)
        rotated = transform_images(image[None], "small_angle")[0]
        mass_ratios.append(float(rotated.sum() / image.sum()))
        active = rotated > 0
        fractional_ratios.append(float(((rotated > 0) & (rotated < 1)).sum() / active.sum()))
        if args.preview_city == row["city"]:
            rgb = torch.stack((rotated[0], rotated[2], rotated[3]), dim=-1).numpy()
            example = Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8)).resize((384, 384))
    if example is not None:
        example.save(figure("small_rotation", slug(args.preview_city)))
    elif args.preview_city:
        raise ValueError(f"Preview city is not in the model manifest: {args.preview_city}")
    report = {"cities": len(manifest), "rotation": "uniform -15 to +15 degrees",
              "sampling": "bilinear with black zero padding",
              "mean_pixel_mass_ratio": round(statistics.mean(mass_ratios), 4),
              "min_pixel_mass_ratio": round(min(mass_ratios), 4),
              "mean_fraction_of_active_pixels_interpolated": round(statistics.mean(fractional_ratios), 4),
              "caveat": "Pixel mass and interpolation do not directly measure graph connectivity."}
    artifact("small_rotation_audit").write_text(json.dumps(report, indent=2) + "\n",
                                                       encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

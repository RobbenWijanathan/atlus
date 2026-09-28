"""Offline checks for geographic population alignment and tile boundaries."""

import tempfile
import unittest
import zipfile
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from prepare_population import density_grid, tile_indices


class PopulationGridTests(unittest.TestCase):
    def test_tile_indices_cross_longitude_boundary(self):
        self.assertEqual(tile_indices((9.98, 50.72, 10.02, 50.78)),
                         [(4, 19), (4, 20)])
        self.assertEqual(tile_indices((-76.7, 39.2, -76.6, 39.3)), [(5, 11)])

    def test_density_has_no_seam_at_tile_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            sources = {}
            for col, west in ((19, 0), (20, 10)):
                data = np.full((1200, 1200), 100000, dtype=np.float32)
                with MemoryFile() as memory:
                    with memory.open(driver="GTiff", width=1200, height=1200,
                                     count=1, dtype="float32", crs="EPSG:4326",
                                     transform=from_origin(west, 60, 1 / 120, 1 / 120),
                                     nodata=-200) as dataset:
                        dataset.write(data, 1)
                    archive_path = Path(folder) / f"tile_{col}.zip"
                    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                        archive.writestr(f"tile_{col}.tif", memory.read())
                    sources[(4, col)] = archive_path
            with patch("prepare_population.ensure_tile", side_effect=lambda row, col: sources[row, col]):
                image, features, audit = density_grid((9.98, 50.72, 10.02, 50.78))
            self.assertEqual(image.shape, (96, 96))
            self.assertEqual(audit["coverage_fraction"], 1.0)
            self.assertTrue(np.all(image > 0))
            self.assertEqual(image.dtype, np.float32)
            self.assertGreater(float(image.min()), 1.0)
            expected_residents = 100000 * (0.04 * 0.06) / ((1 / 120) ** 2)
            self.assertAlmostEqual(math.expm1(features[0]), expected_residents / 25,
                                   places=2)


if __name__ == "__main__":
    unittest.main()

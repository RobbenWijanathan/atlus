# Road network experiments

The unit of prediction is **one city**. Each city has one Numbeo Traffic Index
label and one OpenStreetMap major-road network. A tile, road segment, or
intersection is not an independently labelled traffic observation.

## Pipeline

Use Python 3.10 or newer from the repository root. All stages use the settings
in `experiments/config.json`; set the `ATLUS_CONFIG` environment variable to an
absolute JSON path to use a different input dataset or artifact root. The
pipeline resolves dependencies and stops if a stage fails.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r experiments/requirements-models.txt
.\.venv\Scripts\python -m pip install -r experiments/requirements-data.txt
.\.venv\Scripts\python experiments/pipeline.py list
.\.venv\Scripts\python experiments/pipeline.py run --target coverage --dry-run
.\.venv\Scripts\python experiments/pipeline.py run --target coverage
```

Available targets are `baseline` (statistics, join and sample-size audits),
`coverage` (baseline, fixed windows, model inputs and coverage audit),
`evaluation` (all saved model treatments and comparisons), `collection`
(coverage preparation, uniform road queries and collection audit), and
`collection_audit` (audit already downloaded windows). The `population` target
downloads GHSL population tiles, aligns residential density to the same fixed
road windows, compares population-only, road-only and combined ridge models,
and trains population-aware CNN and graph models. Run it with:

```powershell
.\.venv\Scripts\python experiments/pipeline.py run --target population
```

The first run downloads official 2020 GHS-POP R2023A 30-arcsecond tiles from
the European Commission into `artifacts/cache/population/source/`. Later runs
reuse verified tiles. `artifacts/tables/population__features.csv` contains
window-level population density statistics; city-aligned raster inputs reside
under `artifacts/cache/population/inputs/`. Country- and city-held-out ridge
predictions and scores are in `artifacts/predictions/population__predictions.csv`
and `artifacts/reports/population_evaluation__report.json`. Population-aware
neural predictions and scores use the registered `learned_nested__population`
artifact names. The original road-only artifacts remain separately available.
Residential population describes where people live, not where trips occur;
Numbeo's Traffic Index CSV does not specify its observation date or geographic
footprint. GHSL values are estimated residents per source cell, converted to
residents per square kilometre before rendering. Upsampling to the road image
size does not add geographic detail.

The verified 2026-09-28 run prepared all 185 eligible cities from 81 official
tile archives and saved one held-out prediction per city and model. On the
same country-held-out folds, the mean predictor scored 40.923 MAE;
population-only ridge 43.727; fixed-window roads-only ridge 41.995;
fixed-window roads plus density ridge 41.907; original-graph extent ridge
39.693; and original-graph extent plus density ridge 41.515. The road-only
CNN scored 41.836 versus 44.041 with a density image; the road-only GNN
scored 39.205 versus 39.765 with node density. Lower MAE is better. These
results do not establish a gain from population density on this dataset.
See `docs/PROJECT_LOG.md` for the comparison design and limitations.

The pipeline migrates
historical outputs on its first real run. To migrate without running stages:
`python experiments/pipeline.py migrate`. The migration checks for conflicting
files and can be rerun. Each real run appends stages, outcome and config hash
to `artifacts/reports/pipeline__runs.jsonl`. Before replacing a stage's reports,
tables or predictions, it copies their previous versions under
`artifacts/history/pipeline_runs/<run_id>/`.

To collect all eligible cities rather than the audit-flagged subset:

```powershell
.\.venv\Scripts\python experiments/pipeline.py run --target collection --collection-scope all
```

For a controlled batch, repeat `--city` with exact labels from
`road_baseline__features.csv`:

```powershell
.\.venv\Scripts\python experiments/pipeline.py run --target collection --city "Beirut, Lebanon"
```

The collector resumes saved cities. An alternate Overpass endpoint can be
selected with `--overpass-url`, but a reachable status page does not establish
that graph queries will succeed. Both the configured endpoint and
`https://lz4.overpass-api.de/api` reset the controlled Beirut queries on
2026-09-28. A regional-extract import path is not yet implemented.
Transient connection, timeout, and HTTP 429/5xx errors receive bounded retries
with delays configured in `experiments/config.json`. Empty-road responses and
query/metadata mismatches are recorded without retries. Each attempt is logged
in `artifacts/reports/uniform_collection__attempts.jsonl`; a configured streak
of transient city failures stops the batch and leaves a partial run report.
Cached cities are skipped without a network request. The collector runs one
query at a time and enforces a minimum interval between query starts.
After changing the road-window query settings, use `--collection-overwrite` to
replace the separate uniform-window cache; otherwise mismatched cached windows
are reported as failures. Replaced GraphML and metadata are copied to
`artifacts/history/uniform_windows/` first.
The query settings in `experiments/config.json` include network type,
component retention, edge truncation and simplification. Cache reuse checks
those settings alongside the city, center, bounds and road filter.
An incomplete graph/metadata pair is rejected for review; an explicit
overwrite archives whichever previous files exist before replacing them.
The collection target writes its audit even if individual city queries fail,
then exits with an incomplete status and lists failed cities in the run report.
The audit distinguishes missing cities from incomplete graph/metadata pairs
and records whether the last collection run succeeded.
Offline retry checks run with
`.\.venv\Scripts\python -B -m unittest discover -s experiments -p 'test_*.py'`.
Public instances are listed on the
[OpenStreetMap Overpass wiki](https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances).
The pipeline never treats tiles as independent traffic labels.
The comparison stage checks every saved prediction's city, country, observed
target and country fold against the current 185-city table before reporting
scores. It cannot establish that a saved prediction used the current graph
version; run the `evaluation` target after changing graph inputs.

## Generated artifacts

Everything generated by Atlus is under the configured `artifacts/` root:

| Folder | Contents | Version control |
| --- | --- | --- |
| `tables/` | City features and data audits | Kept with the research record |
| `reports/` | Experiment metrics, manifests, collection status | Kept with the research record |
| `predictions/` | City-level held-out predictions | Kept with the research record |
| `figures/` | Optional previews | Kept with the research record |
| `cache/` | Prepared tensors, OSMnx responses and new GraphML | Ignored by Git; reproducible from sources |
| `history/` | Superseded reports, predictions, GraphML and metadata | Ignored by Git; older local snapshots |

Generated filenames use `stage__role.ext` or `stage__variant__role.ext`.
`experiments/project.py` owns the naming registry and all generated paths;
stages do not construct their own result filenames. The OSMnx cache is a
third-party internal cache and retains that library's hash filenames.
Graph discovery reads one GraphML file per configured city folder and does not
require a separately saved PNG; model preparation renders images from GraphML.
If a graph has uniform-query metadata, crops use its recorded request bounds
after validating the center and width. The partial uniform dataset has not
been used to train or compare models.

The uniform collector requests geodesic windows around reference coordinates
using the configured major-road tags and retains disconnected components.
The requested bounds document extraction, not OSM completeness. The four-city
pilot found 202/359 nodes/edges for Aachen, 40/57 for Muscat, 48/72 for
Orlando, and 3/3 for Los Angeles. As of 2026-09-28, 15 of 203 windows are
saved. The controlled Beirut query failed after three connection resets at
both the configured and alternate endpoints. Anchorage returned no qualifying
roads at its chosen reference point and remains under center review. See
`artifacts/reports/uniform_collection__audit.json` and
`artifacts/reports/uniform_collection__manifest.json`. Existing model scores
still use the original 185-city dataset.

## Data checks

There are 203 usable road graph and traffic-label pairs. City names are matched
to the `worldcities.csv` reference using the center of each graph's geographic
bounding box, then checked against country names. Explicit aliases cover Cebu
City, Czechia, Hong Kong, and South Korea. All 203 pairs matched a reference
city within 75 km. This coordinate check reduces name collisions; it does not
prove that the scraped road boundary represents a consistent urban area.

The network bounding boxes vary widely: widths range from about 1.8 to 179 km
and heights from about 1.2 to 199 km. The fixed-window experiment uses a
5-by-5 km square centered on the reference city's coordinates. It retains 185
cities whose graph bounding boxes cover that square and have at least 20 nodes
and edges within it. Bounding-box coverage is an approximate check; a
nonrectangular place boundary could still leave parts of the square without
downloaded roads. All road data here includes only major road classes specified
in the archived original scraper at `docs/legacy/osm_scraper.py.txt`.

## Results

All reported errors below are mean absolute error (MAE), in Traffic Index
points. Country-held-out five-fold evaluation tests cities in countries absent
from each training fold. The comparison uses the **same 185 cities and folds**.

| Model | Input | Parameters | Country-held-out MAE | R² |
| --- | --- | ---: | ---: | ---: |
| Training mean | No road input | 0 | 40.923 | -0.024 |
| Extent ridge | Four full-network size/extent features | 4 coefficients | 39.693 | 0.025 |
| CNN | 4-channel, 96×96 fixed-window road image | 42,901 | 41.836 | -0.056 |
| CNN with rotations/reflections | Same image; transformed during training only | 42,901 | 40.212 | -0.015 |
| CNN with ±15° rotations | Same image; bilinear rotation during training | 42,901 | 42.533 | -0.102 |
| CNN with ±15° rotations and flips | Same image; bilinear rotation and reflection | 42,901 | 42.526 | -0.099 |
| Graph model | Fixed-window intersections, segments and road attributes | 6,625 | 39.205 | 0.028 |

The CNN and graph model rows use an inner validation split **within each outer
training fold** to choose epoch count, followed by refitting on that entire
outer training fold. The CNN also had a fixed 25-epoch run, which performed
worse; its results are retained in `learned_model_report.json` as a training
schedule diagnostic.

The rotation/reflection experiment applies a random multiple of 90 degrees
and an optional horizontal flip to **the entire four-channel image** during
training. Validation and test images keep their original orientation. On the
same held-out cities, MAE decreased by 1.624 points relative to the CNN
without augmentation. A paired country-resampling interval for that difference
spans -3.184 to +0.416 points, so this single run does not establish a reliable
gain. Rotations and reflections preserve the connectivity visible in the crop;
they do not add independently labelled cities.

Randomly erasing a rectangle from these sparse road maps is not in the current
training pipeline. Black pixels mean no drawn road, so erasure would present
missing roads as real gaps and artificial road endings. A future missing-data
robustness experiment should mark erased pixels as **unknown** with a separate
mask channel. Zooming into subregions also changes the geographic extent of
the input: if tested, multiple subregions should be aggregated into one
city-level prediction instead of treating each crop as a separately measured
Traffic Index.

The ±15° experiments used bilinear resampling with zero padding outside the
rotated square. In this pilot they performed worse than the right-angle
rotations. The image audit found mean road-pixel mass retention of 98.03%
(minimum 89.5%) and widespread interpolation of thin lines. See the living
[project log](../docs/PROJECT_LOG.md) for the complete decision record and
interpretation; these experiments do not justify a final augmentation choice
without independent replication.

The graph model's MAE is 0.488 points lower than the extent ridge model on the
same cities. A paired country-resampling interval for that difference spans
-3.597 to +3.324 points. Thus the current run does not establish a reliable
advantage for the graph model. The interval is exploratory because the project
design and data preparation were developed using this dataset.

The full 203-city ridge baseline achieved 37.877 MAE on held-out cities but
40.192 MAE on held-out countries. A diagnostic shows that full-network extent
features explain most of its improvement; features normalized by network area
perform poorly. The simple-model learning curve improves from 25 to 100
training cities, then is nearly flat between 100 and 150. That plateau is
specific to these features and labels, not proof that more data could not help
a different representation or target.

## Interpretation and next data work

These are predictive associations, not causal estimates of what a road change
would do to traffic. Numbeo's Traffic Index is a user-contributed composite,
and road geometry omits demand, transit, land use, and time-varying conditions.
The key data improvement is collecting the same physical city-center extent
for every city and obtaining more reliable, comparable target measurements.
Adding road tiles without additional city labels does not increase the number
of independently labelled outcomes.

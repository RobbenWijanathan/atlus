# Atlus: project guide and research log

**Status:** exploratory research prototype
**Last updated:** 2026-09-28

This document is the living record of what Atlus is trying to predict, which
data and methods were actually used, what each run found, and which decisions
are still provisional. Update it whenever an experiment changes the evidence
or a design choice changes. Keep measured results separate from hypotheses.

## Research question

Can the structure and shape of a city's major-road network help predict its
city-level traffic conditions? A secondary, harder question in the repository
README is whether road-network data helps predict quality of life.

The present experiments predict **Numbeo Traffic Index**. This is a
user-contributed composite involving commute time and related quantities; it
is not a sensor-measured, time-resolved congestion series. Quality of life has
not yet been modeled. Road maps alone omit population, trip demand, public
transit, land use, income, and temporal conditions; the present density
experiment supplies one of those missing inputs. The work measures
predictive association, not what would happen if a road were built or removed.

## Current data and supervision

| Item | Current state | Source in repository |
| --- | --- | --- |
| Traffic labels | 206 CSV rows; 203 location-validated graph/label pairs | `dataset/city_quality/city_traffic.csv` |
| Road networks | 203 GraphML/PNG pairs in the original snapshot, major roads only; the pipeline uses GraphML | `dataset/city_visuals/osm_city_data/` |
| Reference coordinates | City names, countries and coordinates for join checks | `dataset/city_master/worldcities.csv` |
| Comparable fixed windows | 185 cities with usable 5-by-5 km graph bounding boxes | `artifacts/tables/fixed_window__features.csv` |
| Residential population density | 2020 GHS-POP R2023A gridded estimates, aligned to all 185 eligible road windows; 81 verified source tiles | `artifacts/cache/population/`, `artifacts/reports/population__report.json` |
| Quality-of-life labels | 305 CSV rows; no training experiment yet | `dataset/city_quality/city_quality_of_life.csv` |

**Independent sample:** a city with one observed target. Rotations, flips,
image tiles, nodes and road segments provide alternative views or features of
that city; they do not create independently measured city labels.

The graph-location join checks normalized city and country names against the
reference coordinates, with explicit aliases for Cebu City, Czechia, Hong Kong
and South Korea. All 203 graph centers were within 75 km of the selected
reference city. The audit is in `artifacts/reports/join__audit.json`. It does
not prove that every scraped polygon captures the same fraction of an urban
area. Raw network bounding-box widths range from about 1.8 to 179 km and
heights from about 1.2 to 199 km.

## Evaluation rules

1. Split by city, never by image tile or road patch.
2. Primary transfer check: five folds that each hold out whole countries.
   A city-held-out split is a less stringent supplementary check.
3. Fit scaling, regression coefficients, neural weights and epoch choices
   using only the relevant training cities. For neural models, an inner
   validation subset of the outer training fold selects the epoch count.
4. Compare methods on the **same eligible cities and folds**. Report mean
   absolute error (MAE) in Traffic Index points and coefficient of
   determination (R²). Lower MAE is better. R² measures squared-error
   improvement over the overall observed-mean reference, so it can be
   negative for poor out-of-fold predictions.
5. Treat intervals from country resampling as exploratory. This dataset has
   informed model and preprocessing development; an untouched external test
   set would be needed for a stronger final claim.

## Decision log

| Date | Decision | Reason | Status |
| --- | --- | --- | --- |
| 2026-09-28 | Start with Traffic Index, then consider quality of life | Traffic is the closer target to roads; quality of life combines many other factors | Active |
| 2026-09-28 | Begin with simple road statistics and a training-mean predictor | Around 200 city labels make a check against low-capacity models essential | Active |
| 2026-09-28 | Audit joins with reference coordinates and countries | Name-only joins confuse London, San Jose and Vancouver | Active |
| 2026-09-28 | Use 5-by-5 km city-center crops for learned models | Full networks cover highly unequal geographic extents | Provisional; bounding-box coverage is an imperfect test |
| 2026-09-28 | Try small CNN and graph models | Test whether spatial layout or connectivity adds information beyond statistics | Experimental |
| 2026-09-28 | Select neural training epochs inside the outer training folds | Avoid choosing training duration from held-out countries | Active |
| 2026-09-28 | Test whole-image right-angle rotations and flips | They preserve pixels and road connectivity while changing orientation | Experimental |
| 2026-09-28 | Keep right-angle rotations as the current image augmentation candidate; do not adopt ±15° raster rotation | Smaller angles with and without flips had worse held-out MAE and introduced interpolation/black corner padding | Provisional, needs independent replication |
| 2026-09-28 | Avoid random rectangular road deletion in the main training input | Black means no road; erasure would introduce fictitious gaps and endings | Active pending a masked-unknown design |
| 2026-09-28 | Audit the saved 5 km inputs and collect new windows in a separate dataset | The original bounding-box screen admits nearly empty center crops and excludes graphs too small to contain the window | Pilot; do not replace the 185-city evaluation set until coverage and center choices are reviewed |
| 2026-09-28 | Use one configured pipeline and artifact root | Earlier scripts required a manual order and embedded output locations and names | Active; original graph snapshot remains versioned input, new generated outputs use `artifacts/` |
| 2026-09-28 | Validate cached road queries and held-out prediction rows before reuse | A changed query or mismatched prediction file could otherwise silently enter later results | Active |
| 2026-09-28 | Bound and log Overpass retries; pause broad collection after repeated service failures | Two tested endpoints reset Beirut queries, so continuing through all 203 cities would not establish usable coverage | Active; collection still incomplete |
| 2026-09-28 | Evaluate defined urban-area graphs as the main citywide input candidate | A 5-by-5 km center square cannot be assumed to represent a whole-city Traffic Index; a published global graph dataset provides explicit urban boundaries | Research direction; city-label matching and predictive value untested |
| 2026-09-28 | Keep aligned residential density as an optional, auditable model input, but do not adopt it as a demonstrated accuracy improvement | On the same 185 cities and country-held-out folds, none of the tested density-aware models beat the existing road-only extent ridge (MAE 39.693); density-aware GNN scored 39.765 versus road-only GNN 39.205 | Tested once; geographic and temporal target mismatch remains |

## Experiment log

### 2026-09-28: GHSL population density integration

**Question:** Does estimated residential density within the same geographic
window as the roads improve held-out prediction of Numbeo Traffic Index?

**Inputs and eligible count:** 203 location-validated original city/graph/label
pairs; 185 cities eligible for the established 5-by-5 km road-window analysis.
The population source is the European Commission's [GHS-POP R2023A](https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php),
epoch 2020, WGS84 30-arcsecond (~1 km) cells. Source cells contain estimated
residents, not density or traffic counts. The Traffic Index CSV has neither a
date nor a geographic footprint.

**Method:** the configured `population` pipeline target downloads and verifies
the required official GHSL tile archives, preserving URLs and hashes in
`artifacts/reports/population__report.json`. It calculates fractional cell
overlap with each road window to estimate the residents inside that window,
divides cell counts by latitude-adjusted cell area to obtain residents per
square kilometre, and aligns this density surface to each road image. The
source grid is coarser than the 96-by-96 model image; image resampling adds no
new spatial observations. City predictors include mean and peak density and
the area fraction above 1,000 residents per square kilometre. Estimated
population inside the window is retained in the audit. The CNN receives a
fifth density channel, and the graph model receives density at each road node.
The pipeline also fits fixed-strength
ridge regressions with population features alone, road features alone and both
together. All comparisons use the same eligible cities and five held-out
country folds; city-held-out ridge scores are secondary. Neural epoch selection
uses only each outer training fold.

**Comparison and result:** 81 official tile archives were downloaded and
verified. All 185 eligible city windows were prepared, with complete GHSL
cell coverage according to the population raster audit. The table reports
mean absolute error (MAE; smaller is better) on the *same* country-held-out
city folds. The road-window ridge uses eleven 5 km road features; the extent
ridge uses four statistics of each original, variable-size road graph. These
are different road representations, so the extent rows are contextual
benchmarks rather than a controlled test of adding population to a fixed
window.

| Model input and method | Country-held-out MAE |
| --- | ---: |
| Training-fold mean | 40.923 |
| Population density features only, ridge | 43.727 |
| Fixed-window road features only, ridge | 41.995 |
| Fixed-window roads + density, ridge | 41.907 |
| Original road-graph extent features only, ridge | 39.693 |
| Original road-graph extent + density, ridge | 41.515 |
| Road image only, CNN | 41.836 |
| Road image + density image, CNN | 44.041 |
| Road graph only, GNN | **39.205** |
| Road graph + node density, GNN | 39.765 |

The combined fixed-window ridge improved its road-only counterpart by 0.088
MAE points, but both were worse than the training-fold mean. The density-aware
CNN and GNN were worse than their respective road-only versions. On secondary
city-held-out ridge folds, fixed-window roads plus density scored 39.165
versus 41.124 for roads alone; the original extent ridge still scored better
at 37.791. Country-held-out evaluation is more relevant to transfer to new
countries and is the main comparison here. Saved reports and predictions are
`artifacts/reports/population_evaluation__report.json`,
`artifacts/predictions/population__predictions.csv`,
`artifacts/reports/learned_nested__population__report.json`, and
`artifacts/predictions/learned_nested__population__predictions.csv`. The
prediction rows were checked for one held-out prediction per model and city,
matching labels, countries and folds. Reproduction command:
`python experiments/pipeline.py run --target population`.

**Limitations and decision:** residential population is an imperfect proxy for
trip demand, employment, vehicle use and transit service. The 5 km road input
still lacks a complete road-coverage guarantee and cannot be assumed to
represent all of a city's travel. Keep density as a separately auditable input;
judge its added value against population-only and road-only predictions on the
same cities. Boundary-matched full-urban-area inputs remain a separate planned
study. The present comparison uses one dataset, one split design, and one
random seed per neural fold; the small score differences should not be
interpreted as a reliable estimate of population's general value. The
population source is from 2020 while the label date is unknown. Retain the
population pipeline for boundary-matched, better-labeled studies, but do not
make density mandatory in the current road model.

### 2026-09-28: Literature review of citywide road coverage and traffic targets

**Question:** How do published studies define a city when linking its roads to
city-level outcomes, and what does that imply for our 5-by-5 km center crop?

**Inputs and eligible count:** the existing 203 location-validated Atlus
graph/Traffic Index pairs and 185 cities eligible for the previous fixed-window
analysis. The Traffic Index CSV has no observation date or geographic boundary
field. This entry reviews literature only: zero additional cities were matched,
collected, or trained.

**Method and comparison:** reviewed the papers, data repositories, and code
below for their geographic unit, road representation, target, and coverage
checks. Compared fixed center buffers, many local tiles summarized per city,
and mapped urban-area polygons. These are research designs, not new Atlus
model scores.

| Source | Geographic unit and method | Relevance and limitation |
| --- | --- | --- |
| [Chen et al. (2024)](https://filipbiljecki.com/publications/2024_land_urn.pdf); [code](https://github.com/ualsg/Global-road-network-patterns) | For 144 cities, use a 15 km radius from a center; split the area into 0.5, 1, and 2 km grids; classify local road patterns and summarize their distribution for the city. | Shows a reproducible multiscale alternative to one center tile. Its target is road morphology, not traffic. The authors estimate the chosen radius covers 57.8% of their cities' populations on average, so even this larger buffer is not necessarily a whole-city input. |
| [Boeing (2026)](https://journals.sagepub.com/doi/pdf/10.1177/23998083261446991); [code](https://github.com/gboeing/street-network-models); [GraphML data](https://doi.org/10.7910/DVN/KA5HJ3) | Use 2025 GHSL urban-area polygons to construct complete OpenStreetMap street graphs for 10,351 qualifying urban areas in 189 countries, with graph files and indicators. | Direct candidate input that avoids live per-city Overpass calls. It does not provide our traffic labels; its public-drivable-street definition differs from Atlus's major-road input. A GHSL urban polygon may not match the area respondents mean by a Numbeo city. |
| [Boeing, Pilgram, and Lu (2024)](https://escholarship.org/content/qt5094g60t/qt5094g60t.pdf) | Relate street-network indicators within GHSL urban areas to transport emissions and other area-level variables defined for the corresponding urban area. | Good example of aligning predictor and outcome footprints, and of including population/density/economic context. The outcome is transport emissions, not Traffic Index. |
| [Zhang et al. (2026)](https://www.nature.com/articles/s41467-026-71377-0) | Study road structure, built form, land use, and measured traffic dynamics across 30 cities. Check HERE road-segment coverage against a 1 km OSM road grid; 28 of 30 cities exceed 70% spatial coverage. | Traffic-related example showing explicit coverage checks and the relevance of non-road factors. Its segment/time-series traffic outcome is different from our one Numbeo index per city. |

**Result and limitations:** no reviewed source demonstrates that a 5-by-5 km
center crop predicts Numbeo's city Traffic Index, and no source establishes a
universal city boundary for that index. The 15 km radius is a deliberate
standardization for road-pattern comparisons, not proof of full-city coverage.
The GHSL network dataset is a promising reusable input, but matching its
urban areas to our 203 city labels must be audited geographically; published
availability does not establish match count, road completeness, or predictive
gain for Atlus. The local Traffic Index CSV lacks a date and footprint, so
temporal and geographic target alignment remain uncertain.

**Decision:** keep the 5 km experiment as a center-view benchmark. Before
more uniform-window collection or model scaling, audit matches between Atlus
city coordinates/names/countries and the published GHSL urban-area metadata;
record ambiguous or unmatched cases and road-class differences. On a common
eligible city set and country-held-out folds, compare a center-view baseline
with full-urban-area network statistics, then consider multiscale tiles
aggregated to one prediction per city. This is a planned comparison, not a
claimed result.

### 2026-09-28: Bounded collection retries and controlled Beirut run

**Question:** Can the existing uniform-window pipeline recover from temporary
Overpass failures without losing provenance or treating an empty response as a
network failure?

**Inputs and eligible count:** 203 original graph/traffic pairs; 18 cities
flagged for first collection by the existing-coverage audit; 15 uniform windows
already saved. Beirut was chosen for a one-city live run because its previous
attempt ended in a connection reset. No new training labels were introduced.

**Method:** configurable maximum attempts per city, exponential retry delay,
minimum spacing between network requests, and a stop condition after repeated
transient city failures were added to the existing collector and pipeline.
Connection, timeout and HTTP 429/5xx failures are retryable; empty-road results,
metadata mismatches, HTTP 406 and local socket permission denial are not.
Each noncached collection attempt is appended to the registered
`uniform_collection__attempts.jsonl`; the registered run report and audit
record final outcomes. The pipeline accepts exact `--city` labels for a
controlled batch. Offline tests covered two transient failures followed by a
success, a nontransient failure, HTTP status classification and local socket
permission denial. The audit now exposes incomplete graph/metadata pairs and
last-run success explicitly. No one-off data scripts or artifact locations
were introduced.

**Comparison and result:** a sandboxed request was denied locally with
`WinError 10013` and interrupted after its first recorded attempt. With network
permission, the default endpoint reset all three Beirut requests. A status
check of the documented alternate endpoint returned HTTP 200, but all three
Beirut graph requests to that endpoint also reset; a tiny read-only query
returned HTTP 406 by both GET and POST. The final pipeline audit still reports
15 of 203 collected windows, zero incomplete cached pairs, and Beirut missing.
The 203-city baseline and 185-city fixed-window counts were reproduced by the
pipeline; no model was retrained or compared on new data. Anchorage's selected
reference point is 15.374 km from the old graph's bounding-box midpoint and
is flagged for center review, not automatically replaced.

**Limitations and decision:** the connection resets and HTTP 406 do not prove
whether the service, this machine's network path, or its query policy caused
the failure. The uniform dataset remains too partial for a replacement model
comparison. Do not run the full-city collection until query access works and
flagged centers have been reviewed; keep the original 185-city results clearly
separate from these partial uniform inputs.

**Follow-up transport diagnosis:** inspection of the installed OSMnx 2.1.1
code confirmed that `graph_from_bbox` sends a POST to the Overpass interpreter
after checking its status endpoint. Python reached the alternate status
endpoint with HTTP 200. A tiny, read-only Overpass POST sent directly with
`requests`, bypassing OSMnx and this project's city query, returned HTTP 504.
This is evidence that the service/query path was failing independently of our
graph construction; it does not isolate the server from every possible network
intermediary. A Python HEAD request to Geofabrik's Lebanon `.osm.pbf` extract
returned HTTP 200 and a 52,504,966-byte content length. Thus code-based
regional downloads appear reachable, although a full download and an import
pipeline have not yet been tested. Manual per-city download is not required by
the research design. The next engineering option is a reusable regional-PBF
ingest stage with provenance and the same city-window/road-class checks.

### 2026-09-28: Pipeline and artifact migration

**Question:** Can the saved 203-city source analysis and 185-city model-input
analysis be reproduced through one configured command without changing their
results?

**Inputs and eligible count:** the versioned original GraphML and traffic CSV
gave 203 matched cities; 185 passed the historical fixed-window screen. The
existing held-out predictions were retained for a result-preservation check.

**Method:** `experiments/config.json` holds source locations, the one artifact
root, road-window settings, collection settings and quality thresholds.
`experiments/project.py` registers generated paths and names. The reusable
`experiments/pipeline.py` runs dependencies, supports a dry run and migrates
legacy results and caches without overwriting conflicting files. Generated
names follow `stage__role.ext` or `stage__variant__role.ext` under
`artifacts/{tables,reports,predictions,figures,cache}`. The previous raw graph
snapshot remains a versioned input under `dataset/`. Its original scraper was
archived as `docs/legacy/osm_scraper.py.txt`; the redundant
`dataset/compare_dataset.py` was removed because the join audit supersedes it.
Pipeline runs append stage outcomes and a configuration hash to
`artifacts/reports/pipeline__runs.jsonl`.
The pipeline copies prior reports, tables and predictions into local
`artifacts/history/pipeline_runs/` before a stage replaces them.
Graph discovery now accepts one GraphML file of any name per city folder and
does not require a companion PNG; learned-model preparation renders the raster
from GraphML.
For future uniformly queried graphs, fixed-window preparation reads the saved
request bounds and checks that the center and width match the configured
reference. This path has not been evaluated as a model dataset because only
15 of 203 uniform windows have been collected.
Uniform collection writes its latest report atomically, appends future
per-city attempts to `uniform_collection__attempts.jsonl`, and archives old
GraphML and metadata under `artifacts/history/` before an explicit overwrite.
Query settings for network type, component retention, edge truncation and
simplification are configurable; cache reuse compares them with the saved
metadata, as well as center, bounds, city and highway filter. The comparison
stage checks prediction model, city, country, observed target, fold, duplicate
rows and complete city coverage before scoring.
An incomplete graph/metadata pair is rejected for review, and explicit
overwrite archives whichever prior files exist.

**Comparison and result:** `pipeline.py run --target coverage` regenerated
the 203-city baseline, 185 fixed-window inputs and existing-coverage audit.
It reproduced the earlier 183 ordinary accepted, two sparse-warning accepted,
15 extent-failed and three too-few-road counts. Recomputing
`compare_models.py` from the migrated held-out predictions reproduced every
reported country-held-out MAE, including 40.923 for the training mean, 39.693
for extent ridge, 41.836 for the plain CNN and 39.205 for the graph model.
The stricter comparison reproduced the same seven model scores. Aachen's saved
uniform query was accepted with unchanged settings; changing simplification
in the query plan was rejected before a network request.
The small-rotation image audit reproduced 0.9803 mean pixel-mass retention.
No model was retrained in this refactor. The migration moved 27 result files,
186 model-input cache items, 30 uniform-window files and 16 OSMnx cache files.

**Limitations and decision:** a matching score checks preservation of the
existing computation; it does not validate the traffic target, coverage of
OpenStreetMap, city-center choice, or model generalization. The uniform-road
dataset remains partial at 15 of 203 cities, and model results still use the
old fixed-window data. A saved prediction can still be stale if the graph
changed while city labels and targets stayed identical; rerun the evaluation
target after input changes. Use the pipeline for subsequent attempts so data and
evaluation steps are not run in an accidental order.


### 2026-09-28: Existing-road coverage audit and uniform-query pilot

**Question:** Do the saved road graphs actually support comparable 5-by-5 km
city-center inputs, and can the same major-road query be issued for affected
cities?

**Inputs and eligible count:** all 203 location-validated graph/traffic-label
pairs, the existing 185-city fixed-window report, and the cached raster inputs.
The city reference coordinates come from `worldcities.csv`.

**Method:** `experiments/audit_existing_coverage.py` checks the prior window
screen and counts empty cells in a 4-by-4 partition of each accepted road
image. `experiments/collect_uniform_networks.py` defines geodesic 5 km bounds
around the same reference coordinates and queries the original eight major
road tags through OSMnx. It retains disconnected road components and stores
each query's bounds, center, settings, date and graph separately from the
original data. A dry run requires no network access.

**Audit result:** 183 accepted crops had fewer than eight empty grid cells;
two accepted crops had at least eight. Fifteen old graph bounding boxes did
not surround the intended window. Three further graphs passed that geometric
check but had fewer than 20 road nodes or edges in the window: Anchorage,
Los Angeles and Orlando. The audit files are
`artifacts/reports/existing_coverage__summary.json` and
`artifacts/tables/existing_coverage__audit.csv`.

**Four-city query pilot:** Aachen, Muscat, Los Angeles and Orlando were
queried with the same new 5 km method. New node/edge counts were 202/359,
40/57, 3/3 and 48/72, respectively. The corresponding counts from the old
graphs inside the proposed windows were 202/359, 32/45, 2/3 and 0/0. Aachen
is a useful unchanged reference; the Orlando and Muscat differences show that
the new extraction can supply roads missing from those old crops. They do not
isolate the cause: OSM may have changed, the old query boundary varied, and
the new collection retains disconnected components. Los Angeles remains
nearly empty under the same major-road filter at the chosen center. Saved
GraphML and metadata are under `artifacts/cache/uniform_windows/`.

**Collection attempt and recovery:** the default
`https://overpass-api.de/api` endpoint succeeded for the initial cities but
reset connections for Alexandria and Almaty during the 203-city run. A test of
the public Japan endpoint failed TLS hostname verification, so its certificate
was not bypassed. The OpenStreetMap wiki's direct
`https://lz4.overpass-api.de/api` endpoint successfully returned Alexandria
and Almaty; the resumable full run was restarted there. It later reset an
Auckland request. A focused run of the 18 audit-flagged cities returned no
qualifying major roads for Anchorage and suffered a connection reset for
Beirut. A subsequent attempt through `https://overpass.kumi.systems/api`
did not return a Beirut result before being stopped. Across the pilot and
partial runs, **15 of 203** city windows were saved, including three of the
18 review-first cities. `artifacts/reports/uniform_collection__audit.json`,
`artifacts/tables/uniform_collection__audit.csv`, and
`artifacts/reports/uniform_collection__manifest.json` record
the saved state. These are service availability observations, not evidence
about city road coverage. Do not rerun model comparisons on this partial new
dataset or claim that input coverage has been fixed for all cities.

The audit also flags 26 reference points more than 10 km from the old graph's
bounding-box midpoint for manual center review. This is a screening rule, not
proof of an incorrect city coordinate: old graphs can have asymmetric or
nonurban extents. Los Angeles is sparse despite a smaller midpoint distance,
which shows why node counts and geography need review too.

**Limitations and decision:** graph bounds only show geographic extent; they
cannot establish that every road within it was downloaded. Empty cells may
represent water, genuinely sparse roads, the major-road-only filter, an old
query boundary, or missing OSM data. The chosen reference coordinate may
also represent a poor part of a metropolitan area, notably for Los Angeles.
This audit does not establish missing OSM roads or improved predictive
performance. Keep the original graphs and comparison intact while piloting a
documented re-query. Review city centers before treating the new windows as
comparable urban cores.


### 2026-09-28: Full-network statistics baseline

**Input:** 12 GraphML-derived features for 203 cities, including network
size, bounding-box extent, directed road length, approximate density, degree
fractions, one-way fraction and road-class length fractions.

**Method:** fixed-strength ridge regression versus a training-fold mean;
five-fold city and country splits. Standardization is fitted on training
folds. Code: `experiments/road_baseline.py`.

| Held-out unit | Training mean MAE | Ridge MAE | Ridge R² |
| --- | ---: | ---: | ---: |
| City | 40.382 | 37.877 | 0.112 |
| Country | 40.915 | 40.192 | 0.012 |

**Interpretation:** some apparent signal, but transfer to unseen countries is
weak. A feature ablation found that full-network extent alone nearly matches
the full feature set, while normalized structure performs worse. A training
size diagnostic improved from 25 to 100 cities and changed little from 100 to
150 for this fixed feature set. These findings do not prove that additional
labels cannot help another representation. Artifacts:
`artifacts/reports/road_baseline__report.json`,
`artifacts/reports/sample_size__report.json`.

### 2026-09-28: Fixed-window features

**Input:** main roads within a 5-by-5 km square centered on the reference
city coordinate. The 185 retained graph bounding boxes cover this square and
contain at least 20 nodes and edges inside it.

**Method:** extract cropped structural statistics; compare against statistics
from the original full network on those same 185 cities. Code:
`experiments/fixed_window.py`.

| Held-out unit | Original full network MAE | Fixed-window MAE |
| --- | ---: | ---: |
| City | 39.654 | 41.124 |
| Country | 41.713 | 41.995 |

**Interpretation:** the fixed-window statistics did not improve this small
linear model. It is still the fairer input for checking learned spatial
representations because it reduces differences in geographic scale. The
original full-network model remains useful as a size/extent reference.
Artifact: `artifacts/reports/fixed_window__report.json`.

### 2026-09-28: CNN and road graph model

**Input:** four-channel 96-by-96 road images and intersection graphs rendered
from the same 5-by-5 km GraphML crops for 185 cities. The original 1600-by-1600
PNG files were not fed directly to the CNN. Prepared data lives in ignored
`artifacts/cache/model_inputs/`. Code: `prepare_learning_data.py`.

**Models:** a 42,901-parameter CNN and a 6,625-parameter, two-step message
passing graph regressor. Each uses a city-level regression target. Inner
validation chooses the epoch count, followed by refitting on the outer
training cities. Code: `train_learned.py`,
`train_with_inner_validation.py`.

| Country-held-out model, same 185 cities | MAE | R² |
| --- | ---: | ---: |
| Training mean | 40.923 | -0.024 |
| Full-network extent ridge | 39.693 | 0.025 |
| CNN | 41.836 | -0.056 |
| Graph model | 39.205 | 0.028 |

The graph model beats extent ridge by only 0.488 MAE points. The exploratory
paired country-resampling interval for graph minus ridge MAE is -3.597 to
+3.324, so this run does not establish a reliable advantage. Artifacts:
`artifacts/reports/learned_nested__plain__report.json`,
`artifacts/reports/model_comparison__report.json`.

### 2026-09-28: Whole-image right-angle rotation plus reflection

**Hypothesis:** orientation variation may reduce CNN overfitting without
altering the visible road network.

**Method:** during training only, apply a random 0°, 90°, 180° or 270°
rotation and optional horizontal reflection to all four channels together.
Validation and outer test inputs retain their original orientation. The
country folds, model size and inner epoch-selection procedure remain the
same. Code: `train_learned.py` with `--augment-images` via
`train_with_inner_validation.py`.

| CNN treatment | Country-held-out MAE | R² |
| --- | ---: | ---: |
| No image augmentation | 41.836 | -0.056 |
| Right-angle rotations and flips | 40.212 | -0.015 |

The 1.624-point MAE improvement is promising but uncertain: the exploratory
paired country-resampling interval for augmented minus plain CNN MAE is
-3.184 to +0.416. This augmentation does not outperform the extent ridge
baseline. Artifacts: `artifacts/reports/learned_nested__right_angle_flip__report.json`,
`artifacts/reports/model_comparison__report.json`.

### 2026-09-28: Smaller-angle raster rotations, alone and with flips

**Hypothesis:** angles between the right-angle steps might add useful
orientation variation. These are separate tests of uniform random rotations
from -15° to +15°, with and without a random horizontal reflection.

**Method:** apply the transformation only during CNN training. PyTorch
bilinear sampling keeps the output at 96-by-96 pixels and fills areas outside
the rotated source square with black. Use the same 185 city inputs, country
folds, architecture, optimizer and inner epoch-selection procedure as above.
Validation and test images remain untransformed. Code: `train_learned.py` and
`train_with_inner_validation.py` with `--augmentation small_angle` or
`--augmentation small_angle_flip`.

| CNN training treatment | Country-held-out MAE | R² |
| --- | ---: | ---: |
| No augmentation | 41.836 | -0.056 |
| Right-angle rotations and flips | 40.212 | -0.015 |
| ±15° rotations | 42.533 | -0.102 |
| ±15° rotations and flips | 42.526 | -0.099 |

**Image-integrity check:** across 185 randomly rotated input images, the mean
sum of road-pixel values after rotation was 98.03% of the original (minimum
89.5%). An average 93.86% of nonblack output pixels had interpolated rather
than original binary values. This does not directly measure road-graph
connectivity, but it shows that arbitrary-angle raster rotation substantially
changes the appearance of these one-pixel roads. A Jakarta example is saved
as `artifacts/figures/small_rotation__jakarta.png`. The image-integrity report is
`artifacts/reports/small_rotation__audit.json`.

**Interpretation:** in this pilot, smaller angles did not improve held-out
prediction, and adding flips to them made almost no difference. A paired
country-resampling interval for small-angle-plus-flip minus right-angle-plus-
flip MAE spans -0.831 to +5.642 points; it includes no difference. This is an
exploratory comparison after several trials on the same dataset, so it should
not be used as a universal rule. Re-rendering rotated *vector roads* with a
larger geographic margin would be a different, cleaner experiment than
interpolating these already cropped rasters. Artifacts:
`artifacts/reports/learned_nested__small_angle__report.json`,
`artifacts/reports/learned_nested__small_angle_flip__report.json`,
`artifacts/reports/model_comparison__report.json`.

## Open questions and next experiments

1. Standardize input coverage at the data-source level. The existing 5 km
   bounding-box check cannot establish that each square is fully covered by
   downloaded roads, and the raw network sizes vary dramatically. Re-fetch
   uniformly centered extents where feasible; record source date, boundary,
   missingness and road-class definitions.
2. Seek better target and explanatory data: a dated, comparable traffic
   measurement if available, plus daytime travel demand, employment,
   public transit and land-use indicators. The current residential population
   grid has been tested, but does not measure where or when trips occur.
   Keep road-only and non-road baselines separate to measure each input
   group's contribution.
3. If subcity crops are explored, aggregate multiple views to **one city
   prediction**. Record the physical size of each view; do not treat each
   crop as a separately observed Traffic Index.
4. If arbitrary-angle augmentation is revisited, render from vector roads
   with a source area large enough to cover the rotated window, and evaluate
   pixel and topology effects before training.
5. Assess quality of life as a separate target with an explicit account of
   the many non-road factors in its definition.

## How to update this log

For every new attempt, append a dated entry under **Experiment log** with:
the question, exact inputs and eligible city count, training and evaluation
procedure, commands or code paths, a comparison on matching folds, observed
numbers, uncertainty or failure modes, and the decision it changes. Add a
short row to **Decision log** when a provisional choice is adopted or reversed.
Preserve previous attempts, including unsuccessful ones.

## External method references

- [Numbeo Traffic Index definition](https://www.numbeo.com/traffic/indices_explained.jsp)
- [OSMnx paper](https://joss.theoj.org/papers/10.21105/joss.00215)
- [Road-pattern image study and code](https://github.com/ualsg/Global-road-network-patterns)
- [Graph Isomorphism Network paper](https://arxiv.org/abs/1810.00826)
- [Cutout masking study](https://arxiv.org/abs/1708.04552)
- [PyTorch grid sampling and out-of-bounds behavior](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)

A GIS research project testing whether major-road network structure helps
predict a city's Traffic Index. The current experiments use graph statistics,
small convolutional and graph models on OpenStreetMap roads. Quality of life
is a later research question; it has not yet been modeled.

Run the configured pipeline from the repository root:

```powershell
.\.venv\Scripts\python experiments/pipeline.py run --target coverage
```

Initial reproducible traffic-index baselines and small image/graph model experiments are documented in [experiments/README.md](experiments/README.md).

The evolving research questions, decisions, attempts, results, and limitations are recorded in [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md).

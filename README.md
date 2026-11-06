# OULAD: Academic Risk Prediction and Learning Resource Recommendations

This research project uses the Open University Learning Analytics Dataset (OULAD) to estimate a student's risk of not completing a course successfully and rank learning resources from that student's course. The risk ensemble combines LSTM, BiLSTM with attention, and LightGBM. The full recommender combines activity history, resource preferences, the course calendar, and a SASRec-based sequence model.

## Quick start

These commands use PowerShell, Python 3.12, and Git LFS. The `rtk` and `rtk proxy` prefixes are optional wrappers.

```powershell
rtk git lfs install
rtk git clone --branch master https://github.com/RAli2505/oulad-bilstm-attention-rec-system.git
cd oulad-bilstm-attention-rec-system
rtk proxy py -3.12 -m pip install -r requirements.txt psutil
```

Download the seven CSV files from the [official OULAD dataset page](https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/) and extract them into `data/oulad/raw/`. See [data attribution](research_archive/DATA_ATTRIBUTION.md) for the dataset citation and license.

From the repository root, check the environment and inputs, then run the system experiment with seed 42:

```powershell
rtk proxy py -3.12 -X utf8 -u run_system_experiment.py --seeds 42 --check
rtk proxy py -3.12 -X utf8 -u run_system_experiment.py --seeds 42 --device auto
```

The system experiment trains the risk ensemble and evaluates the baseline adaptive recommender at course days 7, 14, 21, 28, 56, 84, 112, 140, and 168. The console prints the paths to the forecast and system runs.

To evaluate the full recommender, supply the **completed forecast run** that contains the requested day and seed. Replace `RISK_RUN_DIRECTORY` with the actual directory name printed by the preceding run.

```powershell
rtk proxy py -3.12 -X utf8 -u run_oulad_agent_experiment.py --risk-run "runs/RISK_RUN_DIRECTORY" --windows 56 --seeds 42 --device auto
```

The repeat-versus-discovery experiment uses artifacts from the full recommender for matching days and seeds. See the [full recommender protocol](docs/OULAD_FULL_AGENT_DESIGN_RU.md), the [repeat-versus-discovery protocol](REPEAT_DISCOVERY_EXPERIMENT_RU.md), and the [commands for ablations, subgroup analysis, and cross-course evaluation](docs/ARTICLE_FOLLOWUP_COMMANDS_RU.md). These detailed notes are in Russian.

## Source code

| Path | Purpose |
| --- | --- |
| [`data/oulad_loader.py`](data/oulad_loader.py) | Load and prepare OULAD data |
| [`models/`](models/) and [`risk_experiments/`](risk_experiments/) | Risk models, learner splits, training, calibration, and evaluation |
| [`recommendations/`](recommendations/) | Preference profiles, candidate retrieval, SASRec, and ranking |
| [`run_system_experiment.py`](run_system_experiment.py) | Run risk prediction and baseline recommendation evaluation |
| [`run_oulad_agent_experiment.py`](run_oulad_agent_experiment.py) | Train and evaluate the full recommender |
| [`run_repeat_discovery_experiment.py`](run_repeat_discovery_experiment.py) | Compare policies for recommending previously visited and new resources |
| [`tests/`](tests/) | Automated checks |

## Results and trained weights

| Location | Contents |
| --- | --- |
| [All-window results index](research_archive/results_20261106/manifest.json) | Coverage of 24 days and five seeds for the full recommender, a directly readable ranking table, and supplementary aggregate results, including all five recommendation ablation runs |
| [November 5, 2026 snapshot](research_archive/results_20261105/) | Completed full recommender, `balance_v1`, and `rankmix_v2` runs. See `reports/` for summaries and `artifacts/` for metrics, settings, runtime, and memory measurements |
| [November 4, 2026 snapshot](research_archive/results_20261104/) | Risk prediction, cross-course evaluation, recommendation ablations, and additional analyses |
| [Historical reports](research_archive/snapshot_20261101/reports/) | Tables from earlier experiments |
| `runs/` | Local output from new runs, including logs, metrics, and checkpoints |

The [day 56, seed 42 model bundle](research_archive/results_20261105/artifacts/models_day56_seed42.zip) contains trained risk models, preprocessing, calibration, ensemble coefficients, and the full recommender bundle. If the ZIP files contain Git LFS pointers, run `rtk git lfs pull`. To resume an interrupted experiment, use its original run directory and checkpoints.

Risk prediction reports ROC-AUC, PR-AUC, precision, recall, F1, and calibration. Recommendation evaluation reports Precision@K, Recall@K, NDCG@K, MAP@K, and MRR@K. Observed future visits serve as a proxy for relevance. The published results use exploratory validation on 2013 data. The external 2014 split was held out. See the [risk prediction protocol](docs/RISK_ARTICLE_PROTOCOL_RU.md) for architecture attribution and evaluation details.

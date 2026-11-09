# Early in-hospital mortality from labs, a chest X-ray, and prior notes

Student project. The label is `hospital_expire_flag`: whether the patient dies during the current hospital stay. Every input has to be knowable in the first 24 hours of that stay, or it has to come from an earlier stay.

This is a course project, not a clinical tool. Do not use the predictions for patient care.

The PhysioNet files were not in this folder when the code was written. `src/data/schema.py` uses the published MIMIC-IV 2.2, MIMIC-CXR-JPG 2.0.0, and MIMIC-IV-Note column names. `scripts/01_build_cohort.py` reads the headers on disk before it builds anything and stops if a required name is missing. `outputs/data_inspection.md` records what it found.

Python 3.11 or 3.12 is fine. One GPU with 16 GB is the intended training machine. A laptop CPU can run the smoke test.

## Leakage rules

These are implemented in `src/data/cohort.py` and checked in `src/data/checks.py`.

- Tabular features are age, sex, and the median of each lab in `src/data/schema.py` with `charttime` inside `[admittime, admittime + 24 hours]`.
- The image is the earliest PA or AP chest X-ray in that same window, and not after `dischtime`.
- A note is kept only when its `charttime` is strictly before `admittime` and its `hadm_id` is a different admission. The discharge summary of the stay being predicted is excluded. A radiology report written during the current stay is excluded too.
- `deathtime`, date of death, race, insurance, and admission type are not model inputs.
- Train / validation / test are 70% / 15% / 15% by `subject_id`. One patient is in one split. The official MIMIC-CXR split file is not used, because this task is defined on admissions and the fractions are fixed in `config.yaml`.
- The test split is first used in `scripts/07_evaluate.py`. Earlier scripts load train and validation only, except script 05, which caches test embeddings without fitting on them.
- Median imputation is fit on the training split. The fusion model reuses that same imputer.
- Neural-net checkpoints are the epoch with the best validation AUROC. XGBoost early stopping watches validation AUROC. The F1 threshold is chosen on validation.

Age is `anchor_age + (admission year - anchor_year)`, which is the usual MIMIC-IV definition.

## Data you were given

Expected layout once PhysioNet access is approved:

```text
data/mimiciv/hosp/admissions.csv.gz
data/mimiciv/hosp/patients.csv.gz
data/mimiciv/hosp/labevents.csv.gz
data/mimiciv/hosp/d_labitems.csv.gz
data/mimic-cxr-jpg/files/...
data/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz
data/mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv.gz
data/mimic-iv-note/note/discharge.csv.gz
data/mimic-iv-note/note/radiology.csv.gz
```

`data/` is in `.gitignore`. Do not commit those files.

X-rays have no `hadm_id`. A study is linked to an admission when `subject_id` matches and the study datetime falls inside the admission, within the first 24 hours.

Set `data.source` to `mimic` in `config.yaml` after the files are in place. The cohort is capped at `cohort.max_admissions` (8000) after the three modalities have been required.

## Synthetic data

`scripts/00_make_synthetic_data.py` writes a fake copy of the same columns under `data/synthetic/`. It also writes `data/synthetic/SYNTHETIC_DO_NOT_REPORT.txt`.

The fake data plants bad rows on purpose:

| Planted row | Why it must be dropped |
| --- | --- |
| X-ray id `before-...` | PA film from before this admission |
| X-ray id `lateral-...` | inside 24 hours, but not frontal |
| X-ray id `late-...` | AP film 36 hours after admission |
| X-ray id `frontal-...` | the only film that should be kept |
| creatinine `99999` at 48 hours | lab outside the window |
| notes containing `LEAK_` | current admission, including one with an early timestamp but the current `hadm_id` |
| notes containing `PRIOR_NOTE_OK` | the two notes that should be kept |

Positive synthetic exams have a white square in the corner, and prior notes contain `synthetic_high_risk` or `synthetic_low_risk`. Those cues exist so a smoke test has a signal to fit. They are not medical findings. Metrics from this data are a pipeline check. The results file says `data_source = synthetic` on every row.

## Setup

From this folder, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first real training run downloads DenseNet-121, ViT, and `emilyalsentzer/Bio_ClinicalBERT`.

If the GPU runs out of memory, lower `image.batch_size` and `text.batch_size` in `config.yaml`. Mixed precision is on automatically when CUDA is available. Script 05 writes embeddings to `outputs/embeddings/` so script 06 does not load the image or text network again.

`image.num_workers` is 0 because that is the reliable setting on Windows. On Colab you can raise it.

## How to run

Smoke test (small cohort, one epoch):

```powershell
python scripts/run_pipeline.py --smoke
```

Full run, one stage at a time:

```powershell
python scripts/00_make_synthetic_data.py
python scripts/01_build_cohort.py
python scripts/02_train_tabular.py
python scripts/03_train_images.py
python scripts/04_train_text.py
python scripts/05_cache_embeddings.py
python scripts/06_train_fusion.py
python scripts/07_evaluate.py
python scripts/08_explain.py
```

Skip script 00 when `data.source` is `mimic`. `python scripts/run_pipeline.py` runs the same sequence.

`outputs/cohort/cohort.csv.gz` is the modeling table. `outputs/cohort/cohort_preview.csv` is the first 20 rows without the full note text. `outputs/cohort/funnel.txt` counts how many admissions survive each filter. `outputs/data_inspection.md` lists the column names that were actually in the files.

## What each model uses

1. **Tabular.** XGBoost on age, sex, and the 12 first-day lab medians. Missing labs are median-imputed from the training split. `scale_pos_weight` is the number of training negatives divided by the number of training positives.
2. **Images.** DenseNet-121 and `vit_base_patch16_224`, both from `timm`, at 224x224. Training uses a 10-degree rotation and a small brightness/contrast change. There is no left-right flip, because that swaps the patient's sides. The better of the two on validation AUROC is the image block in the fusion model. Both are still scored on the test set.
3. **Text.** `emilyalsentzer/Bio_ClinicalBERT`, truncated to 512 tokens. The most recent prior notes are placed at the start of the string, so truncation keeps those. The embedding is the final-layer `[CLS]` vector, before the linear layer.
4. **Fusion.** That image embedding, the text embedding, and the imputed tabular features are concatenated. A second XGBoost model is trained on the vector. It is fit on train and early-stopped on validation.

## Evaluation

`scripts/07_evaluate.py` writes:

- `outputs/metrics/results.csv`
- `outputs/metrics/results.md`
- `outputs/metrics/calibration_*.png`
- `outputs/metrics/predictions_test.csv`

The table has one row each for tabular, DenseNet, ViT, text, and fusion. Columns are AUROC, AUPRC, F1, the validation-chosen threshold, and the data source. AUROC and AUPRC ignore the threshold. There is no results table in the repository on purpose: the numbers are whatever this script computes. If fusion is not the best row, that is the result.

`outputs/metrics/val_scores.json` stores validation AUROC used to pick checkpoints. Those numbers are not test results.

The random seed is `seed` in `config.yaml`.

## Explainability

`scripts/08_explain.py` writes into `outputs/explain/`:

- `shap_summary.png` and three patient waterfall plots for the tabular XGBoost model
- Grad-CAM overlays on DenseNet for a handful of test X-rays (highest and lowest predicted risk)
- `fusion_modality_shap.csv` and a bar chart

The modality file reports the share of total mean absolute SHAP in the tabular block, the image block, and the text block. It also reports the average per feature. The image and text blocks have hundreds of columns, so their total share can be large even when each single column is small. The sample size is `explain.shap_samples`.

## Where to point during a presentation

| File | What to walk through |
| --- | --- |
| `config.yaml` | paths, the 8000 cap, the 24 hour window, hyperparameters |
| `src/data/schema.py` | the only tabular feature list, and the lab item ids |
| `src/data/inspect.py` | header check that runs before any join |
| `src/data/cohort.py` | the X-ray link, the note rule, the lab window, the subject split |
| `src/data/checks.py` | the assertions, including the synthetic traps |
| `src/data/synthetic.py` | how the fake files and the planted bad rows are built |
| `src/models/loop.py` | one training epoch, mixed precision, checkpoint by validation AUROC |
| `src/models/tabular.py` | imputation and `scale_pos_weight` |
| `src/models/image.py` | DenseNet, ViT, and the penultimate embedding |
| `src/models/text.py` | ClinicalBERT and the `[CLS]` vector |
| `src/models/fusion.py` | concatenation order |
| `src/eval/metrics.py` | AUROC, AUPRC, F1 threshold, calibration |
| `scripts/07_evaluate.py` | the only place test metrics are written |

## Cohort columns that are not features

`split`, `image_path`, `note_text`, `study_datetime`, and the ids are stored so later scripts can load the right file and check the window. Training scripts pass an explicit feature list. They do not feed every numeric column to the model.

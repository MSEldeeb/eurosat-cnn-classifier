# Step-by-step guide

This guide takes the project from an empty folder to a published GitHub repository. Each step lists what to run, what to check, and what to commit, so that the history of the repository documents how the project was built.

**Time needed:** about half a day, plus training time.

| Hardware | Baseline (15 epochs) | Improved (30 epochs) |
|---|---|---|
| Single CPU core (measured) | about 18 min | about 2 h |
| Modern laptop, 4–8 cores | roughly 5–10 min | roughly 30–60 min |
| Google Colab, free GPU | about 1–2 min | about 5 min |

---

## Step 0: Set up the environment

Requirements: Python 3.10, 3.11 or 3.12, and Git.

```bash
mkdir eurosat-cnn-classifier && cd eurosat-cnn-classifier
git init
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

Copy in the project skeleton: `README.md`, `LICENSE`, `.gitignore`, `pyproject.toml`, `requirements.txt` and `src/eurosat_cnn/__init__.py` + `config.py`. Then install:

```bash
pip install -e ".[dev,notebooks,presentation]"
python -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"
```

Create an empty repository on GitHub named `eurosat-cnn-classifier` (no README, since you already have one), then:

```bash
git add .
git commit -m "Project skeleton: package layout, config, licence"
git branch -M main
git remote add origin https://github.com/MSEldeeb/eurosat-cnn-classifier.git
git push -u origin main
```

## Step 1: Get the data

Add `src/eurosat_cnn/download.py`, then:

```bash
python -m eurosat_cnn.download
```

**Check:** `data/raw/EuroSAT_RGB/` contains 10 class folders and 27,000 `.jpg` files in total.

If the automatic download fails, download `EuroSAT_RGB.zip` manually from Zenodo (see `docs/DATASET.md`) and extract it to `data/raw/`.

```bash
git add src/eurosat_cnn/download.py docs/DATASET.md
git commit -m "Add dataset download script and dataset documentation"
```

The `data/` folder is in `.gitignore` and must never be committed.

## Step 2: Data loading and splitting (with tests)

Add `src/eurosat_cnn/data.py`, `tests/conftest.py` and `tests/test_data.py`.

```bash
pytest tests/test_data.py -q
```

**Check:** all tests pass. They verify that the split is stratified, disjoint and reproducible, and that pixels are scaled to [0, 1].

```bash
git add src/eurosat_cnn/data.py tests/
git commit -m "Add stratified train/val/test split and tf.data pipeline with tests"
```

## Step 3: Exploratory data analysis

Open `notebooks/01_data_exploration.ipynb` (`jupyter lab`) and run all cells. Then **write your own observations** in the final cell, in your own words.

Optional script version: `python -m eurosat_cnn.eda`.

**Check:** figures in `reports/figures/` (class distribution, sample grid, mean colour per class).

```bash
git add notebooks/01_data_exploration.ipynb src/eurosat_cnn/eda.py reports/figures/
git commit -m "Exploratory data analysis: class balance, samples, colour statistics"
```

## Step 4: Baseline CNN

Add `src/eurosat_cnn/models.py`, `src/eurosat_cnn/train.py` and `tests/test_models.py`.

```bash
pytest -q                                                              # all tests green
python -m eurosat_cnn.train --model baseline --epochs 1 --fraction 0.05    # 1-minute smoke test
```

Then train for real, either in `notebooks/02_baseline_cnn.ipynb` or with:

```bash
python -m eurosat_cnn.train --model baseline --epochs 15
```

**Check:** `models/baseline.keras` and `reports/baseline_training.json` exist. Note the best validation accuracy and whether training accuracy pulls away from validation accuracy (overfitting). Write this in the notebook's observations cell.

```bash
git add src/eurosat_cnn/models.py src/eurosat_cnn/train.py tests/test_models.py \
        notebooks/02_baseline_cnn.ipynb reports/baseline_*
git commit -m "Baseline CNN: training script, model tests and first results"
```

## Step 5: Improved CNN

Run `notebooks/03_improved_cnn.ipynb` or:

```bash
python -m eurosat_cnn.train --model improved --epochs 30
```

Validation accuracy can be low in the first epoch or two. That is normal for batch normalisation; judge the whole curve.

**Check:** `models/improved.keras` exists, and the validation curve is higher and closer to the training curve than the baseline's.

```bash
git add notebooks/03_improved_cnn.ipynb reports/improved_* reports/figures/
git commit -m "Improved CNN: rotation/flip augmentation, batch norm, global pooling"
```

## Step 6: Final evaluation and error analysis

Run `notebooks/04_evaluation_and_error_analysis.ipynb` or:

```bash
python -m eurosat_cnn.evaluate
```

This evaluates both models **once** on the test set, saves `reports/metrics_*.json`, the confusion matrix and misclassified examples, writes `reports/results.md`, and **fills in the Results table in `README.md` automatically**.

Look at the misclassified tiles and write your conclusions in the notebook: which classes are confused, and why?

```bash
git add src/eurosat_cnn/evaluate.py notebooks/04_evaluation_and_error_analysis.ipynb \
        reports/ README.md
git commit -m "Test-set evaluation, confusion matrix and error analysis"
```

## Step 7: Presentation

```bash
python presentation/make_presentation.py
```

This fills `presentation/presentation_template.pptx` with **your** results (accuracies, per-class chart, activation maps from your model, your real misclassified tiles) and writes `presentation/EuroSAT_CNN_presentation.pptx`. Open it, check every slide and read the speaker notes. Each slide has notes written for a general audience.

```bash
git add presentation/
git commit -m "Presentation for a general audience, generated from results"
```

## Step 8: Continuous integration

Add `.github/workflows/tests.yml` and push. On GitHub, open the **Actions** tab and confirm that the tests run green. The badge at the top of the README then turns green.

```bash
git add .github/
git commit -m "Add GitHub Actions workflow running the test suite"
git push
```

## Step 9: Polish the repository on GitHub

- **About** (right side of the repo page): description *"CNN land-use classification of Sentinel-2 satellite images (EuroSAT) with TensorFlow/Keras"*, topics `deep-learning`, `computer-vision`, `tensorflow`, `keras`, `cnn`, `remote-sensing`, `satellite-imagery`, `eurosat`.
- **Release:** create release `v1.0.0` and attach `models/improved.keras`. Model files are kept out of Git, and a release is the clean way to share them.


---

## Running on Google Colab (free GPU)

```python
!git clone https://github.com/MSEldeeb/eurosat-cnn-classifier.git
%cd eurosat-cnn-classifier
!pip install -q -e ".[presentation]"
!python -m eurosat_cnn.download
!python -m eurosat_cnn.train --model baseline --epochs 15
!python -m eurosat_cnn.train --model improved --epochs 30
!python -m eurosat_cnn.evaluate
```

Select *Runtime → Change runtime type → GPU* first. Download `models/` and `reports/` afterwards and commit the reports locally.

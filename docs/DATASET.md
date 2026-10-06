# Dataset: EuroSAT (RGB)

## Where to get it

| Option | How |
|---|---|
| **Recommended** | `python -m eurosat_cnn.download` downloads and extracts the data to `data/raw/EuroSAT_RGB/` |
| Manual (Zenodo) | https://zenodo.org/records/7711810 → download `EuroSAT_RGB.zip` (about 90 MB) and extract it so that `data/raw/EuroSAT_RGB/<ClassName>/*.jpg` exists |
| Original project page | https://github.com/phelber/EuroSAT |
| TensorFlow Datasets | https://www.tensorflow.org/datasets/catalog/eurosat (`eurosat/rgb`) |

After downloading, the folder must contain these 10 sub-folders:

```
data/raw/EuroSAT_RGB/
├── AnnualCrop/            3,000 images
├── Forest/                3,000
├── HerbaceousVegetation/  3,000
├── Highway/               2,500
├── Industrial/            2,500
├── Pasture/               2,000
├── PermanentCrop/         2,500
├── Residential/           3,000
├── River/                 2,500
└── SeaLake/               3,000      total: 27,000
```

`data/` is listed in `.gitignore`. The images are **not** committed to the repository; anyone can reproduce the project with the download step.

## What it is

- **Source:** the Sentinel-2 satellites of the EU Copernicus programme. The imagery is free and open.
- **Content:** 27,000 labelled image tiles from 34 European countries.
- **Format:** 64 × 64 pixels, RGB (JPEG); 10 m ground resolution, so each tile covers 640 m × 640 m.
- **Labels:** 10 land-use / land-cover classes (see above).
- **Citation:** P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.
- **Licence:** released by the authors for research use (see the original repository for the current licence terms). The images contain modified Copernicus Sentinel data.

## Why this dataset

| Criterion | EuroSAT | Why it matters |
|---|---|---|
| **Real-world, industry-relevant** | Real satellite imagery, not toy pictures | The same kind of data is used commercially, e.g. to estimate roof areas for solar panels, monitor crops or track urban growth. |
| **Balanced classes** | Largest class is only 1.5× the smallest | Accuracy is an honest metric; a model cannot score well by always predicting the majority class. |
| **Hard enough to be interesting** | Visually similar classes (pasture vs herbaceous vegetation, highway vs river, annual vs permanent crop) | There is real room for model design and meaningful error analysis, unlike MNIST-style datasets where everything reaches 99 %. |
| **Domain knowledge can be used** | Overhead images have no "up" | Justifies a specific, explainable augmentation choice (90° rotations and flips). |
| **Trainable without special hardware** | 64 × 64 px tiles, about 90 MB | A full experiment runs on a laptop CPU or a free Colab GPU. |
| **Public benchmark** | Widely cited; published reference results | Results can be placed in context against the literature. |

### Alternatives considered

| Dataset | Why not chosen |
|---|---|
| MNIST / Fashion-MNIST | Grayscale 28 × 28 px; too easy, and not representative of real computer-vision work |
| CIFAR-10 | Generic objects; widely overused as a portfolio project; less connected to applied, geospatial problems |
| UC Merced Land Use | Only 2,100 images; too small to train a CNN from scratch reliably |
| EuroSAT multispectral (13 bands) | Very interesting, but less intuitive to present; a natural next step (see README) |

## How the data is split

A stratified split with a fixed seed (`SEED = 42` in `config.py`), produced by `eurosat_cnn.data.stratified_split`:

| Split | Images | Purpose |
|---|---:|---|
| Train | 18,899 (70 %) | fit model weights |
| Validation | 4,051 (15 %) | checkpointing, early stopping, learning-rate schedule |
| Test | 4,050 (15 %) | final, one-time evaluation |

Every class has the same share in each split. The split is identical in the notebooks, the scripts and the presentation.

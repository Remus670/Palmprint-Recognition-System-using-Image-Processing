# Fingerprint Recognition

A small project for an image processing class. The goal is to recognize fingerprints by matching them against a small database, using classical computer vision methods (SIFT and a few preprocessing tricks) instead of deep learning. I tried 4 different methods and compared them on a public dataset to see which one works best.


## Dataset

I used the [Fingerprint Dataset for FVC2000_DB4_B](https://www.kaggle.com/datasets/peace1019/fingerprint-dataset-for-fvc2000-db4-b?resource=download) from Kaggle. It is the public part (Set B) of the fourth database from the **Fingerprint Verification Competition 2000**, generated synthetically with the SFinGE algorithm. The images are clean and artifact-free, which makes it a nice dataset for testing classical methods.

After downloading, the folder structure is:

```
dataset_FVC2000_DB4_B/dataset/
    train_data/   <- training images
    real_data/    <- test images (used for evaluation)
    np_data/      <- not used (preprocessed numpy arrays)
```

Each image is named like `101_1.tif`, `101_2.tif`, etc. The first number is the finger ID, and the second is the impression number.

## Methods I tried

All four methods use SIFT for keypoint detection and a brute-force matcher with Lowe's ratio test (0.70). The difference is in the preprocessing step before SIFT:

- **SIFT + KNN** — grayscale + histogram equalization (baseline)
- **Gaussian + SIFT** — adds a Gaussian blur to reduce noise
- **Gabor + SIFT** — applies a Gabor filter to enhance the ridges (this one is specific to fingerprints)
- **LBP + SIFT** — uses Local Binary Patterns to encode the texture before SIFT

A ResNet18 reference value is also shown on the comparison chart, but it is not actually trained — it is just a hardcoded number used as a deep learning reference.

## Files

- **`model_utils.py`** — the core module. Contains all the preprocessing functions (Gaussian, Gabor, LBP), SIFT descriptor extraction, the matcher with Lowe's ratio test, and the evaluation function. Every other script imports from here.
- **`train.py`** — runs SIFT on every image in `train_data/` and saves all the descriptors into `trained_model.pkl`. This is not real training in the ML sense — it just builds a database of descriptors.
- **`test.py`** — loads the saved model and evaluates it on `real_data/`. Prints the total accuracy and shows visual side-by-side keypoint matches for the first few images.
- **`main.py`** — a small CLI that wraps train / test / predict in one place. Useful when you want to run everything with one command and choose the method via arguments.
- **`compare_methods.py`** — runs the full train + test pipeline for all 4 methods automatically and saves a bar chart with the results in `results/method_comparison.png`. This is the main script for the report.
- **`predict_select.py`** — opens a file dialog so you can manually pick fingerprint images one by one and see the prediction with a visual match. Used for the demo.
- **`requirements.txt`** — the Python libraries needed.

## How to run

Install the requirements:

```bash
pip install -r requirements.txt
```

Then:

```bash
python train.py              # build the descriptor database
python test.py               # evaluate on the test set
python compare_methods.py    # compare all 4 methods + save chart
python predict_select.py     # demo with manual image selection
```

## Notes

- The "score" you see in the test output (e.g., 3876) is not a percentage — it is the total number of good SIFT matches summed across all training images of the predicted user. Bigger means more confident.
- The "KNN" in the method names refers to OpenCV's `knnMatch(k=2)` used for Lowe's ratio test, not to a global KNN classifier.
- Because FVC2000_DB4 is synthetic and very clean, the accuracy here is higher than what you would get on a real, noisy dataset.

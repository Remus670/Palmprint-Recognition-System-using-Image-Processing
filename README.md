# Fingerprint Recognition

A small project for an image processing class. The goal is to recognize fingerprints by matching them against a small database, using classical computer vision methods (SIFT and a few preprocessing tricks) instead of deep learning. I tried 4 different methods and compared them on a public dataset to see which one works best.


## Dataset

I used the Fingerprint Dataset for FVC2000_DB4_B from Kaggle:
https://www.kaggle.com/datasets/peace1019/fingerprint-dataset-for-fvc2000-db4-b?resource=download

It is the public part (Set B) of the fourth database from the Fingerprint Verification Competition 2000, generated synthetically with the SFinGE algorithm. The images are clean and artifact free, which makes it a nice dataset for testing classical methods.

After downloading, the folder structure is:

dataset_FVC2000_DB4_B/dataset/
    train_data/   <- training images
    real_data/    <- test images (used for evaluation)
    np_data/      <- not used (preprocessed numpy arrays)

Each image is named like:
101_1.tif
101_2.tif

The first number is the finger ID, and the second is the impression number.


## Methods I tried

All four methods use SIFT for keypoint detection and a brute force matcher with Lowe's ratio test (0.70).

The difference is in the preprocessing step before SIFT:

- SIFT + KNN
  grayscale + histogram equalization (baseline)

- Gaussian + SIFT
  adds a Gaussian blur to reduce noise

- Gabor + SIFT
  applies a Gabor filter to enhance the ridges (specific to fingerprints)

- LBP + SIFT
  uses Local Binary Patterns to encode the texture before SIFT

A ResNet18 reference value is also shown on the comparison chart, but it is not actually trained. It is just a hardcoded number used as a deep learning reference.


## Files

model_utils.py
    The core module.
    Contains all preprocessing functions (Gaussian, Gabor, LBP),
    SIFT descriptor extraction, the matcher with Lowe's ratio test,
    and the evaluation function.

train.py
    Runs SIFT on every image in train_data/
    and saves all descriptors into trained_model.pkl

    This is not real training in the ML sense.
    It only builds a database of descriptors.

test.py
    Loads the saved model and evaluates it on real_data/
    Prints the total accuracy and shows visual side by side keypoint matches.

main.py
    Small CLI wrapper for train / test / predict.

compare_methods.py
    Runs the full train + test pipeline for all 4 methods
    and saves a comparison chart in:
    results/method_comparison.png

predict_select.py
    Opens a file dialog so fingerprint images can be selected manually
    and predicted one by one with visual matches.

requirements.txt
    Python dependencies.


## How to run

Install dependencies:

pip install -r requirements.txt

Run training:

python train.py

Run evaluation:

python test.py

Compare all methods:

python compare_methods.py

Run demo mode:

python predict_select.py


## Notes

- The "score" shown in the output (example: 3876)
  is NOT a percentage.
  It is the total number of good SIFT matches summed across
  all training images of the predicted user.

- The "KNN" in the method names refers to:
  OpenCV knnMatch(k=2) used for Lowe's ratio test.

  It is NOT a global KNN classifier.

- Because FVC2000_DB4 is synthetic and very clean,
  the accuracy is higher than on a real noisy fingerprint dataset.

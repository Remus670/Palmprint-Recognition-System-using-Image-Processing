# Fingerprint Recognition

Fingerprint recognition project based on classical computer vision methods using SIFT descriptors and image preprocessing techniques.

## Dataset

Dataset used:
https://www.kaggle.com/datasets/peace1019/fingerprint-dataset-for-fvc2000-db4-b?resource=download

Structure:

dataset_FVC2000_DB4_B/dataset/
    train_data/
    real_data/
    np_data/

## Methods

Implemented methods:

- SIFT + KNN
- Gaussian + SIFT
- Gabor + SIFT
- LBP + SIFT

All methods use:
- SIFT keypoint descriptors
- Brute force matching
- Lowe ratio test (0.70)

## Files

- model_utils.py
  Core preprocessing, descriptor extraction, matching, and evaluation functions.

- train.py
  Builds and saves the descriptor database.

- test.py
  Evaluates the model on the test dataset.

- compare_methods.py
  Compares all implemented methods and generates a results chart.

- predict_select.py
  Manual fingerprint prediction demo with visual matching.

## Installation

pip install -r requirements.txt

## Usage

Train:
python train.py

Test:
python test.py

Compare methods:
python compare_methods.py

Demo:
python predict_select.py

## Notes

- Scores represent the number of valid SIFT matches, not percentages.
- "KNN" refers to OpenCV knnMatch(k=2), not a KNN classifier.
- Results are higher than real world scenarios because the dataset is synthetic and clean.

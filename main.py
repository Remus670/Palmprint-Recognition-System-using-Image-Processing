from pathlib import Path
import argparse

from model_utils import (
    build_training_database,
    save_database,
    load_database,
    evaluate_folder,
    predict_image,
)

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FOLDER = BASE_DIR / "dataset_FVC2000_DB4_B" / "dataset" / "train_data"
TEST_FOLDER = BASE_DIR / "dataset_FVC2000_DB4_B" / "dataset" / "real_data"
MODEL_PATH = BASE_DIR / "trained_model.pkl"


def run_train(method):
    print(f"[INFO] Training with method: {method}")
    database = build_training_database(str(TRAIN_FOLDER), method=method)
    save_database(database, str(MODEL_PATH))
    print(f"[INFO] Model saved in {MODEL_PATH}")


def run_test(method, ratio):
    print(f"[INFO] Testing with method: {method}")
    database = load_database(str(MODEL_PATH))
    accuracy, results = evaluate_folder(str(TEST_FOLDER), database, ratio=ratio, method=method)

    correct = 0
    for i, item in enumerate(results, start=1):
        print(f"Processing image {i} ...")
        print(
            f"Predict result: {item['pred_label']} | "
            f"Ground truth: {item['true_label']} | "
            f"Score: {item['score']}"
        )
        if item["correct"]:
            correct += 1

    print("-" * 60)
    print(f"Correct: {correct}/{len(results)}")
    print(f"Accuracy: {accuracy:.4f}")


def run_predict(method, image_path, ratio):
    print(f"[INFO] Predict one image with method: {method}")
    database = load_database(str(MODEL_PATH))
    label, score, details = predict_image(image_path, database, ratio=ratio, method=method)

    if label is None:
        print("Could not extract descriptors from image.")
        return

    print(f"Predicted user: {label}")
    print(f"Best score: {score}")

    print("\nTop matches:")
    details = sorted(details, key=lambda x: x[2], reverse=True)[:5]
    for path, lbl, scr in details:
        print(f"{lbl} | score={scr} | {path}")


def main():
    parser = argparse.ArgumentParser(description="Fingerprint Recognition (FVC2000_DB4_B)")
    parser.add_argument("--mode", choices=["train", "test", "predict"], required=True)
    parser.add_argument("--method", choices=["sift", "gaussian_sift", "gabor_sift", "lbp_sift"], default="sift")
    parser.add_argument("--image", type=str, help="Path to image for predict")
    parser.add_argument("--ratio", type=float, default=0.70)

    args = parser.parse_args()

    if args.mode == "train":
        run_train(args.method)
    elif args.mode == "test":
        run_test(args.method, args.ratio)
    elif args.mode == "predict":
        if not args.image:
            raise ValueError("For --mode predict you also need --image")
        run_predict(args.method, args.image, args.ratio)


if __name__ == "__main__":
    main()

from pathlib import Path
from model_utils import build_training_database, save_database

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FOLDER = BASE_DIR / "dataset_FVC2000_DB4_B" / "dataset" / "train_data"
MODEL_PATH = BASE_DIR / "trained_model.pkl"


def main():
    print("[INFO] Building training database...")
    print(f"[INFO] Training folder: {TRAIN_FOLDER}")
    database = build_training_database(str(TRAIN_FOLDER))

    print(f"[INFO] Number of training images: {len(database)}")
    save_database(database, str(MODEL_PATH))

    labels = sorted(set(item["label"] for item in database))
    print(f"[INFO] Number of users: {len(labels)}")
    print(f"[INFO] Model saved in: {MODEL_PATH}")


if __name__ == "__main__":
    main()

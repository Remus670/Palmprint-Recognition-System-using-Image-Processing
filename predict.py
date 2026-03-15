import os
import cv2
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from model_utils import (
    load_database,
    preprocess_image,
    extract_sift_descriptors,
    infer_label_from_filename,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model.pkl"
TEST_FOLDER = BASE_DIR / "dataset_FVC2000_DB4_B" / "dataset" / "real_data"

RATIO = 0.70
METHOD = "sift"


def find_best_match(query_path, database, ratio, method):
    img_query = cv2.imread(query_path)
    if img_query is None:
        return None

    processed_query = preprocess_image(img_query, method=method)
    kp_query, des_query = extract_sift_descriptors(processed_query)

    if des_query is None or len(des_query) == 0:
        return None

    bf = cv2.BFMatcher()

    label_scores = {}
    best_per_label = {}

    for item in database:
        img_train = cv2.imread(item["path"])
        if img_train is None:
            continue

        processed_train = preprocess_image(img_train, method=method)
        kp_train, des_train = extract_sift_descriptors(processed_train)

        if des_train is None or len(des_train) == 0:
            continue

        matches = bf.knnMatch(des_query, des_train, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

        score = len(good)
        label = item["label"]
        label_scores[label] = label_scores.get(label, 0) + score

        if label not in best_per_label or score > best_per_label[label]["score"]:
            best_per_label[label] = {
                "score": score,
                "train_path": item["path"],
                "train_img": img_train,
                "train_kp": kp_train,
                "good_matches": good,
            }

    if not label_scores:
        return None

    pred_label = max(label_scores, key=label_scores.get)
    pred_total_score = label_scores[pred_label]
    best = best_per_label[pred_label]

    return {
        "query_img": img_query,
        "query_kp": kp_query,
        "pred_label": pred_label,
        "pred_total_score": pred_total_score,
        "best_match_path": best["train_path"],
        "train_img": best["train_img"],
        "train_kp": best["train_kp"],
        "good_matches": best["good_matches"],
    }


def show_result(result, true_label):
    pred_label = result["pred_label"]
    is_correct = pred_label == true_label
    status = "CORRECT" if is_correct else "WRONG"
    color = (0, 255, 0) if is_correct else (0, 0, 255)

    match_img = cv2.drawMatches(
        result["query_img"],
        result["query_kp"],
        result["train_img"],
        result["train_kp"],
        result["good_matches"][:40],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    match_img = cv2.resize(match_img, (900, 450))

    cv2.putText(
        match_img,
        f"Predicted: {pred_label} | True: {true_label} | {status}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    cv2.putText(
        match_img,
        f"Score: {result['pred_total_score']} | Good matches: {len(result['good_matches'])}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    cv2.imshow("Fingerprint Prediction", match_img)


def pick_image(initial_dir):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select a fingerprint image",
        initialdir=initial_dir,
        filetypes=[
            ("Image files", "*.bmp *.png *.jpg *.jpeg *.tif *.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model does not exist: {MODEL_PATH}")
        print("[INFO] Run first: python train.py")
        return

    print("[INFO] Loading model...")
    database = load_database(str(MODEL_PATH))
    print(f"[INFO] Database loaded: {len(database)} samples\n")

    initial_dir = str(TEST_FOLDER) if TEST_FOLDER.exists() else str(BASE_DIR)

    selected = 0
    correct_count = 0

    while True:
        print("=" * 60)
        print(f"[INFO] Selection #{selected + 1} - opening file dialog...")
        chosen = pick_image(initial_dir)

        if not chosen:
            print("[INFO] No file selected. Exiting.")
            break

        true_label = infer_label_from_filename(chosen)

        print(f"[INFO] Image: {os.path.basename(chosen)}")
        print(f"[INFO] True label: {true_label}")
        print("[INFO] Predicting...")

        result = find_best_match(chosen, database, ratio=RATIO, method=METHOD)

        if result is None:
            print("[WARN] Could not predict (no descriptors).")
            continue

        is_correct = result["pred_label"] == true_label
        if is_correct:
            correct_count += 1
        status = "CORRECT" if is_correct else "WRONG"

        print(f"[RESULT] Predicted: {result['pred_label']}  ->  {status}")
        print(f"[RESULT] Total score: {result['pred_total_score']}")
        print(f"[RESULT] Best train match: {os.path.basename(result['best_match_path'])}\n")

        show_result(result, true_label)

        selected += 1

        print("[INFO] Press any key in image window to select another, ESC to exit.")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 27:
            break

    print("\n" + "=" * 60)
    print(f"[SUMMARY] Selected: {selected} | Correct: {correct_count}")
    if selected > 0:
        print(f"[SUMMARY] Accuracy: {correct_count / selected:.2%}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

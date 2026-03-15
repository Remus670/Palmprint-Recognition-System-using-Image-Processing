import os
import cv2
import pickle
import numpy as np

VALID_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def get_sift():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
        return cv2.xfeatures2d.SIFT_create()
    raise RuntimeError("SIFT is not available. Install opencv-contrib-python.")


def list_images(folder):
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(VALID_EXTENSIONS):
            files.append(path)
    return files


def infer_label_from_filename(filepath):
    name = os.path.basename(filepath)
    stem = os.path.splitext(name)[0]

    digits = ""
    for ch in stem:
        if ch.isdigit():
            digits += ch
        else:
            break

    if not digits:
        current = ""
        for ch in stem:
            if ch.isdigit():
                current += ch
            else:
                if current:
                    digits = current
                    break
                current = ""
        if not digits and current:
            digits = current

    if digits:
        return f"user_{digits.zfill(3)}"
    return stem


def preprocess_base(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def preprocess_gaussian(img):
    gray = preprocess_base(img)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def preprocess_gabor(img):
    gray = preprocess_base(img)

    kernel = cv2.getGaborKernel(
        ksize=(21, 21),
        sigma=5.0,
        theta=np.pi / 4,
        lambd=10.0,
        gamma=0.5,
        psi=0,
        ktype=cv2.CV_32F,
    )
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
    return filtered


def preprocess_lbp(img):
    gray = preprocess_base(img)
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = gray[y, x]
            code = 0
            code |= (gray[y - 1, x - 1] > center) << 7
            code |= (gray[y - 1, x] > center) << 6
            code |= (gray[y - 1, x + 1] > center) << 5
            code |= (gray[y, x + 1] > center) << 4
            code |= (gray[y + 1, x + 1] > center) << 3
            code |= (gray[y + 1, x] > center) << 2
            code |= (gray[y + 1, x - 1] > center) << 1
            code |= (gray[y, x - 1] > center) << 0
            lbp[y, x] = code

    return lbp


def preprocess_image(img, method="sift"):
    method = method.lower()

    if method == "sift":
        return preprocess_base(img)
    if method == "gaussian_sift":
        return preprocess_gaussian(img)
    if method == "gabor_sift":
        return preprocess_gabor(img)
    if method == "lbp_sift":
        return preprocess_lbp(img)

    raise ValueError(f"Unknown method: {method}")


def extract_sift_descriptors(processed_img):
    sift = get_sift()
    keypoints, descriptors = sift.detectAndCompute(processed_img, None)
    return keypoints, descriptors


def build_training_database(training_folder, method="sift"):
    database = []

    image_paths = list_images(training_folder)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {training_folder}")

    print(f"[INFO] Processing {len(image_paths)} images from {training_folder}")

    for i, path in enumerate(image_paths, 1):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Cannot read image: {path}")
            continue

        processed = preprocess_image(img, method=method)
        _, des = extract_sift_descriptors(processed)

        if des is None or len(des) == 0:
            print(f"[WARN] No descriptors for: {path}")
            continue

        label = infer_label_from_filename(path)
        database.append({
            "path": path,
            "label": label,
            "descriptors": des
        })

        if i % 50 == 0:
            print(f"[INFO]   {i}/{len(image_paths)} processed...")

    if not database:
        raise RuntimeError("Training database is empty.")

    unique_labels = sorted(set(item["label"] for item in database))
    print(f"[INFO] Total {len(database)} images, {len(unique_labels)} unique classes")
    return database


def save_database(database, output_path="trained_model.pkl"):
    with open(output_path, "wb") as f:
        pickle.dump(database, f)


def load_database(model_path="trained_model.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def match_descriptors(des_query, des_train, ratio=0.70):
    if des_query is None or des_train is None:
        return 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_query, des_train, k=2)

    good = 0
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good += 1

    return good


def predict_image(image_path, database, ratio=0.70, method="sift"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    processed = preprocess_image(img, method=method)
    _, des_query = extract_sift_descriptors(processed)

    if des_query is None or len(des_query) == 0:
        return None, 0, []

    scores = {}
    details = []

    for item in database:
        score = match_descriptors(des_query, item["descriptors"], ratio=ratio)
        label = item["label"]

        scores[label] = scores.get(label, 0) + score
        details.append((item["path"], label, score))

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    return best_label, best_score, details


def evaluate_folder(test_folder, database, ratio=0.70, method="sift"):
    image_paths = list_images(test_folder)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {test_folder}")

    total = 0
    correct = 0
    results = []

    for path in image_paths:
        true_label = infer_label_from_filename(path)
        pred_label, score, _ = predict_image(path, database, ratio=ratio, method=method)

        total += 1
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1

        results.append({
            "image": path,
            "true_label": true_label,
            "pred_label": pred_label,
            "score": score,
            "correct": is_correct
        })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results

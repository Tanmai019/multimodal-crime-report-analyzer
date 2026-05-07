"""Image analysis pipeline for the assignment's scene-photo modality."""

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
import yaml
from PIL import Image
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def get_image_dirs(dataset_location: Path) -> List[Path]:
    # Roboflow-style datasets keep images split by train/valid/test, and we
    # want inference over all of them for a simple assignment demo.
    return [
        dataset_location / "train" / "images",
        dataset_location / "valid" / "images",
        dataset_location / "test" / "images",
    ]


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    if not folder.exists():
        return []
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in exts]
    )


def classify_scene(detected_labels: Set[str]) -> str:
    # These scene labels are intentionally assignment-friendly summaries built
    # on top of the raw YOLO detections.
    if "fire" in detected_labels and "smoke" in detected_labels:
        return "Fire Scene"
    if "fire" in detected_labels:
        return "Fire Incident"
    if "smoke" in detected_labels:
        return "Smoke / Hazard"
    if "person" in detected_labels:
        return "Public Disturbance"
    if detected_labels:
        return "Incident - " + ", ".join(sorted(detected_labels))
    return "Unknown"


def format_bounding_boxes(boxes, names: Dict[int, str]) -> str:
    label_counts = Counter(names[int(box.cls[0])] for box in boxes)
    suffix_map = {
        "fire": "fire region",
        "smoke": "smoke plume",
    }
    parts = []
    for label, count in label_counts.items():
        suffix = suffix_map.get(label, f"{label} detection")
        if count > 1:
            suffix += "s"
        parts.append(f"{count} {suffix}")
    return ", ".join(parts) if parts else "None"


def download_dataset(cfg: Dict) -> Path:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key or api_key == "your_roboflow_api_key_here":
        raise ValueError(
            "ROBOFLOW_API_KEY is missing or still set to the placeholder value in .env. "
            "Set DATASET_LOCATION to a local dataset path or provide a real Roboflow API key."
        )

    rf_cfg = cfg["roboflow"]
    roboflow_client = Roboflow(api_key=api_key)
    project = roboflow_client.workspace(rf_cfg["workspace"]).project(rf_cfg["project"])
    version = project.version(int(rf_cfg["version"]))
    dataset = version.download(rf_cfg["format"])
    return Path(dataset.location)


def train_model(dataset_location: Path, cfg: Dict) -> Path:
    train_cfg = cfg["train"]
    model = YOLO(train_cfg["base_model"])
    model.train(
        data=str(dataset_location / "data.yaml"),
        epochs=int(train_cfg["epochs"]),
        imgsz=int(train_cfg["imgsz"]),
        batch=int(train_cfg["batch"]),
        name=train_cfg["run_name"],
        patience=int(train_cfg["patience"]),
        save=True,
        verbose=True,
    )

    best_weights = PROJECT_ROOT / "runs" / "detect" / train_cfg["run_name"] / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Could not find weights at {best_weights}")
    return best_weights


def extract_text(img_path: Path) -> str:
    with Image.open(img_path) as image_obj:
        raw_text = pytesseract.image_to_string(image_obj).strip().replace("\n", " ")
    return raw_text if raw_text else "None"


def process_images(model: YOLO, image_dirs: List[Path], cfg: Dict) -> pd.DataFrame:
    infer_conf = float(cfg["inference"]["confidence"])
    max_images = int(cfg["inference"].get("max_images", 0) or 0)
    results_list = []
    processed = 0

    for split_dir in image_dirs:
        for img_path in list_images(split_dir):
            if max_images > 0 and processed >= max_images:
                return pd.DataFrame(results_list)

            # Convert every image into one structured row so the integration
            # layer can treat image analysis like any other CSV modality.
            img_id = f"IMG_{img_path.stem}"
            detection = model(str(img_path), conf=infer_conf, verbose=False)[0]

            detected_labels = set()
            detected_objects_str = "None"
            bounding_box_str = "None"
            avg_conf = 0.0

            if detection.boxes is not None and len(detection.boxes) > 0:
                detected_labels = {model.names[int(box.cls[0])] for box in detection.boxes}
                detected_objects_str = ", ".join(sorted(detected_labels))
                bounding_box_str = format_bounding_boxes(detection.boxes, model.names)
                avg_conf = round(
                    sum(float(box.conf[0]) for box in detection.boxes) / len(detection.boxes),
                    2,
                )

            results_list.append(
                {
                    "Image_ID": img_id,
                    "Scene_Type": classify_scene(detected_labels),
                    "Objects_Detected": detected_objects_str,
                    "Bounding_Boxes": bounding_box_str,
                    "Text_Extracted": extract_text(img_path),
                    "Confidence": avg_conf,
                }
            )
            processed += 1

    return pd.DataFrame(results_list)


def save_sample_visualization(model: YOLO, image_dirs: List[Path], cfg: Dict) -> None:
    infer_conf = float(cfg["inference"]["confidence"])
    output_path = resolve_path(cfg["outputs"]["sample_image"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_path = None
    result_vis = None
    # Grab the first image with at least one detection so the README/demo has a
    # visual artifact without needing any manual selection.
    for split_dir in image_dirs:
        for test_path in list_images(split_dir):
            test_result = model(str(test_path), conf=infer_conf, verbose=False)[0]
            if test_result.boxes is not None and len(test_result.boxes) > 0:
                sample_path = test_path
                result_vis = test_result
                break
        if sample_path:
            break

    if not sample_path or result_vis is None:
        print("No detections found in sample. Try lowering confidence threshold.")
        return

    img_bgr = cv2.imread(str(sample_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.imshow(img_rgb)

    for box in result_vis.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        color = "red" if label == "fire" else "orange"

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        axis.add_patch(rect)
        axis.text(
            x1,
            y1 - 8,
            f"{label} {conf:.2f}",
            color=color,
            fontsize=11,
            bbox=dict(facecolor="black", alpha=0.4, pad=2),
        )

    axis.axis("off")
    plt.title(f"YOLOv8 Fire Detection - {sample_path.name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Visualization saved -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Image Analyst pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "all"],
        default="all",
        help="train: train only, infer: infer only, all: train then infer",
    )
    parser.add_argument(
        "--config",
        default="app.config.yaml",
        help="Path to config yaml file",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Override inference max images (0 = all images).",
    )
    args = parser.parse_args()

    # Environment variables let teammates swap datasets or OCR paths without
    # editing the checked-in YAML config.
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    config_path = resolve_path(args.config)
    cfg = load_config(config_path)

    if args.max_images is not None:
        cfg.setdefault("inference", {})
        cfg["inference"]["max_images"] = args.max_images

    tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    dataset_location_env = os.getenv("DATASET_LOCATION", "").strip()
    dataset_location = (
        resolve_path(dataset_location_env) if dataset_location_env else download_dataset(cfg)
    )
    if not dataset_location.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_location}")
    print(f"Dataset location: {dataset_location}")

    image_dirs = get_image_dirs(dataset_location)
    total_images = sum(len(list_images(image_dir)) for image_dir in image_dirs)
    print(f"Total images loaded: {total_images}")
    if total_images == 0:
        print(
            "Warning: no images found under train/valid/test image folders. "
            "Check DATASET_LOCATION or dataset download contents."
        )

    trained_weights = resolve_path(cfg["paths"]["weights"])
    if args.mode in {"train", "all"}:
        trained_weights = train_model(dataset_location, cfg)
        print(f"Training complete. Best weights saved at: {trained_weights}")
    elif not trained_weights.exists():
        raise FileNotFoundError(
            f"Weights not found at {trained_weights}. Run with --mode train/all first."
        )

    if args.mode in {"infer", "all"}:
        model = YOLO(str(trained_weights))
        print("Model classes:", model.names)

        df = process_images(model, image_dirs, cfg)
        output_csv = resolve_path(cfg["outputs"]["csv"])
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Processed {len(df)} images.")
        print(f"CSV saved -> {output_csv}")
        print(df.head())

        save_sample_visualization(model, image_dirs, cfg)


if __name__ == "__main__":
    main()

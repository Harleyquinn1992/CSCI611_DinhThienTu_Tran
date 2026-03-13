import os
import json
import shutil

ANNOT_DIR = "tests/mtsd_v2_fully_annotated/annotations"
SPLIT_DIR = "tests/mtsd_v2_fully_annotated/splits"

IMAGE_DIRS = ["tests/images1", "tests/images2", "tests/images3", "tests/val"]

OUT_DIR = "dataset"

os.makedirs(f"{OUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/val", exist_ok=True)


def find_image(img_id):
    for d in IMAGE_DIRS:
        path = os.path.join(d, img_id + ".jpg")
        if os.path.exists(path):
            return path
    return None


def convert_bbox(xmin, ymin, xmax, ymax, w, h):
    x_center = ((xmin + xmax) / 2) / w
    y_center = ((ymin + ymax) / 2) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return x_center, y_center, bw, bh


def process_split(split_name):

    split_file = os.path.join(SPLIT_DIR, split_name + ".txt")

    with open(split_file) as f:
        ids = [line.strip() for line in f]

    for img_id in ids:

        json_path = os.path.join(ANNOT_DIR, img_id + ".json")
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        w = data["width"]
        h = data["height"]

        yolo_lines = []

        for obj in data["objects"]:

            props = obj["properties"]
            bbox = obj["bbox"]

            if not props.get("included", False):
                continue
            if props.get("ambiguous", False):
                continue
            if props.get("out-of-frame", False):
                continue

            xmin = bbox["xmin"]
            ymin = bbox["ymin"]
            xmax = bbox["xmax"]
            ymax = bbox["ymax"]

            x, y, bw, bh = convert_bbox(xmin, ymin, xmax, ymax, w, h)

            yolo_lines.append(f"0 {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            continue

        img_path = find_image(img_id)
        if img_path is None:
            continue

        out_img = f"{OUT_DIR}/images/{split_name}/{img_id}.jpg"
        out_lbl = f"{OUT_DIR}/labels/{split_name}/{img_id}.txt"

        shutil.copy2(img_path, out_img)

        with open(out_lbl, "w") as f:
            f.write("\n".join(yolo_lines))


process_split("train")
process_split("val")

print("Dataset conversion finished.")
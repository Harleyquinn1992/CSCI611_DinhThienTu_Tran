import os
import random
import shutil

random.seed(42)

IMG_DIR = "dataset/images/train"
LBL_DIR = "dataset/labels/train"

TEST_IMG_DIR = "dataset/images/test"
TEST_LBL_DIR = "dataset/labels/test"

os.makedirs(TEST_IMG_DIR, exist_ok=True)
os.makedirs(TEST_LBL_DIR, exist_ok=True)

# take all imgages from images/train and shuffle
images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
random.shuffle(images) 

# take 15% for test
test_count = int(len(images) * 0.15)
test_images = images[:test_count]

for img_name in test_images:
    base = os.path.splitext(img_name)[0]
    lbl_name = base + ".txt"

    shutil.move(os.path.join(IMG_DIR, img_name), os.path.join(TEST_IMG_DIR, img_name))
    shutil.move(os.path.join(LBL_DIR, lbl_name), os.path.join(TEST_LBL_DIR, lbl_name))

print(f"Moved {len(test_images)} images to test set.")
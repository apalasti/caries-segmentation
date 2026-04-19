import argparse
import pathlib
import cv2
import albumentations as A
from tqdm import tqdm

def process_and_augment(img_path, lbl_path, transform, visualize):
    # Kép betöltése OpenCV-vel (BGR -> RGB)
    image = cv2.imread(str(img_path))
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes = []
    class_labels = []

    # YOLO label beolvasása: class x_center y_center width height
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x, y, w, h = map(float, parts)
                bboxes.append([x, y, w, h])
                class_labels.append(int(class_id))

    if not bboxes:
        return None

    # Augmentáció végrehajtása
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_class_labels = transformed['class_labels']

    # Vizualizáció mód (nem ment, csak megmutatja a képernyőn)
    if visualize:
        vis_image = cv2.cvtColor(transformed_image.copy(), cv2.COLOR_RGB2BGR)
        h_img, w_img, _ = vis_image.shape
        for bbox, cls in zip(transformed_bboxes, transformed_class_labels):
            xc, yc, w, h_box = bbox
            x1 = int((xc - w / 2) * w_img)
            y1 = int((yc - h_box / 2) * h_img)
            x2 = int((xc + w / 2) * w_img)
            y2 = int((yc + h_box / 2) * h_img)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, str(cls), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Augmented (Press any key for next)", vis_image)
        cv2.waitKey(0)

    return transformed_image, transformed_bboxes, transformed_class_labels

def main():
    parser = argparse.ArgumentParser(description="YOLO offline augmentáció (Kizárólag Train adatokra)")
    parser.add_argument("--data-dir", type=pathlib.Path, default="data/preprocessed_detection", help="Dataset gyökérkönyvtára")
    parser.add_argument("--visualize", action="store_true", help="Csak megjelenítés mentés nélkül")
    args = parser.parse_args()

    # ERŐS ÁLTALÁNOSÍTÓ (GENERALIZATION) AUGMENTÁCIÓK (Vízszintes tükrözés NÉLKÜL) 
    transform = A.Compose([
        # Színek és röntgen/kamera minőség szimulálása (Min/Max transformációk)
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3), # Kép kiemelése (Lokális kontraszt javítás)
        
        # Kamera életlenség és szenzor zaj szimulálása
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
        
        # Geometriai transzformációk (kifejezetten skálázás bbbox biztonsággal)
        A.SafeRotate(limit=10, p=0.4), # Enyhe forgatás biztonságosan
        A.RandomScale(scale_limit=0.2, p=0.4) # Kisebb / Nagyobb zoomolás
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

    train_images = args.data_dir / "train" / "images"
    train_labels = args.data_dir / "train" / "labels"

    img_paths = list(train_images.glob("*.jpg"))
    
    for img_path in tqdm(img_paths, desc="Augmenting train data"):
        lbl_path = train_labels / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
            
        result = process_and_augment(img_path, lbl_path, transform, args.visualize)
        if result is None or args.visualize:
            continue

        transformed_image, transformed_bboxes, transformed_class_labels = result
        
        # Mentés _aug postfix-szel
        aug_img_path = train_images / f"{img_path.stem}_aug.jpg"
        aug_lbl_path = train_labels / f"{img_path.stem}_aug.txt"

        cv2.imwrite(str(aug_img_path), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
        with open(aug_lbl_path, "w") as f:
            for bbox, cls in zip(transformed_bboxes, transformed_class_labels):
                f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Module làm giàu và mở rộng tập dữ liệu đa miền (Multi-Domain Dataset Enrichment) — Hướng 1.

Mục đích:
1. Trích xuất các khung hình đại diện từ video giao thông thực tế: data/videos/video_test_traffic.mp4.
2. Tự động hóa gán nhãn đa nhiệm Pixel-Perfect cho các frame video thông qua gt_chuan_xac.py:
   - Bounding Box 4 phân lớp phương tiện (Car, Motorcycle, Bus, Truck).
   - Mặt nạ lòng đường (Drivable Area) và vạch làn (Lane Lines) chuẩn viễn cận 3D.
   - Khử che khuất (Occlusion Masking).
3. Đa dạng hóa các biến thể tăng cường dữ liệu thích nghi:
   - Gốc (goc)
   - Phối cảnh (aug_persp với warp_boxes)
   - Ánh sáng / Độ bão hòa (aug_light)
   - Bóng râm cây cối / tòa nhà (aug_shadow)
   - Lật gương (aug_flip)
4. Mở rộng quy mô từ 15 mẫu lên 35 mẫu đa miền:
   - Train: 24 mẫu (~70%)
   - Val:   7 mẫu (~20%)
   - Test:  4 mẫu (~10%)
5. Cập nhật đồng bộ cấu hình data/dataset/dataset.yaml.
"""

import os
import sys
import shutil
from pathlib import Path
import random
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.gt_chuan_xac import tao_ground_truth_chuan_xac
from training.prepare_multidomain_dataset import warp_boxes


def apply_shadow(img):
    """Mô phỏng bóng râm cây cối / nhà cao tầng đổ xuống đường."""
    h, w = img.shape[:2]
    out = img.copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    x1 = random.randint(0, w // 3)
    y1 = random.randint(h // 3, h - 60)
    x2 = random.randint(x1 + 80, w)
    y2 = random.randint(y1 + 40, h)
    shadow_mask = np.ones((h, w), dtype=np.float32)
    shadow_mask[y1:y2, x1:x2] = random.uniform(0.60, 0.75)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * shadow_mask, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def lam_giau_dataset():
    print("=" * 75)
    print(" BẮT ĐẦU LÀM GIÀU TẬP DỮ LIỆU ĐA MIỀN TỪ VIDEO & TĂNG CƯỜNG CHUYÊN SÂU ")
    print("=" * 75)

    dataset_dir = ROOT / "data" / "dataset"
    raw_images_dir = ROOT / "data" / "images"
    video_path = ROOT / "data" / "videos" / "video_test_traffic.mp4"

    # 1. Khởi tạo và dọn sạch các thư mục
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels", "drivable", "lane"]:
            p = dataset_dir / split / sub
            p.mkdir(parents=True, exist_ok=True)
            for f in p.glob("*.*"):
                f.unlink()

    print("✓ Bước 1: Đã làm sạch và cấu trúc lại thư mục train, val, test.")

    # 2. Cập nhật dataset.yaml
    dataset_yaml = dataset_dir / "dataset.yaml"
    with open(dataset_yaml, "w", encoding="utf-8") as f:
        f.write(f"""# Cấu hình tập dữ liệu Đa miền Mở rộng cho YOLOPv2 (Enriched Multi-Domain Dataset)
path: {str(dataset_dir).replace('\\', '/')}
train: train/images
val: val/images
test: test/images

# Hệ thống 4 phân lớp phương tiện chuẩn hóa
names:
  0: car          # Ô tô con, taxi, SUV
  1: motorcycle   # Xe máy, xe tay ga (Đặc thù giao thông Việt Nam)
  2: bus          # Xe buýt, xe khách
  3: truck        # Xe tải, xe container
nc: 4

# Tác vụ phân đoạn (Segmentation)
tasks:
  - det           # Object Detection (Bounding Boxes)
  - da            # Drivable Area Segmentation (Pixel 255: lòng đường xe chạy)
  - ll            # Lane Line Segmentation (Pixel 255: vạch kẻ làn đường)
""")
    print("✓ Bước 2: Đã lưu cấu hình dataset.yaml.")

    # 3. Trích xuất frame từ video
    print(f"\n--- Trích xuất các frame đại diện từ video: {video_path.name} ---")
    video_frames = {}
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        target_indices = [0, 3, 6, 9, 12, 15, 18, 21, 24, 28]
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx in target_indices:
                vname = f"vid_traffic_f{idx:02d}.jpg"
                print(f"Trích xuất Ground Truth cho video frame {idx:02d}...")
                boxes, da, ll = tao_ground_truth_chuan_xac("Giao-Thong-Tphcm-10.jpg", frame, refine=True)
                video_frames[vname] = (frame, boxes, da, ll)
            idx += 1
        cap.release()
        print(f"✓ Đã trích xuất thành công {len(video_frames)} frames video với nhãn đa nhiệm chuẩn.")

    # 4. Trích xuất Ground Truth cho 5 ảnh tĩnh nguồn
    print("\n--- Trích xuất Ground Truth cho 5 ảnh tĩnh nguồn ---")
    anh_nguon = [
        "images (1).jpg",
        "Giao-Thong-Tphcm-10.jpg",
        "Vo-Chi-Cong-06.jpg",
        "images.jpg",
        "images (2).jpg",
    ]
    cache_static = {}
    for ten in anh_nguon:
        p = raw_images_dir / ten
        if p.exists():
            img = cv2.imread(str(p))
            print(f"Trích xuất Ground Truth cho ảnh {ten}...")
            boxes, da, ll = tao_ground_truth_chuan_xac(ten, img, refine=True)
            cache_static[ten] = (img, boxes, da, ll)

    # 5. Thiết lập danh sách phân chia 35 mẫu đa miền
    # 24 train (~70%), 7 val (~20%), 4 test (~10%)
    danh_sach_phan_chia = [
        # --- TRAIN SET (24 mẫu) ---
        ("train", "static", "images (1).jpg", "goc"),
        ("train", "static", "images (1).jpg", "aug_persp"),
        ("train", "static", "images (1).jpg", "aug_shadow"),
        ("train", "static", "images (1).jpg", "aug_flip"),
        ("train", "static", "Giao-Thong-Tphcm-10.jpg", "goc"),
        ("train", "static", "Giao-Thong-Tphcm-10.jpg", "aug_persp"),
        ("train", "static", "Giao-Thong-Tphcm-10.jpg", "aug_shadow"),
        ("train", "static", "Vo-Chi-Cong-06.jpg", "goc"),
        ("train", "static", "Vo-Chi-Cong-06.jpg", "aug_persp"),
        ("train", "static", "Vo-Chi-Cong-06.jpg", "aug_shadow"),
        ("train", "static", "Vo-Chi-Cong-06.jpg", "aug_flip"),
        ("train", "static", "images.jpg", "aug_light"),
        ("train", "static", "images.jpg", "aug_shadow"),
        ("train", "static", "images (2).jpg", "goc"),
        ("train", "static", "images (2).jpg", "aug_shadow"),
        # Video frames vào train set:
        ("train", "video", "vid_traffic_f00.jpg", "goc"),
        ("train", "video", "vid_traffic_f03.jpg", "goc"),
        ("train", "video", "vid_traffic_f06.jpg", "goc"),
        ("train", "video", "vid_traffic_f09.jpg", "goc"),
        ("train", "video", "vid_traffic_f12.jpg", "goc"),
        ("train", "video", "vid_traffic_f15.jpg", "goc"),
        ("train", "video", "vid_traffic_f18.jpg", "goc"),
        ("train", "video", "vid_traffic_f21.jpg", "goc"),
        ("train", "video", "vid_traffic_f24.jpg", "goc"),

        # --- VAL SET (7 mẫu) ---
        ("val", "static", "images (1).jpg", "aug_light"),
        ("val", "static", "Giao-Thong-Tphcm-10.jpg", "aug_light"),
        ("val", "static", "Vo-Chi-Cong-06.jpg", "aug_light"),
        ("val", "static", "images.jpg", "aug_persp"),
        ("val", "static", "images (2).jpg", "aug_persp"),
        ("val", "video", "vid_traffic_f28.jpg", "goc"),
        ("val", "video", "vid_traffic_f15.jpg", "aug_light"),

        # --- TEST SET (4 mẫu) ---
        ("test", "static", "images.jpg", "goc"),
        ("test", "static", "images (2).jpg", "aug_light"),
        ("test", "static", "Giao-Thong-Tphcm-10.jpg", "aug_flip"),
        ("test", "video", "vid_traffic_f09.jpg", "aug_light"),
    ]

    print(f"\n--- Tiến hành xử lý và lưu 35 mẫu vào data/dataset/ ---")
    count_saved = 0

    for split, nguon_loai, ten_file, kieu_aug in danh_sach_phan_chia:
        if nguon_loai == "static":
            if ten_file not in cache_static:
                continue
            img_raw, boxes_raw, da_raw, ll_raw = cache_static[ten_file]
            stem = Path(ten_file).stem
        else:
            if ten_file not in video_frames:
                continue
            img_raw, boxes_raw, da_raw, ll_raw = video_frames[ten_file]
            stem = Path(ten_file).stem

        h, w = img_raw.shape[:2]
        img_out = img_raw.copy()
        da_out = da_raw.copy()
        ll_out = ll_raw.copy()
        boxes_out = [list(b) for b in boxes_raw]

        save_name = f"{stem}_{kieu_aug}"

        # Áp dụng Augmentation theo kiểu
        if kieu_aug == "aug_light":
            hsv = cv2.cvtColor(img_out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.10, 0, 255)
            img_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        elif kieu_aug == "aug_shadow":
            img_out = apply_shadow(img_out)

        elif kieu_aug == "aug_flip":
            img_out = np.fliplr(img_out)
            da_out = np.fliplr(da_out)
            ll_out = np.fliplr(ll_out)
            for b in boxes_out:
                b[1] = 1.0 - b[1]  # xc' = 1 - xc

        elif kieu_aug == "aug_persp":
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            dx = int(w * 0.02)
            pts2 = np.float32([[dx, 0], [w - dx, 0], [0, h], [w, h]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            img_out = cv2.warpPerspective(img_out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            da_out = cv2.warpPerspective(da_out, M, (w, h), flags=cv2.INTER_NEAREST)
            ll_out = cv2.warpPerspective(ll_out, M, (w, h), flags=cv2.INTER_NEAREST)
            boxes_out = warp_boxes(boxes_out, M, w, h)

        # Lưu ảnh và mặt nạ
        cv2.imwrite(str(dataset_dir / split / "images" / f"{save_name}.jpg"), img_out)
        cv2.imwrite(str(dataset_dir / split / "drivable" / f"{save_name}.png"), da_out)
        cv2.imwrite(str(dataset_dir / split / "lane" / f"{save_name}.png"), ll_out)

        # Lưu nhãn txt
        lbl_p = dataset_dir / split / "labels" / f"{save_name}.txt"
        with open(lbl_p, "w", encoding="utf-8") as f:
            for b in boxes_out:
                f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

        count_saved += 1

    print(f"\n✓ Đã hoàn thiện toàn bộ {count_saved} mẫu dữ liệu đa miền phong phú!")
    print(f"  • Train: {len(list((dataset_dir / 'train' / 'images').glob('*.jpg')))} mẫu")
    print(f"  • Val:   {len(list((dataset_dir / 'val' / 'images').glob('*.jpg')))} mẫu")
    print(f"  • Test:  {len(list((dataset_dir / 'test' / 'images').glob('*.jpg')))} mẫu")
    print("=" * 75)


if __name__ == "__main__":
    lam_giau_dataset()

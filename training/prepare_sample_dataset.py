"""
Script chuẩn bị cấu trúc tập dữ liệu (Dataset Preparation) cho bài toán đa nhiệm YOLOPv2.

Tạo cấu trúc thư mục chuẩn:
data/dataset/
├── train/
│   ├── images/   (Ảnh RGB)
│   ├── labels/   (Tọa độ box xe theo chuẩn YOLO txt: class x y w h)
│   ├── drivable/ (Mặt nạ nhị phân vùng đường chạy được .png)
│   └── lane/     (Mặt nạ nhị phân vạch làn đường .png)
└── val/
    ├── images/
    ├── labels/
    ├── drivable/
    └── lane/
"""

import os
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))


def tao_du_lieu_mau():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / 'data' / 'dataset'
    images_source = root / 'data' / 'images'

    print("=" * 60)
    print("CHUẨN BỊ TẬP DỮ LIỆU ĐA NHIỆM CHO HUẤN LUYỆN YOLOPv2")
    print("=" * 60)

    # 1. Khởi tạo cấu trúc thư mục
    splits = ['train', 'val']
    subfolders = ['images', 'labels', 'drivable', 'lane']
    for sp in splits:
        for sub in subfolders:
            (data_dir / sp / sub).mkdir(parents=True, exist_ok=True)
    print("✓ Đã tạo cấu trúc thư mục data/dataset/train và data/dataset/val!")

    # 2. Lấy danh sách ảnh từ data/images/
    danh_sach_anh = [
        f for f in images_source.glob('*.*')
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
    ]

    if not danh_sach_anh:
        print("✗ Không tìm thấy ảnh nguồn trong data/images/!")
        return

    print(f"✓ Tìm thấy {len(danh_sach_anh)} ảnh nguồn trong data/images/.")

    # 3. Phân bổ vào train (80%) và val (20%)
    # Dùng detector YOLOPv2 để pseudo-label / khởi tạo nhãn ban đầu chất lượng cao
    weights_path = root / 'weights' / 'yolopv2.pt'
    from lane_detection.yolopv2_detector import YOLOPv2Detector
    detector = YOLOPv2Detector(duong_dan_weights=weights_path, device='cpu')

    for i, img_path in enumerate(danh_sach_anh):
        sp = 'train' if (i % 5 != 0 or len(danh_sach_anh) < 2) else 'val'
        target_img_dir = data_dir / sp / 'images'
        target_lbl_dir = data_dir / sp / 'labels'
        target_da_dir = data_dir / sp / 'drivable'
        target_ll_dir = data_dir / sp / 'lane'

        # Đọc ảnh
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Copy ảnh
        dest_img = target_img_dir / img_path.name
        shutil.copy2(str(img_path), str(dest_img))

        # Chạy detector để lấy nhãn khởi tạo
        res = detector.phat_hien(img)
        stem = img_path.stem

        # Lưu mặt nạ lòng đường (drivable)
        da_mask = (res['mat_na_duong'] > 0).astype(np.uint8) * 255
        cv2.imwrite(str(target_da_dir / f"{stem}.png"), da_mask)

        # Lưu mặt nạ vạch làn (lane)
        ll_mask = (res['mat_na_lan'] > 0).astype(np.uint8) * 255
        cv2.imwrite(str(target_ll_dir / f"{stem}.png"), ll_mask)

        # Lưu nhãn Bounding Box (YOLO format: class x_center y_center width height)
        # Class 0: Xe ô tô / xe tải / xe buýt; Class 1: Xe máy
        lbl_file = target_lbl_dir / f"{stem}.txt"
        with open(lbl_file, 'w', encoding='utf-8') as f:
            for box in res['hop_xe']:
                x1, y1, x2, y2, conf, cls_id = box[:6]
                xc = ((x1 + x2) / 2.0) / w
                yc = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                # Ghi vào file txt
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"   [{sp.upper()}] Đã xử lý và đóng gói: {img_path.name} (Boxes: {len(res['hop_xe'])})")

    # Đảm bảo có ít nhất 1 ảnh trong val
    val_imgs = list((data_dir / 'val' / 'images').glob('*.*'))
    if len(val_imgs) == 0 and len(danh_sach_anh) > 0:
        first = danh_sach_anh[0].name
        for sub in subfolders:
            src = data_dir / 'train' / sub / (first if sub == 'images' else f"{danh_sach_anh[0].stem}.{'txt' if sub == 'labels' else 'png'}")
            dst = data_dir / 'val' / sub / src.name
            if src.exists():
                shutil.copy2(str(src), str(dst))
        print("   [VAL] Đã đồng bộ mẫu kiểm thử vào thư mục val/.")

    print("\n✓ HOÀN TẤT CHUẨN BỊ TẬP DỮ LIỆU ĐA NHIỆM TẠI:")
    print(f"   {data_dir}")
    print("=" * 60)


if __name__ == '__main__':
    tao_du_lieu_mau()

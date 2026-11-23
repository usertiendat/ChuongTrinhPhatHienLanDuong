# -*- coding: utf-8 -*-
"""
Script trích xuất chọn lọc toàn bộ 3.626 mẫu có nhãn từ TuSimple dataset.zip
và chuyển đổi sang định dạng đa nhiệm chuẩn hóa cho YOLOPv2.
"""

import os
import sys
import json
import time
import random
import zipfile
import shutil
from pathlib import Path
import cv2
import numpy as np

def main():
    root_dir = Path(__file__).resolve().parent.parent
    zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (Path.home() / "Downloads" / "dataset.zip")
    out_dir = root_dir / "data" / "dataset_tusimple"
    bundle_zip = root_dir / "tusimple_colab_bundle.zip"

    if not zip_path.exists():
        print(f"Lỗi: Không tìm thấy file {zip_path}")
        return

    print("==================================================")
    print("🚀 BẮT ĐẦU TRÍCH XUẤT TOÀN BỘ 3.626 MẪU TUSIMPLE")
    print(f"Nguồn: {zip_path}")
    print(f"Đích: {out_dir}")
    print("==================================================")

    # 1. Đọc danh sách tất cả mẫu có nhãn từ các file JSON
    label_files = [
        "TUSimple/train_set/label_data_0313.json",
        "TUSimple/train_set/label_data_0531.json",
        "TUSimple/train_set/label_data_0601.json"
    ]

    all_samples = []
    t0 = time.time()

    with zipfile.ZipFile(str(zip_path), 'r') as z:
        namelist_set = set(z.namelist())

        for jf in label_files:
            if jf in namelist_set:
                with z.open(jf) as f:
                    for line in f:
                        data = json.loads(line.decode('utf-8'))
                        raw_file = data['raw_file'] # e.g. clips/0313-1/6040/20.jpg
                        img_path = f"TUSimple/train_set/{raw_file}"
                        seg_path = img_path.replace("clips/", "seg_label/").replace(".jpg", ".png")
                        
                        if img_path in namelist_set and seg_path in namelist_set:
                            all_samples.append({
                                'img_path': img_path,
                                'seg_path': seg_path,
                                'raw_file': raw_file
                            })

    print(f"✓ Đã tìm thấy {len(all_samples)} mẫu có nhãn 100% hợp lệ trong TuSimple!")

    # 2. Xáo trộn và phân chia tập dữ liệu (80% Train, 10% Val, 10% Test)
    random.seed(42)
    random.shuffle(all_samples)

    n_total = len(all_samples)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)
    n_test = n_total - n_train - n_val

    splits = {
        'train': all_samples[:n_train],
        'val': all_samples[n_train:n_train + n_val],
        'test': all_samples[n_train + n_val:]
    }

    print(f"Phân chia: Train = {len(splits['train'])} | Val = {len(splits['val'])} | Test = {len(splits['test'])}")

    # 3. Tạo cấu trúc thư mục đích
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for sp in ['train', 'val', 'test']:
        for sub in ['images', 'lane', 'drivable', 'labels']:
            (out_dir / sp / sub).mkdir(parents=True, exist_ok=True)

    # 4. Trích xuất và chuyển đổi từng mẫu
    print("\n⏳ Đang trích xuất ảnh và tạo mặt nạ làn đường / vạch kẻ...")
    processed_count = 0

    with zipfile.ZipFile(str(zip_path), 'r') as z:
        for sp, samples in splits.items():
            for idx, item in enumerate(samples):
                stem = f"tusimple_{sp}_{idx:04d}"
                
                # Đọc ảnh gốc
                img_bytes = z.read(item['img_path'])
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                # Đọc mask phân đoạn gốc
                seg_bytes = z.read(item['seg_path'])
                mask = cv2.imdecode(np.frombuffer(seg_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

                h, w = mask.shape

                # 1. Mặt nạ vạch kẻ đường (tất cả các vạch sơn > 0)
                lane_mask = (mask > 0).astype(np.uint8) * 255

                # 2. Mặt nạ làn đường xe chạy (Ego-lane giữa lane 3 và lane 4)
                drivable_mask = np.zeros((h, w), dtype=np.uint8)
                pts_left = []
                pts_right = []
                for y in range(h):
                    xs_3 = np.where(mask[y, :] == 3)[0]
                    xs_4 = np.where(mask[y, :] == 4)[0]
                    if len(xs_3) > 0:
                        pts_left.append([int(xs_3.mean()), y])
                    if len(xs_4) > 0:
                        pts_right.append([int(xs_4.mean()), y])

                if len(pts_left) >= 5 and len(pts_right) >= 5:
                    poly_pts = np.vstack([np.array(pts_left), np.array(pts_right)[::-1]])
                    cv2.fillPoly(drivable_mask, [poly_pts], 255)
                else:
                    # Fallback nếu góc quay khuất: lấy vạch trái và vạch phải gần tâm nhất
                    pts_l_alt, pts_r_alt = [], []
                    for y in range(h):
                        xs = np.where(mask[y, :] > 0)[0]
                        if len(xs) >= 2:
                            pts_l_alt.append([xs[0], y])
                            pts_r_alt.append([xs[-1], y])
                    if len(pts_l_alt) >= 5 and len(pts_r_alt) >= 5:
                        poly_pts = np.vstack([np.array(pts_l_alt), np.array(pts_r_alt)[::-1]])
                        cv2.fillPoly(drivable_mask, [poly_pts], 255)

                # Lưu ảnh và mặt nạ
                cv2.imwrite(str(out_dir / sp / 'images' / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                cv2.imwrite(str(out_dir / sp / 'lane' / f"{stem}.png"), lane_mask)
                cv2.imwrite(str(out_dir / sp / 'drivable' / f"{stem}.png"), drivable_mask)

                # Tạo nhãn text rỗng cho phát hiện xe
                (out_dir / sp / 'labels' / f"{stem}.txt").write_text("", encoding='utf-8')

                processed_count += 1
                if processed_count % 300 == 0 or processed_count == n_total:
                    elapsed = time.time() - t0
                    speed = processed_count / max(elapsed, 0.001)
                    rem = (n_total - processed_count) / max(speed, 0.001)
                    print(f"  • Tiến độ: {processed_count}/{n_total} ({processed_count/n_total*100:.1f}%) | "
                          f"Tốc độ: {speed:.1f} ảnh/s | Còn lại: {rem/60:.1f} phút")

    # 5. Tích hợp dữ liệu Việt Nam vào tập test và val
    vn_img_dir = root_dir / "data" / "dataset" / "test"
    if vn_img_dir.exists():
        print("\n🇻🇳 Đang tích hợp 5 cảnh ảnh giao thông Việt Nam vào tập test & val...")
        for sp in ['val', 'test']:
            for sub in ['images', 'lane', 'drivable', 'labels']:
                src_sub = vn_img_dir / sub
                if src_sub.exists():
                    for f in src_sub.iterdir():
                        if f.is_file():
                            shutil.copy2(f, out_dir / sp / sub / f"vn_{f.name}")

    # 6. Đóng gói toàn bộ tập dữ liệu thành tusimple_colab_bundle.zip
    print("\n📦 Đang đóng gói dữ liệu thành tệp nén tusimple_colab_bundle.zip để sẵn sàng dùng cho Colab...")
    if bundle_zip.exists():
        bundle_zip.unlink()

    # Thêm code training và dataset vào bundle
    temp_zip_dir = root_dir / "temp_bundle_dir"
    if temp_zip_dir.exists():
        shutil.rmtree(temp_zip_dir)
    temp_zip_dir.mkdir(parents=True, exist_ok=True)

    # Chép thư mục dataset
    shutil.copytree(out_dir, temp_zip_dir / "data" / "dataset")
    # Chép thư mục training, models, utils
    for d in ['training', 'models', 'utils', 'lane_detection']:
        src_d = root_dir / d
        if src_d.exists():
            shutil.copytree(src_d, temp_zip_dir / d)

    print("  Đang nén zip...")
    shutil.make_archive(str(root_dir / "tusimple_colab_bundle"), 'zip', temp_zip_dir)
    shutil.rmtree(temp_zip_dir)

    zip_size_mb = bundle_zip.stat().st_size / (1024 * 1024)
    total_time = time.time() - t0
    print("\n==================================================")
    print("🎉 HOÀN THÀNH XUẤT SẮC TRÍCH XUẤT & ĐÓNG GÓI!")
    print(f"Tổng số ảnh xử lý: {n_total} ảnh TuSimple + ảnh Việt Nam")
    print(f"Tệp nén Colab: {bundle_zip.name} ({zip_size_mb:.1f} MB)")
    print(f"Thời gian thực hiện: {total_time / 60:.1f} phút")
    print("==================================================")

if __name__ == '__main__':
    main()

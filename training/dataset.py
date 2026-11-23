# -*- coding: utf-8 -*-
"""
Module Dataset và DataLoader cho bài toán huấn luyện đa nhiệm YOLOPv2.

Đặc điểm chuẩn hóa:
1. Độ phân giải chuẩn YOLOPv2: (H=384, W=640) — tỷ lệ 16:9 native.
2. Đồng bộ hóa không gian hình học tuyệt đối:
   - Ảnh RGB, Mặt nạ lòng đường (DA), và Mặt nạ vạch làn (LL) được biến đổi đồng nhất.
   - Bounding Box được hiệu chỉnh bù đắp padding (top 12px, bottom 12px) chính xác.
3. Kỹ thuật Tăng cường dữ liệu thích nghi góc nhìn (Perspective & Environmental Augmentation):
   - Lật ngang ngẫu nhiên (Random Horizontal Flip) đồng bộ cả ảnh, masks và boxes.
   - Biến đổi độ sáng & tương phản (Color Jitter / Brightness modulation).
   - Mô phỏng bóng râm cây cối & nhà cao tầng (Shadow Simulation).
"""

import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PanopticDrivingDataset(Dataset):
    """Dataset đa nhiệm cho bài toán nhận diện phương tiện và phân đoạn làn đường."""

    def __init__(self, root_dir, split='train', target_h=384, target_w=640, augment=True):
        """
        Tham số:
            root_dir: Thư mục chứa dataset (ví dụ: 'data/dataset')
            split: 'train', 'val', hoặc 'test'
            target_h: Chiều cao ảnh đầu vào mô hình (384)
            target_w: Chiều rộng ảnh đầu vào mô hình (640)
            augment: Có áp dụng Data Augmentation hay không (chỉ bật khi split == 'train')
        """
        self.root_dir = Path(root_dir) / split
        self.target_h = target_h
        self.target_w = target_w
        self.augment = augment and (split == 'train')

        self.img_dir = self.root_dir / 'images'
        self.label_dir = self.root_dir / 'labels'
        self.drivable_dir = self.root_dir / 'drivable'
        self.lane_dir = self.root_dir / 'lane'

        dinh_dang_anh = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_files = []
        if self.img_dir.exists():
            for f in sorted(self.img_dir.iterdir()):
                if f.suffix.lower() in dinh_dang_anh:
                    self.image_files.append(f)

    def __len__(self):
        return len(self.image_files)

    def _apply_shadow_simulation(self, img):
        """Giả lập bóng râm cây cối hoặc tòa nhà cao tầng đổ xuống mặt đường."""
        h, w = img.shape[:2]
        x1 = random.randint(0, w // 2)
        y1 = random.randint(h // 3, h - 50)
        x2 = random.randint(x1 + 50, w)
        y2 = random.randint(y1 + 30, h)

        shadow_mask = np.ones((h, w), dtype=np.float32)
        shadow_factor = random.uniform(0.55, 0.75)
        shadow_mask[y1:y2, x1:x2] = shadow_factor

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * shadow_mask, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        stem = img_path.stem

        # 1. Đọc ảnh gốc
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {img_path}")
        orig_h, orig_w = img.shape[:2]

        # 2. Đọc mặt nạ vùng lái (Drivable Area Mask)
        da_path = self.drivable_dir / f"{stem}.png"
        if da_path.exists():
            da_mask = cv2.imread(str(da_path), cv2.IMREAD_GRAYSCALE)
        else:
            da_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # 3. Đọc mặt nạ vạch làn (Lane Line Mask)
        ll_path = self.lane_dir / f"{stem}.png"
        if ll_path.exists():
            ll_mask = cv2.imread(str(ll_path), cv2.IMREAD_GRAYSCALE)
        else:
            ll_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # 4. Đọc nhãn Bounding Box (YOLO txt format: class xc yc w h chuẩn hóa 0..1)
        label_path = self.label_dir / f"{stem}.txt"
        boxes = []
        if label_path.exists() and label_path.stat().st_size > 0:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        boxes.append([cls_id, xc, yc, bw, bh])
        boxes = np.array(boxes, dtype=np.float32) if len(boxes) else np.zeros((0, 5), dtype=np.float32)

        # 5. Data Augmentation (Chỉ áp dụng khi huấn luyện)
        if self.augment:
            # A. Lật ngang ngẫu nhiên (50%)
            if random.random() > 0.5:
                img = np.fliplr(img)
                da_mask = np.fliplr(da_mask)
                ll_mask = np.fliplr(ll_mask)
                if len(boxes):
                    boxes[:, 1] = 1.0 - boxes[:, 1]  # xc' = 1 - xc

            # B. Biến đổi màu & độ sáng ngẫu nhiên (Color Jitter)
            if random.random() > 0.4:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                scale_v = random.uniform(0.85, 1.15)
                scale_s = random.uniform(0.85, 1.15)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale_s, 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale_v, 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # C. Giả lập bóng râm (Shadow Simulation) (30%)
            if random.random() > 0.7:
                img = self._apply_shadow_simulation(img)

        # 6. Biến đổi hình học chuẩn xác YOLOPv2 (1280x720 -> 640x360 -> Pad 12px -> 384x640)
        # Resize về tỷ lệ chuẩn 1280x720
        img_1280 = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
        da_1280 = cv2.resize(da_mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
        ll_1280 = cv2.resize(ll_mask, (1280, 720), interpolation=cv2.INTER_NEAREST)

        # Thu phóng về 640x360
        img_360 = cv2.resize(img_1280, (640, 360), interpolation=cv2.INTER_LINEAR)
        da_360 = cv2.resize(da_1280, (640, 360), interpolation=cv2.INTER_NEAREST)
        ll_360 = cv2.resize(ll_1280, (640, 360), interpolation=cv2.INTER_NEAREST)

        # Thêm viền Padding (top=12, bottom=12) để thành đúng 384x640
        pad_top = 12
        pad_bot = 12
        pad_color = (114, 114, 114)
        img_384 = cv2.copyMakeBorder(img_360, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=pad_color)
        da_384 = cv2.copyMakeBorder(da_360, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=0)
        ll_384 = cv2.copyMakeBorder(ll_360, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=0)

        # Hiệu chỉnh tọa độ Bounding Box về hệ tọa độ (384, 640)
        # Trên ảnh 1280x720 (tương ứng vùng 640x360 nằm từ y=12 đến y=372 trong 384):
        # xc' = xc
        # yc' = (yc * 360 + 12) / 384
        # bw' = bw
        # bh' = bh * (360 / 384)
        if len(boxes):
            boxes[:, 2] = (boxes[:, 2] * 360.0 + pad_top) / 384.0
            boxes[:, 4] = boxes[:, 4] * (360.0 / 384.0)

        # 7. Nhị phân hóa nhãn mặt nạ
        da_norm = (da_384 > 0).astype(np.int64)
        ll_norm = (ll_384 > 0).astype(np.float32)

        # 8. Chuyển đổi sang PyTorch Tensor (C, H, W)
        img_rgb = img_384[:, :, ::-1].transpose(2, 0, 1)
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_rgb)).float() / 255.0

        da_tensor = torch.from_numpy(da_norm).long()
        ll_tensor = torch.from_numpy(ll_norm).float().unsqueeze(0)

        target_boxes = torch.zeros((len(boxes), 6), dtype=torch.float32)
        if len(boxes):
            target_boxes[:, 1:] = torch.from_numpy(boxes)

        return img_tensor, target_boxes, da_tensor, ll_tensor


def collate_fn(batch):
    """Gom nhóm các mẫu trong batch và gán chỉ số batch index cho bounding box."""
    imgs, boxes, da_masks, ll_masks = zip(*batch)

    imgs = torch.stack(imgs, 0)
    da_masks = torch.stack(da_masks, 0)
    ll_masks = torch.stack(ll_masks, 0)

    combined_boxes = []
    for i, box in enumerate(boxes):
        if len(box):
            box[:, 0] = i
            combined_boxes.append(box)

    if len(combined_boxes):
        combined_boxes = torch.cat(combined_boxes, 0)
    else:
        combined_boxes = torch.zeros((0, 6), dtype=torch.float32)

    return imgs, combined_boxes, da_masks, ll_masks

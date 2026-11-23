# -*- coding: utf-8 -*-
"""
Script huấn luyện và tinh chỉnh chuyên sâu (Two-Phase Fine-Tuning) mô hình YOLOPv2.

Đặc điểm theo chuẩn Giai đoạn 2:
1. Lộ trình Huấn luyện 2 Pha nghiêm ngặt (Two-Phase Fine-Tuning):
   - Pha 1 (Freeze Backbone): Khóa toàn bộ E-ELAN Backbone & FPN (layer < 105),
     chỉ huấn luyện 3 Heads đầu ra (Detection, Drivable Area, Lane Line) với lr = 1e-4.
   - Pha 2 (Unfreeze All): Mở khóa toàn bộ mạng nơ-ron, giảm tốc độ học theo Cosine Annealing.
2. Tái cân bằng Hàm mất mát đa nhiệm (MultiTaskLoss):
   - Lane Line: Focal Loss + Dice Loss (lambda_ll = 2.5).
   - Drivable Area: Cross-Entropy + Multi-Class Dice Loss (lambda_da = 1.0).
   - Detection: Objectness BCE Loss (lambda_det = 1.0).
3. Hỗ trợ đa môi trường:
   - Chạy trên máy cục bộ (CPU/GPU).
   - Chạy trên Google Colab (GPU Tesla T4 / A100).
4. Tự động lưu biểu đồ Loss curves và Metrics (IoU) chuẩn báo cáo.
5. Xuất trọng số hoàn thiện: weights/yolopv2_multidomain.pt.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from training.dataset import PanopticDrivingDataset, collate_fn
from training.losses import MultiTaskLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện chuyên sâu 2 pha YOLOPv2")
    parser.add_argument('--weights', type=str, default=str(root_dir / 'weights' / 'yolopv2.pt'),
                        help='Đường dẫn trọng số gốc')
    parser.add_argument('--data', type=str, default=str(root_dir / 'data' / 'dataset'),
                        help='Thư mục dataset chuẩn hóa')
    parser.add_argument('--epochs-p1', type=int, default=3,
                        help='Số epoch Pha 1 (Freeze Backbone)')
    parser.add_argument('--epochs-p2', type=int, default=5,
                        help='Số epoch Pha 2 (Unfreeze All)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Kích thước batch')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Tốc độ học (learning rate khởi tạo)')
    parser.add_argument('--device', type=str, default='',
                        help='Thiết bị tính toán (cpu hoặc cuda:0)')
    parser.add_argument('--output-dir', type=str, default=str(root_dir / 'runs' / 'train'),
                        help='Thư mục lưu kết quả và biểu đồ')
    return parser.parse_args()


def set_backbone_frozen(model, freeze=True):
    """
    Khóa hoặc mở khóa E-ELAN Backbone & FPN Neck (các tầng có chỉ số < 105).
    Các tầng >= 105 là 3 Heads: Detection (105), Drivable Area (106-116), Lane Line (117-125).
    """
    count_frozen = 0
    count_trainable = 0

    for name, p in model.named_parameters():
        parts = name.split('.')
        layer_idx = int(parts[1]) if (len(parts) > 1 and parts[1].isdigit()) else 0

        if freeze and layer_idx < 105:
            p.requires_grad = False
            count_frozen += 1
        else:
            p.requires_grad = True
            count_trainable += 1

    status = "ĐÃ KHÓA BACKBONE (Chỉ train 3 Heads)" if freeze else "MỞ KHÓA TOÀN BỘ MẠNG (Full Network Fine-Tuning)"
    print(f"[{status}] Số layer đóng băng: {count_frozen} | Số layer huấn luyện: {count_trainable}")


def tinh_miou(preds, targets, num_classes=2):
    """Tính mIoU cho bài toán phân đoạn nhị phân / đa lớp."""
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)
    for c in range(num_classes):
        p_c = (preds == c)
        t_c = (targets == c)
        intersection = (p_c & t_c).sum().float()
        union = (p_c | t_c).sum().float()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append((intersection / union).item())
    return float(np.mean(ious))


def train_epoch(model, loader, optimizer, criterion, device):
    """Huấn luyện 1 epoch."""
    model.train()
    running_loss = 0.0
    running_det = 0.0
    running_da = 0.0
    running_ll = 0.0

    num_batches = len(loader)
    if num_batches == 0:
        return 0, 0, 0, 0

    for i, (imgs, boxes, da_masks, ll_masks) in enumerate(loader):
        imgs = imgs.to(device)
        boxes = boxes.to(device)
        da_masks = da_masks.to(device)
        ll_masks = ll_masks.to(device)

        optimizer.zero_grad()

        # Lan truyền tiến (Forward pass)
        predictions = model(imgs)

        # Tính toán loss đa nhiệm cân bằng
        loss, loss_dict = criterion(predictions, boxes, da_masks, ll_masks)

        # Lan truyền ngược & Cập nhật trọng số
        loss.backward()
        # Gradient clipping chống bùng nổ gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running_loss += loss_dict['total_loss']
        running_det += loss_dict['loss_det']
        running_da += loss_dict['loss_da']
        running_ll += loss_dict['loss_ll']

    return (
        running_loss / num_batches,
        running_det / num_batches,
        running_da / num_batches,
        running_ll / num_batches
    )


def validate_epoch(model, loader, criterion, device):
    """Đánh giá mô hình trên tập validation."""
    model.eval()
    running_loss = 0.0
    lane_ious = []
    drivable_ious = []

    num_batches = len(loader)
    if num_batches == 0:
        return 0, 0, 0

    with torch.no_grad():
        for imgs, boxes, da_masks, ll_masks in loader:
            imgs = imgs.to(device)
            boxes = boxes.to(device)
            da_masks = da_masks.to(device)
            ll_masks = ll_masks.to(device)

            predictions = model(imgs)
            loss, loss_dict = criterion(predictions, boxes, da_masks, ll_masks)
            running_loss += loss_dict['total_loss']

            # Tính IoU vạch làn (predictions[2] đã là xác suất sigmoid từ mô hình)
            ll_pred = (predictions[2] > 0.5).int()
            lane_ious.append(tinh_miou(ll_pred.cpu(), ll_masks.cpu().int()))

            # Tính IoU mặt đường xe chạy (argmax kênh 1)
            da_pred = torch.argmax(predictions[1], dim=1)
            drivable_ious.append(tinh_miou(da_pred.cpu(), da_masks.cpu()))

    val_loss = running_loss / num_batches
    mean_lane_iou = float(np.mean(lane_ious)) if len(lane_ious) else 0.0
    mean_da_iou = float(np.mean(drivable_ious)) if len(drivable_ious) else 0.0

    return val_loss, mean_lane_iou, mean_da_iou


def ve_bieu_do_ket_qua(history, save_dir):
    """Vẽ và lưu biểu đồ Loss curves và Metrics chuẩn học thuật."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Biểu đồ Hàm Mất Mát (Multi-Task Loss Curves)
    plt.figure(figsize=(10, 6), dpi=160)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Total Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r--s', label='Total Val Loss', linewidth=2)
    plt.plot(epochs, history['loss_da'], 'g:', label='Drivable Area Loss (x1.0)', linewidth=1.5)
    plt.plot(epochs, history['loss_ll'], 'm-.', label='Lane Line Loss (x2.5)', linewidth=1.5)
    plt.plot(epochs, history['loss_det'], 'c-.', label='Detection Loss (x1.0)', linewidth=1.5)
    plt.title('Hội tụ Hàm Mất Mát Đa Nhiệm YOLOPv2 (Two-Phase Fine-Tuning)', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss Value', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    loss_chart_path = save_dir / 'training_loss_curves.png'
    plt.savefig(str(loss_chart_path), facecolor='#ffffff')
    plt.close()
    print(f"✓ Đã lưu biểu đồ Loss tại: {loss_chart_path}")

    # 2. Biểu đồ Chỉ số Đánh giá (Segmentation IoU)
    plt.figure(figsize=(10, 5), dpi=160)
    plt.plot(epochs, history['lane_iou'], 'm-^', label='Lane Line IoU', linewidth=2)
    plt.plot(epochs, history['drivable_iou'], 'g-d', label='Drivable Area mIoU', linewidth=2)
    plt.title('Chỉ số Phân Đoạn (Validation IoU Metrics)', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('mIoU Score (0.0 - 1.0)', fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    metrics_chart_path = save_dir / 'metrics_curves.png'
    plt.savefig(str(metrics_chart_path), facecolor='#ffffff')
    plt.close()
    print(f"✓ Đã lưu biểu đồ Metrics tại: {metrics_chart_path}")


def train(args):
    print("=" * 75)
    print(" BẮT ĐẦU HUẤN LUYỆN CHUYÊN SÂU 2 PHA CHO MÔ HÌNH YOLOPv2 (GIAI ĐOẠN 2) ")
    print("=" * 75)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Thiết bị tính toán:    {device}")
    print(f"Pha 1 (Freeze Backbone): {args.epochs_p1} Epochs")
    print(f"Pha 2 (Unfreeze All):   {args.epochs_p2} Epochs")
    print(f"Tổng số Epochs:         {args.epochs_p1 + args.epochs_p2} Epochs")
    print(f"Kích thước Batch:       {args.batch_size}")
    print(f"Tốc độ học (LR):        {args.lr}")
    print(f"Thư mục Dataset:        {args.data}")
    print(f"File trọng số gốc:      {Path(args.weights).name}")

    # 1. Khởi tạo Dataset & DataLoader
    train_dataset = PanopticDrivingDataset(args.data, split='train', target_h=384, target_w=640, augment=True)
    val_dataset = PanopticDrivingDataset(args.data, split='val', target_h=384, target_w=640, augment=False)

    print(f"\n✓ Mẫu tập huấn luyện (Train): {len(train_dataset)}")
    print(f"✓ Mẫu tập kiểm thử (Val):       {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("✗ Tập dữ liệu train rỗng! Vui lòng kiểm tra lại data/dataset.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 2. Tải mô hình YOLOPv2
    print(f"\nĐang tải mô hình từ: {args.weights}...")
    model = torch.jit.load(args.weights, map_location=device)

    # 3. Khởi tạo Loss Function
    criterion = MultiTaskLoss(w_det=1.0, w_da=1.0, w_ll=2.5)

    # 4. Cấu hình lịch sử huấn luyện
    history = {
        'train_loss': [],
        'val_loss': [],
        'loss_det': [],
        'loss_da': [],
        'loss_ll': [],
        'lane_iou': [],
        'drivable_iou': []
    }

    best_val_loss = float('inf')
    best_weights_path = output_dir / 'yolopv2_multidomain_best.pt'
    final_weights_path = root_dir / 'weights' / 'yolopv2_multidomain.pt'
    finetuned_alias_path = root_dir / 'weights' / 'yolopv2_finetuned.pt'

    total_epochs = args.epochs_p1 + args.epochs_p2
    cur_epoch = 0

    # =========================================================================
    # PHA 1: FREEZE BACKBONE (Chỉ huấn luyện 3 Heads)
    # =========================================================================
    if args.epochs_p1 > 0:
        print("\n" + "=" * 70)
        print(" BƯỚC VÀO PHA 1: ĐÓNG BĂNG BACKBONE — THÍCH NGHI NHANH VỚI NHÃN MỚI ")
        print("=" * 70)
        set_backbone_frozen(model, freeze=True)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=1e-4
        )

        for ep in range(1, args.epochs_p1 + 1):
            cur_epoch += 1
            t_start = time.time()
            tr_loss, tr_det, tr_da, tr_ll = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, lane_iou, da_iou = validate_epoch(model, val_loader, criterion, device)
            ep_time = time.time() - t_start

            history['train_loss'].append(tr_loss)
            history['val_loss'].append(val_loss)
            history['loss_det'].append(tr_det)
            history['loss_da'].append(tr_da)
            history['loss_ll'].append(tr_ll)
            history['lane_iou'].append(lane_iou)
            history['drivable_iou'].append(da_iou)

            print(f"[Pha 1] Epoch {cur_epoch:2d}/{total_epochs:2d} | Train: {tr_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Lane IoU: {lane_iou:.4f} | Road IoU: {da_iou:.4f} | ({ep_time:.1f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.jit.save(model, str(best_weights_path))

    # =========================================================================
    # PHA 2: UNFREEZE ALL (Huấn luyện toàn diện mạng nơ-ron với Cosine Annealing)
    # =========================================================================
    if args.epochs_p2 > 0:
        print("\n" + "=" * 70)
        print(" BƯỚC VÀO PHA 2: MỞ KHÓA TOÀN BỘ MẠNG — TINH CHỈNH ĐA NHIỆM TOÀN DIỆN ")
        print("=" * 70)
        set_backbone_frozen(model, freeze=False)

        lr_p2 = args.lr * 0.5
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_p2,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_p2,
            eta_min=1e-6
        )

        for ep in range(1, args.epochs_p2 + 1):
            cur_epoch += 1
            t_start = time.time()
            tr_loss, tr_det, tr_da, tr_ll = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, lane_iou, da_iou = validate_epoch(model, val_loader, criterion, device)
            scheduler.step()
            ep_time = time.time() - t_start

            history['train_loss'].append(tr_loss)
            history['val_loss'].append(val_loss)
            history['loss_det'].append(tr_det)
            history['loss_da'].append(tr_da)
            history['loss_ll'].append(tr_ll)
            history['lane_iou'].append(lane_iou)
            history['drivable_iou'].append(da_iou)

            print(f"[Pha 2] Epoch {cur_epoch:2d}/{total_epochs:2d} | Train: {tr_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Lane IoU: {lane_iou:.4f} | Road IoU: {da_iou:.4f} | ({ep_time:.1f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.jit.save(model, str(best_weights_path))

    # Lưu trọng số mô hình hoàn thiện cuối cùng
    torch.jit.save(model, str(final_weights_path))
    torch.jit.save(model, str(finetuned_alias_path))
    print(f"\n✓ Đã xuất trọng số tối ưu tại: {final_weights_path}")
    print(f"✓ Đã lưu bản sao tương thích tại: {finetuned_alias_path}")

    # Vẽ và lưu biểu đồ kết quả
    ve_bieu_do_ket_qua(history, output_dir)
    print("=" * 75)
    print(" HOÀN THÀNH HUẤN LUYỆN GIAI ĐOẠN 2 THÀNH CÔNG! ")
    print("=" * 75)


if __name__ == "__main__":
    args = parse_args()
    train(args)

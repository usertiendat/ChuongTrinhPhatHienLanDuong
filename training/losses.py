# -*- coding: utf-8 -*-
"""
Hàm mất mát đa nhiệm (Multi-Task Loss Balancing) cho mô hình YOLOPv2.

LƯU Ý CỐT LÕI VỀ KIẾN TRÚC YOLOPv2 TORCHSCRIPT:
1. seg (Drivable Area): Đã qua Softmax trong graph TorchScript JIT (tổng = 1.0).
   => Sử dụng NLLLoss trên log(seg) + Multi-Class Dice Loss trực tiếp trên xác suất.
2. ll (Lane Line): Đã qua Sigmoid trong graph TorchScript JIT (range [0, 1]).
   => Sử dụng Focal Loss & Dice Loss trực tiếp trên xác suất (Tránh hoàn toàn lỗi Double-Sigmoid!).
3. pred (Detection): Bounding boxes thô (logits) => Sử dụng BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossFromProbs(nn.Module):
    """Focal Loss tính trực tiếp trên xác suất (đã qua sigmoid), tránh lỗi Double-Sigmoid."""

    def __init__(self, alpha=0.30, gamma=2.0, reduction='mean'):
        super(FocalLossFromProbs, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, probs, targets):
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
        bce = -(targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs))
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLossFromProbs(nn.Module):
    """Dice Loss tối ưu hóa IoU tính trực tiếp trên xác suất."""

    def __init__(self, smooth=1e-5):
        super(DiceLossFromProbs, self).__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class MultiClassDiceLossFromProbs(nn.Module):
    """Dice Loss đa lớp tính trực tiếp trên phân phối Softmax (seg)."""

    def __init__(self, num_classes=2, smooth=1e-5):
        super(MultiClassDiceLossFromProbs, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, probs, targets):
        total_dice = 0.0
        for c in range(self.num_classes):
            p_c = probs[:, c, :, :].contiguous().view(-1)
            t_c = (targets == c).float().contiguous().view(-1)
            intersection = (p_c * t_c).sum()
            dice = (2.0 * intersection + self.smooth) / (p_c.sum() + t_c.sum() + self.smooth)
            total_dice += (1.0 - dice)
        return total_dice / self.num_classes


class MultiTaskLoss(nn.Module):
    """Bộ tính toán hàm mất mát đa nhiệm cân bằng (Balanced Multi-Task Loss)."""

    def __init__(self, w_det=1.0, w_da=1.0, w_ll=2.5):
        """
        Tham số:
            w_det: Trọng số loss bài toán phát hiện xe (1.0)
            w_da:  Trọng số loss bài toán mặt đường (1.0)
            w_ll:  Trọng số loss bài toán vạch làn (2.5 — tăng cường đặc thù vạch mỏng)
        """
        super(MultiTaskLoss, self).__init__()
        self.w_det = w_det
        self.w_da = w_da
        self.w_ll = w_ll

        # 1. Loss mặt đường (Drivable Area)
        self.da_dice = MultiClassDiceLossFromProbs(num_classes=2)

        # 2. Loss vạch làn (Lane Line)
        self.ll_focal = FocalLossFromProbs(alpha=0.30, gamma=2.0)
        self.ll_dice = DiceLossFromProbs()

    def forward(self, predictions, targets_box, targets_da, targets_ll):
        """
        Tham số:
            predictions: Tuple ([pred, grid], seg, ll) từ forward pass YOLOPv2
            targets_box: Tensor (N, 6) nhãn bounding box [batch_idx, class, x, y, w, h]
            targets_da:  Tensor (B, H, W) nhãn mặt nạ lòng đường (0 hoặc 1)
            targets_ll:  Tensor (B, 1, H, W) nhãn mặt nạ vạch làn (0 hoặc 1)

        Trả về:
            total_loss: Tổng hàm mất mát
            loss_dict: Chi tiết các thành phần loss để theo dõi quá trình hội tụ
        """
        [pred, grid], seg, ll = predictions
        device = seg.device

        # =====================================================================
        # A. LOSS PHÂN ĐOẠN MẶT ĐƯỜNG (DRIVABLE AREA LOSS)
        # =====================================================================
        if seg.shape[2:] != targets_da.shape[1:]:
            targets_da_resized = F.interpolate(
                targets_da.unsqueeze(1).float(),
                size=seg.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
        else:
            targets_da_resized = targets_da

        # seg đã là xác suất softmax => dùng nll_loss trên log(seg)
        seg_clamped = torch.clamp(seg, 1e-7, 1.0)
        loss_da_nll = F.nll_loss(torch.log(seg_clamped), targets_da_resized)
        loss_da_dice = self.da_dice(seg, targets_da_resized)
        loss_da = 0.6 * loss_da_nll + 0.4 * loss_da_dice

        # =====================================================================
        # B. LOSS PHÂN ĐOẠN VẠCH LÀN (LANE LINE LOSS)
        # =====================================================================
        if ll.shape[2:] != targets_ll.shape[2:]:
            targets_ll_resized = F.interpolate(
                targets_ll,
                size=ll.shape[2:],
                mode='nearest'
            )
        else:
            targets_ll_resized = targets_ll

        # ll đã là xác suất sigmoid => truyền trực tiếp vào Focal & Dice
        probs_ll = torch.clamp(ll, 1e-7, 1.0 - 1e-7)
        loss_ll_focal = self.ll_focal(probs_ll, targets_ll_resized)
        loss_ll_dice = self.ll_dice(probs_ll, targets_ll_resized)
        loss_ll = 0.5 * loss_ll_focal + 0.5 * loss_ll_dice

        # =====================================================================
        # C. LOSS PHÁT HIỆN XE (DETECTION LOSS)
        # =====================================================================
        loss_det = torch.tensor(0.0, device=device)
        total_elements = 0
        for p in pred:
            bs, c, ny, nx = p.shape
            obj_pred = p[:, 4:5, :, :]
            loss_det = loss_det + F.binary_cross_entropy_with_logits(
                obj_pred,
                torch.zeros_like(obj_pred)
            ) * (bs * ny * nx)
            total_elements += (bs * ny * nx)

        if total_elements > 0:
            loss_det = loss_det / total_elements

        # =====================================================================
        # TỔNG HỢP LOSS ĐA NHIỆM
        # =====================================================================
        total_loss = (
            self.w_da * loss_da +
            self.w_ll * loss_ll +
            0.1 * loss_det
        )

        loss_dict = {
            'total_loss': float(total_loss.item()),
            'loss_da': float(loss_da.item()),
            'loss_ll': float(loss_ll.item()),
            'loss_det': float(loss_det.item())
        }

        return total_loss, loss_dict

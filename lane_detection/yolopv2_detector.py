# -*- coding: utf-8 -*-
"""
Module phát hiện đa nhiệm bằng mô hình học sâu YOLOPv2.

Đặc tính nâng cấp theo yêu cầu người dùng:
1. Làn đường (Drivable Area): Tô màu XANH NƯỚC BIỂN (Deep Sky Blue [235, 145, 25] BGR) trong trẻo, nổi bật.
2. Vạch đường (Lane Line): Kẻ màu ĐỎ TƯƠI RỰC RỠ ([0, 0, 255] BGR), bắt dính cả vạch trắng và vạch vàng đường đèo.
3. Cơ chế Occlusion Masking: Tự động khoét rỗng thân phương tiện (xe tải, ô tô, xe máy, xe buýt) và người đi đường,
   đảm bảo màu xanh nước biển bao quanh mặt đường nhưng KHÔNG TÔ ĐÈ LÊN XE VÀ NGƯỜI.
4. Bộ lọc sinh thái Anti-Vegetation (ExG): Loại trừ triệt để cây cối ven đường, dải phân cách cây xanh và thảm cỏ đèo dốc.
"""

import time
from pathlib import Path
import cv2
import numpy as np
import torch

from models.yolopv2_utils import (
    letterbox, split_for_trace_model, non_max_suppression,
    scale_coords, driving_area_mask, lane_line_mask
)


class YOLOPv2Detector:
    """Bộ nhận diện làn đường và phương tiện giao thông bằng mô hình YOLOPv2 chuẩn hóa."""

    def __init__(self, duong_dan_weights=None, device='cpu', conf_thres=0.25, iou_thres=0.45):
        """Khởi tạo mô hình YOLOPv2.

        Tham số:
            duong_dan_weights: Đường dẫn đến file weights (.pt).
            device: Thiết bị tính toán ('cpu' hoặc 'cuda:0').
            conf_thres: Ngưỡng tự tin phát hiện xe (confidence threshold).
            iou_thres: Ngưỡng NMS IoU.
        """
        self.device = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Xác định đường dẫn weights
        if duong_dan_weights is None:
            thu_muc_goc = Path(__file__).resolve().parent.parent
            duong_dan_weights = thu_muc_goc / 'weights' / 'yolopv2.pt'

        self.duong_dan_weights = Path(duong_dan_weights)
        if not self.duong_dan_weights.exists():
            raise FileNotFoundError(f"Không tìm thấy file weights YOLOPv2 tại: {self.duong_dan_weights}")

        # Tải mô hình TorchScript JIT
        self.model = torch.jit.load(str(self.duong_dan_weights), map_location=self.device)
        self.model.eval()

        # Kích thước chuẩn huấn luyện YOLOPv2 (1280x720 -> letterbox 640)
        self.target_size = (1280, 720)
        self.img_size = 640
        self.stride = 32

    def _loc_co_va_cay(self, img_bgr):
        """Khử cỏ cây và thảm thực vật bằng chỉ số Excess Green (ExG = 2G - R - B)."""
        b = img_bgr[:, :, 0].astype(np.float32)
        g = img_bgr[:, :, 1].astype(np.float32)
        r = img_bgr[:, :, 2].astype(np.float32)
        exg = 2.0 * g - r - b
        mask_cay_co = (exg > 16) & (g > 45)
        return mask_cay_co

    def _phat_hien_vach_vang(self, img_bgr):
        """Phát hiện vạch vàng tim đường trong không gian màu HSV."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([12, 60, 85])
        upper_yellow = np.array([38, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        return (mask_yellow > 0).astype(np.uint8)

    def phat_hien(self, anh_bgr, tinh_toan_xe=True, khoet_phuong_tien=True, loc_cay_co=True):
        """Chạy pipeline nhận diện toàn cảnh trên 1 ảnh.

        Tham số:
            anh_bgr: Ảnh gốc BGR.
            tinh_toan_xe: Có chạy NMS phát hiện xe không (Mặc định True để phục vụ khoét thân xe).
            khoet_phuong_tien: Tự động khoét rỗng thân xe/người khỏi mặt nạ lòng đường (Mặc định True).
            loc_cay_co: Tự động khử cây cỏ khỏi mặt nạ lòng đường (Mặc định True).

        Trả về:
            ket_qua: Dict chứa mặt nạ làn, mặt nạ đường, bounding box và thời gian.
        """
        thoi_gian_bat_dau = time.time()
        orig_h, orig_w = anh_bgr.shape[:2]

        # 1. Resize chuẩn về 1280x720 và letterbox 640 chuẩn của YOLOPv2
        img0_1280 = cv2.resize(anh_bgr, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_lb = letterbox(img0_1280, self.img_size, stride=self.stride)[0]

        # 2. Chuyển đổi BGR sang RGB và tensor CHW chuẩn hóa
        img_rgb = img_lb[:, :, ::-1].transpose(2, 0, 1)
        img_rgb = np.ascontiguousarray(img_rgb)
        tensor_anh = torch.from_numpy(img_rgb).to(self.device).float() / 255.0

        if tensor_anh.ndimension() == 3:
            tensor_anh = tensor_anh.unsqueeze(0)

        # 3. Lan truyền tiến (Inference)
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = self.model(tensor_anh)

            # 4. Hậu xử lý phát hiện xe (Detection)
            hop_xe = np.zeros((0, 6))
            if tinh_toan_xe or khoet_phuong_tien:
                pred_split = split_for_trace_model(pred, anchor_grid)
                hop_xe_list = non_max_suppression(
                    pred_split,
                    conf_thres=self.conf_thres,
                    iou_thres=self.iou_thres
                )
                if len(hop_xe_list) > 0 and len(hop_xe_list[0]) > 0:
                    box_t = hop_xe_list[0].clone()
                    box_t[:, :4] = scale_coords(img_lb.shape[:2], box_t[:, :4], (orig_h, orig_w)).round()
                    hop_xe = box_t.cpu().numpy()

            # 5. Hậu xử lý mặt nạ làn đường & vùng lái bằng hàm chuẩn YOLOPv2
            da_mask_1280 = driving_area_mask(seg)
            ll_mask_1280 = lane_line_mask(ll)

            # Resize mặt nạ về kích thước ảnh gốc ban đầu
            if (orig_w, orig_h) != self.target_size:
                da_mask_goc = cv2.resize(da_mask_1280.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                ll_mask_goc = cv2.resize(ll_mask_1280.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            else:
                da_mask_goc = da_mask_1280.copy()
                ll_mask_goc = ll_mask_1280.copy()

        # 6. Bổ sung vạch vàng nếu phát hiện (ví dụ đường đèo uốn lượn)
        vach_vang = self._phat_hien_vach_vang(anh_bgr)
        # Chỉ bổ sung vạch vàng nếu nằm trong khu vực gần vùng đường
        if np.sum(vach_vang) > 100:
            da_vung = cv2.dilate(da_mask_goc, np.ones((25, 25), np.uint8))
            vach_vang_hop_le = cv2.bitwise_and(vach_vang, vach_vang, mask=da_vung)
            ll_mask_goc = cv2.bitwise_or(ll_mask_goc, vach_vang_hop_le)

        # 7. Bộ lọc sinh thái Anti-Vegetation: Khử cỏ cây khỏi lòng đường
        if loc_cay_co:
            mask_cay = self._loc_co_va_cay(anh_bgr)
            da_mask_goc[mask_cay] = 0

            # Đối với ảnh có độ bão hòa cao (như cỏ sườn đèo):
            hsv = cv2.cvtColor(anh_bgr, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            da_mask_goc[(sat > 70) & (mask_cay)] = 0

        # 8. Cơ chế Occlusion Masking: Khoét rỗng thân xe và người khỏi mặt nạ lòng đường
        if khoet_phuong_tien and len(hop_xe) > 0:
            for box in hop_xe:
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                h_box = y2 - y1
                # Khoét thân xe nhưng chừa lại 5% đáy bánh xe để tiếp giáp tự nhiên với mặt đường
                y2_crop = max(y1, int(y2 - 0.05 * h_box))
                da_mask_goc[y1:y2_crop, x1:x2] = 0

        thoi_gian_thuc_thi = (time.time() - thoi_gian_bat_dau) * 1000

        w_nua = orig_w // 2
        phat_hien_trai = bool(np.sum(ll_mask_goc[:, :w_nua]) > 30) if ll_mask_goc is not None else False
        phat_hien_phai = bool(np.sum(ll_mask_goc[:, w_nua:]) > 30) if ll_mask_goc is not None else False

        return {
            'hop_xe': hop_xe,
            'mat_na_lan': ll_mask_goc,
            'mat_na_duong': da_mask_goc,
            'thoi_gian_ms': thoi_gian_thuc_thi,
            'so_luong_xe': len(hop_xe),
            'phuong_phap': 'YOLOPv2 (Học sâu đa nhiệm)',
            'phat_hien_trai': phat_hien_trai,
            'phat_hien_phai': phat_hien_phai
        }

    def ve_ket_qua(self, anh_bgr, ket_qua, ve_xe=False, ve_lan=True, ve_vung_lai=True):
        """Vẽ toàn bộ kết quả phát hiện lên ảnh gốc theo quy chuẩn nâng cấp mới.

        Quy chuẩn hiển thị:
        - Làn đường (Drivable Area): Đánh dấu màu XANH NƯỚC BIỂN (Deep Sky Blue [235, 145, 25] BGR) trong suốt 45%.
        - Vạch đường (Lane Line): Kẻ màu ĐỎ TƯƠI RỰC RỠ ([0, 0, 255] BGR) sắc nét.
        - Không vẽ đè lên thân xe và người (Nhờ cơ chế Occlusion Masking tự động).
        - Xe cộ: ve_xe=False mặc định theo yêu cầu tinh gọn trọng tâm đề tài.
        """
        anh_ve = anh_bgr.copy()

        da_mask = ket_qua.get('mat_na_duong')
        ll_mask = ket_qua.get('mat_na_lan')

        # 1. Vẽ vùng mặt đường xe chạy: MÀU XANH NƯỚC BIỂN (Deep Sky Blue)
        if ve_vung_lai and da_mask is not None:
            lop_mau_duong = np.zeros_like(anh_bgr)
            if ve_lan and ll_mask is not None:
                # Không tô xanh lên vị trí có vạch làn để giữ nguyên vẹn sắc đỏ rực rỡ
                mask_da_rieng = (da_mask == 1) & (ll_mask == 0)
            else:
                mask_da_rieng = (da_mask == 1)

            # Màu xanh nước biển (BGR): Blue=235, Green=145, Red=25 (Tươi sáng, trong trẻo như biển hồ)
            lop_mau_duong[mask_da_rieng] = [235, 145, 25]
            anh_ve = cv2.addWeighted(anh_ve, 1.0, lop_mau_duong, 0.45, 0)

        # 2. Vẽ vạch làn đường: MÀU ĐỎ TƯƠI RỰC RỠ (Pure Red)
        if ve_lan and ll_mask is not None:
            # Dãn nhẹ 1 pixel (dilate) để vạch kẻ rõ nét và nổi bật trên ảnh độ phân giải cao
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            ll_mask_dilated = cv2.dilate(ll_mask.astype(np.uint8), kernel, iterations=1)
            lop_mau_lan = np.zeros_like(anh_bgr)
            lop_mau_lan[ll_mask_dilated == 1] = [0, 0, 255]  # Vạch đường màu đỏ tươi (BGR)
            anh_ve = cv2.addWeighted(anh_ve, 1.0, lop_mau_lan, 0.95, 0)

        # 3. Tùy chọn vẽ bounding box các phương tiện giao thông (Mặc định tắt)
        if ve_xe and len(ket_qua.get('hop_xe', [])) > 0:
            for box in ket_qua['hop_xe']:
                x1, y1, x2, y2, conf = box[:5]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                mau_hop = (0, 215, 255)
                cv2.rectangle(anh_ve, (x1, y1), (x2, y2), mau_hop, 2)
                ten_nhan = f"Xe {conf:.2f}"
                (w_nhan, h_nhan), _ = cv2.getTextSize(ten_nhan, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(anh_ve, (x1, max(0, y1 - h_nhan - 4)), (x1 + w_nhan, max(0, y1)), mau_hop, -1)
                cv2.putText(anh_ve, ten_nhan, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return anh_ve

# -*- coding: utf-8 -*-
"""
Module Xử lý ảnh và video bằng mô hình AI Học sâu Đa nhiệm YOLOPv2.

Đặc tính:
- Phát hiện vạch làn đường (Lane Line) và mặt đường xe chạy (Drivable Area).
- Tối ưu hóa suy luận CPU bằng cơ chế Frame-Skipping cho video.
- Tùy chỉnh bật/tắt hiển thị từng thành phần và cơ chế khoét thân xe/người.
- Dừng xử lý an toàn và giải phóng tài nguyên đúng chuẩn.
"""

import cv2
import time
from pathlib import Path
import numpy as np


class XuLyVideo:
    """Xử lý ảnh và video chuyên sâu bằng mô hình AI YOLOPv2."""

    def __init__(self):
        """Khởi tạo bộ xử lý YOLOPv2."""
        self.bo_phat_hien_yolopv2 = None
        self.duong_dan_weights_hien_tai = None

        self.cap = None          # cv2.VideoCapture
        self.writer = None       # cv2.VideoWriter
        self.dang_chay = False

    def lay_bo_phat_hien_yolopv2(self, duong_dan_weights=None):
        """Lấy hoặc khởi tạo bộ nhận diện YOLOPv2.

        Tham số:
            duong_dan_weights: File weights .pt tùy chọn. Nếu None, dùng mặc định weights/yolopv2.pt.
        """
        if duong_dan_weights is None:
            thu_muc_goc = Path(__file__).resolve().parent.parent
            duong_dan_weights = thu_muc_goc / 'weights' / 'yolopv2.pt'
        else:
            duong_dan_weights = Path(duong_dan_weights)

        duong_dan_str = str(duong_dan_weights)
        if self.bo_phat_hien_yolopv2 is None or self.duong_dan_weights_hien_tai != duong_dan_str:
            from lane_detection.yolopv2_detector import YOLOPv2Detector
            self.bo_phat_hien_yolopv2 = YOLOPv2Detector(duong_dan_weights=duong_dan_weights)
            self.duong_dan_weights_hien_tai = duong_dan_str

        return self.bo_phat_hien_yolopv2

    def mo_video(self, duong_dan_video):
        """Mở video và đọc thông tin cơ bản.

        Tham số:
            duong_dan_video: Đường dẫn file video.

        Trả về:
            (thanh_cong, thong_tin): thanh_cong=True nếu mở được,
            thong_tin chứa fps, tổng_frame, chiều_rộng, chiều_cao.
        """
        self.cap = cv2.VideoCapture(str(duong_dan_video))

        if not self.cap.isOpened():
            return False, {"loi": f"Không mở được video: {duong_dan_video}"}

        thong_tin = {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'tong_frame': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'chieu_rong': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'chieu_cao': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

        # Xử lý FPS <= 0 (file bị lỗi metadata)
        if thong_tin['fps'] <= 0:
            thong_tin['fps'] = 30.0

        return True, thong_tin

    def xu_ly_mot_anh(self, anh, ve_xe=False, ve_lan=True, ve_vung_lai=True,
                      duong_dan_weights=None, khoet_phuong_tien=True, **kwargs):
        """Xử lý một ảnh đơn bằng YOLOPv2.

        Tham số:
            anh: Ảnh đầu vào BGR.
            ve_xe: Có vẽ hộp xe không (Mặc định False).
            ve_lan: Có vẽ vạch làn không (Mặc định True, màu đỏ tươi).
            ve_vung_lai: Có phủ vùng mặt đường không (Mặc định True, màu xanh nước biển).
            duong_dan_weights: Đường dẫn weights tùy chọn.
            khoet_phuong_tien: Không tô đè màu lên thân xe và người tham gia giao thông.

        Trả về:
            (anh_ket_qua, thong_tin): Ảnh đã vẽ + dictionary thông tin phát hiện.
        """
        detector = self.lay_bo_phat_hien_yolopv2(duong_dan_weights)
        ket_qua = detector.phat_hien(
            anh,
            tinh_toan_xe=ve_xe or khoet_phuong_tien,
            khoet_phuong_tien=khoet_phuong_tien
        )
        anh_ve = detector.ve_ket_qua(anh, ket_qua, ve_xe=ve_xe, ve_lan=ve_lan, ve_vung_lai=ve_vung_lai)
        return anh_ve, ket_qua

    def xu_ly_video(self, duong_dan_video, duong_dan_ket_qua,
                    ham_cap_nhat_tien_trinh=None, su_kien_dung=None,
                    ve_xe=False, ve_lan=True, ve_vung_lai=True,
                    frame_skip=2, duong_dan_weights=None, khoet_phuong_tien=True, **kwargs):
        """Xử lý toàn bộ video bằng mô hình YOLOPv2.

        Tham số:
            duong_dan_video: Đường dẫn video đầu vào.
            duong_dan_ket_qua: Đường dẫn video kết quả.
            ham_cap_nhat_tien_trinh: Callback(frame_hien_tai, tong_frame, fps, thong_tin, frame_ket_qua).
            su_kien_dung: threading.Event dừng xử lý.
            ve_xe: Có vẽ xe không.
            ve_lan: Có vẽ vạch làn không.
            ve_vung_lai: Có vẽ vùng mặt đường không.
            frame_skip: Bỏ qua số frame suy luận để tối ưu FPS trên CPU (mặc định 2: suy luận frame 1, 3, 5...).
            duong_dan_weights: Tùy chọn đường dẫn weights YOLOPv2.
            khoet_phuong_tien: Không tô đè màu lên thân xe và người tham gia giao thông.

        Trả về:
            (thanh_cong, thong_bao): Kết quả xử lý.
        """
        thanh_cong, thong_tin_video = self.mo_video(duong_dan_video)
        if not thanh_cong:
            return False, thong_tin_video.get('loi', 'Lỗi không xác định')

        fps = thong_tin_video['fps']
        tong_frame = thong_tin_video['tong_frame']
        chieu_rong = thong_tin_video['chieu_rong']
        chieu_cao = thong_tin_video['chieu_cao']

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(duong_dan_ket_qua), fourcc, fps, (chieu_rong, chieu_cao)
        )

        if not self.writer.isOpened():
            self.giai_phong()
            return False, "Không tạo được file video kết quả."

        self.dang_chay = True
        frame_hien_tai = 0
        thoi_gian_bat_dau = time.time()

        detector = self.lay_bo_phat_hien_yolopv2(duong_dan_weights)
        ket_qua_truoc = None

        try:
            while self.dang_chay:
                if su_kien_dung is not None and su_kien_dung.is_set():
                    break

                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_hien_tai += 1
                thoi_gian_frame = time.time()

                # Cơ chế Frame-Skipping trên CPU
                can_suy_luan = (frame_skip <= 1) or (frame_hien_tai % frame_skip == 1) or (ket_qua_truoc is None)

                if can_suy_luan:
                    ket_qua = detector.phat_hien(
                        frame,
                        tinh_toan_xe=ve_xe or khoet_phuong_tien,
                        khoet_phuong_tien=khoet_phuong_tien
                    )
                    ket_qua_truoc = ket_qua
                else:
                    ket_qua = ket_qua_truoc.copy()
                    ket_qua['thoi_gian_ms'] = 0.0

                frame_ket_qua = detector.ve_ket_qua(
                    frame, ket_qua, ve_xe=ve_xe, ve_lan=ve_lan, ve_vung_lai=ve_vung_lai
                )
                thong_tin = ket_qua

                self.writer.write(frame_ket_qua)

                thoi_gian_xu_ly = time.time() - thoi_gian_frame
                fps_thuc_te = 1.0 / max(thoi_gian_xu_ly, 0.001)

                if ham_cap_nhat_tien_trinh is not None:
                    ham_cap_nhat_tien_trinh(
                        frame_hien_tai, tong_frame, fps_thuc_te,
                        thong_tin, frame_ket_qua
                    )

            thoi_gian_tong = time.time() - thoi_gian_bat_dau
            fps_trung_binh = frame_hien_tai / max(thoi_gian_tong, 0.001)

            thong_bao = (
                f"Hoàn thành! {frame_hien_tai}/{tong_frame} frame "
                f"trong {thoi_gian_tong:.1f}s "
                f"(FPS trung bình: {fps_trung_binh:.1f})"
            )

            return True, thong_bao

        except Exception as e:
            return False, f"Lỗi khi xử lý video: {str(e)}"

        finally:
            self.giai_phong()

    def giai_phong(self):
        """Giải phóng tài nguyên video."""
        self.dang_chay = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None

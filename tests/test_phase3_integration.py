"""
Kiểm thử tích hợp Giai đoạn 3:
- Kiểm tra xử lý ảnh bằng YOLOPv2 và OpenCV
- Kiểm tra xử lý video bằng YOLOPv2 và OpenCV
- Kiểm tra khởi tạo GUI
"""
import sys
import os
from pathlib import Path
import cv2

# Đảm bảo import đúng
duong_dan_goc = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(duong_dan_goc))

from lane_detection.video_processor import XuLyVideo

def test_anh():
    print("\n--- 1. Kiểm thử xử lý ảnh ---")
    xu_ly = XuLyVideo()
    anh_path = duong_dan_goc / "data" / "images" / "Giao-Thong-Tphcm-10.jpg"
    anh = cv2.imread(str(anh_path))
    assert anh is not None, "Không đọc được ảnh test"

    # Test YOLOPv2
    print("Testing YOLOPv2 trên ảnh...")
    anh_kq_ai, info_ai = xu_ly.xu_ly_mot_anh(anh, ve_xe=True, ve_lan=True, ve_vung_lai=True)
    print(f"  AI YOLOPv2: Phương pháp = {info_ai.get('phuong_phap')}, Số xe = {info_ai.get('so_luong_xe')}, Thời gian = {info_ai.get('thoi_gian_ms'):.1f}ms")
    assert anh_kq_ai.shape == anh.shape, "Ảnh kết quả AI sai kích thước"
    print("✓ Test ảnh YOLOPv2 thành công!")

def test_video():
    print("\n--- 2. Kiểm thử xử lý video ---")
    xu_ly = XuLyVideo()
    video_in = duong_dan_goc / "data" / "videos" / "video_test_traffic.mp4"
    video_out = duong_dan_goc / "results" / "videos" / "test_out_yolopv2.mp4"
    video_out.parent.mkdir(parents=True, exist_ok=True)

    tien_trinh_records = []
    def callback(frame_ht, tong, fps, thong_tin, frame_kq):
        tien_trinh_records.append((frame_ht, fps, thong_tin.get('so_luong_xe', 0)))

    print("Testing YOLOPv2 trên video với Frame-Skipping...")
    ok, msg = xu_ly.xu_ly_video(
        video_in,
        video_out,
        ham_cap_nhat_tien_trinh=callback,
        phuong_phap='yolopv2',
        ve_xe=True,
        ve_lan=True,
        ve_vung_lai=True,
        frame_skip=2
    )
    print(f"  Kết quả: ok={ok}, msg={msg}")
    print(f"  Số frame đã gọi callback: {len(tien_trinh_records)}")
    assert ok, f"Xử lý video thất bại: {msg}"
    assert video_out.exists() and video_out.stat().st_size > 0, "File video kết quả không hợp lệ"
    print("✓ Test video YOLOPv2 thành công!")

def test_gui_class():
    print("\n--- 3. Kiểm thử khởi tạo lớp GUI Tkinter ---")
    import tkinter as tk
    from app.gui import UngDungPhatHienLan

    root = tk.Tk()
    root.withdraw() # Ẩn cửa sổ
    app = UngDungPhatHienLan(root)
    print(f"  GUI khởi tạo thành công với tiêu đề: {root.title()}")
    print(f"  Chế độ mặc định: {app.che_do.get()}")
    print(f"  Phương pháp mặc định: {app.phuong_phap_chon.get()}")
    print(f"  Checkbox hiển thị: Xe={app.hien_xe.get()}, Làn={app.hien_lan.get()}, Đường={app.hien_duong.get()}")
    root.destroy()
    print("✓ Test GUI khởi tạo thành công!")

if __name__ == '__main__':
    test_anh()
    test_video()
    test_gui_class()
    print("\n==========================================")
    print("🎉 TẤT CẢ KIỂM THỬ GIAI ĐOẠN 3 ĐÃ VƯỢT QUA!")
    print("==========================================")

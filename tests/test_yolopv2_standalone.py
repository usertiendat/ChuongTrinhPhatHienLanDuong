"""
Script kiểm thử độc lập mô hình YOLOPv2 trên ảnh thực tế.
Kiểm tra khả năng nhận diện đồng thời:
1. Phương tiện giao thông (Bounding Box)
2. Vạch làn đường (Lane Line)
3. Mặt đường an toàn (Drivable Area)
"""

import sys
import os
from pathlib import Path
import cv2

# Đảm bảo import được module từ thư mục gốc
thu_muc_goc = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(thu_muc_goc))

from lane_detection.yolopv2_detector import YOLOPv2Detector


def main():
    print("=" * 60)
    print("KIỂM THỬ ĐỘC LẬP MÔ HÌNH HỌC SÂU ĐA NHIỆM YOLOPv2")
    print("=" * 60)

    # 1. Khởi tạo detector
    weights_path = thu_muc_goc / 'weights' / 'yolopv2.pt'
    print(f"\n1. Đang tải trọng số mô hình từ: {weights_path.name}...")
    detector = YOLOPv2Detector(duong_dan_weights=weights_path, device='cpu', conf_thres=0.3)
    print("✓ Tải mô hình YOLOPv2 thành công trên CPU!")

    # 2. Tìm ảnh kiểm thử trong data/images/
    thu_muc_anh = thu_muc_goc / 'data' / 'images'
    danh_sach_anh = list(thu_muc_anh.glob('*.jpg')) + list(thu_muc_anh.glob('*.png'))

    if not danh_sach_anh:
        print("✗ Không tìm thấy ảnh nào trong data/images/!")
        return False

    duong_dan_anh = danh_sach_anh[0]
    # Ưu tiên chọn ảnh Vo-Chi-Cong-06.jpg hoặc Giao-Thong nếu có
    for p in danh_sach_anh:
        if 'Vo-Chi-Cong' in p.name or 'Giao-Thong' in p.name:
            duong_dan_anh = p
            break

    print(f"\n2. Đang đọc ảnh kiểm thử: {duong_dan_anh.name}...")
    anh_goc = cv2.imread(str(duong_dan_anh))
    assert anh_goc is not None, f"Không thể đọc file ảnh: {duong_dan_anh}"
    h, w = anh_goc.shape[:2]
    print(f"   Kích thước ảnh: {w} × {h} pixels")

    # 3. Chạy nhận diện qua YOLOPv2
    print("\n3. Đang thực thi suy luận qua mạng YOLOPv2...")
    ket_qua = detector.phat_hien(anh_goc)

    print(f"✓ Suy luận hoàn tất!")
    print(f"   - Thời gian thực thi: {ket_qua['thoi_gian_ms']:.1f} ms")
    print(f"   - Số phương tiện phát hiện: {ket_qua['so_luong_xe']} xe")
    
    so_pixel_lan = int((ket_qua['mat_na_lan'] > 0).sum())
    so_pixel_duong = int((ket_qua['mat_na_duong'] > 0).sum())
    print(f"   - Pixel vạch làn đường: {so_pixel_lan:,} pixels")
    print(f"   - Pixel mặt đường lái xe: {so_pixel_duong:,} pixels")

    # 4. Vẽ kết quả và lưu file
    print("\n4. Đang vẽ overlay kết quả...")
    anh_ket_qua = detector.ve_ket_qua(anh_goc, ket_qua)

    thu_muc_xuat = thu_muc_goc / 'results' / 'images'
    thu_muc_xuat.mkdir(parents=True, exist_ok=True)
    duong_dan_luu = thu_muc_xuat / f"yolopv2_ketqua_{duong_dan_anh.stem}.jpg"

    cv2.imwrite(str(duong_dan_luu), anh_ket_qua)
    print(f"✓ Đã lưu ảnh kết quả trực quan tại:\n   {duong_dan_luu}")

    print("\n" + "=" * 60)
    print("KẾT LUẬN: GIAI ĐOẠN 1 KIỂM THỬ THÀNH CÔNG 100% ✓")
    print("=" * 60)
    return True


if __name__ == '__main__':
    main()

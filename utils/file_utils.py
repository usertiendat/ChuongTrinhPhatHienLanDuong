"""
Tiện ích xử lý file — kiểm tra đường dẫn, định dạng và thao tác hỗ trợ.
"""

import os
from pathlib import Path

# Các định dạng ảnh được hỗ trợ
DINH_DANG_ANH = {'.jpg', '.jpeg', '.png', '.bmp'}

# Các định dạng video được hỗ trợ
DINH_DANG_VIDEO = {'.mp4', '.avi', '.mov'}


def kiem_tra_file_anh(duong_dan):
    """Kiểm tra file ảnh có hợp lệ hay không.
    
    Tham số:
        duong_dan: Đường dẫn đến file ảnh.
        
    Trả về:
        (hop_le, thong_bao): hop_le=True nếu file hợp lệ, 
        thong_bao chứa lý do nếu không hợp lệ.
    """
    if not duong_dan:
        return False, "Chưa chọn file."
    
    duong_dan = Path(duong_dan)
    
    if not duong_dan.exists():
        return False, f"File không tồn tại: {duong_dan.name}"
    
    if not duong_dan.is_file():
        return False, f"Đường dẫn không phải file: {duong_dan.name}"
    
    phan_mo_rong = duong_dan.suffix.lower()
    if phan_mo_rong not in DINH_DANG_ANH:
        dinh_dang_ho_tro = ', '.join(sorted(DINH_DANG_ANH))
        return False, (f"Định dạng '{phan_mo_rong}' không được hỗ trợ. "
                       f"Các định dạng hỗ trợ: {dinh_dang_ho_tro}")
    
    # Kiểm tra kích thước file (không rỗng)
    if duong_dan.stat().st_size == 0:
        return False, "File ảnh rỗng (0 bytes)."
    
    return True, "File ảnh hợp lệ."


def kiem_tra_file_video(duong_dan):
    """Kiểm tra file video có hợp lệ hay không.
    
    Tham số:
        duong_dan: Đường dẫn đến file video.
        
    Trả về:
        (hop_le, thong_bao): hop_le=True nếu file hợp lệ,
        thong_bao chứa lý do nếu không hợp lệ.
    """
    if not duong_dan:
        return False, "Chưa chọn file."
    
    duong_dan = Path(duong_dan)
    
    if not duong_dan.exists():
        return False, f"File không tồn tại: {duong_dan.name}"
    
    if not duong_dan.is_file():
        return False, f"Đường dẫn không phải file: {duong_dan.name}"
    
    phan_mo_rong = duong_dan.suffix.lower()
    if phan_mo_rong not in DINH_DANG_VIDEO:
        dinh_dang_ho_tro = ', '.join(sorted(DINH_DANG_VIDEO))
        return False, (f"Định dạng '{phan_mo_rong}' không được hỗ trợ. "
                       f"Các định dạng hỗ trợ: {dinh_dang_ho_tro}")
    
    if duong_dan.stat().st_size == 0:
        return False, "File video rỗng (0 bytes)."
    
    return True, "File video hợp lệ."


def tao_ten_file_ket_qua(duong_dan_dau_vao, thu_muc_ket_qua, hau_to="_ketqua"):
    """Tạo tên file đầu ra, tránh trùng với file đã có.
    
    Ví dụ: road.jpg → road_ketqua.jpg
    Nếu đã tồn tại: road_ketqua_1.jpg, road_ketqua_2.jpg, ...
    
    Tham số:
        duong_dan_dau_vao: Đường dẫn file gốc.
        thu_muc_ket_qua: Thư mục lưu kết quả.
        hau_to: Hậu tố thêm vào tên file.
        
    Trả về:
        Đường dẫn đầy đủ của file kết quả (pathlib.Path).
    """
    duong_dan_dau_vao = Path(duong_dan_dau_vao)
    thu_muc_ket_qua = Path(thu_muc_ket_qua)
    
    # Đảm bảo thư mục kết quả tồn tại
    dam_bao_thu_muc(thu_muc_ket_qua)
    
    ten_goc = duong_dan_dau_vao.stem  # Tên file không có phần mở rộng
    phan_mo_rong = duong_dan_dau_vao.suffix  # Phần mở rộng (.jpg, .mp4, ...)
    
    # Thử tên cơ bản trước
    ten_ket_qua = thu_muc_ket_qua / f"{ten_goc}{hau_to}{phan_mo_rong}"
    
    # Nếu đã tồn tại, thêm số thứ tự
    dem = 1
    while ten_ket_qua.exists():
        ten_ket_qua = thu_muc_ket_qua / f"{ten_goc}{hau_to}_{dem}{phan_mo_rong}"
        dem += 1
    
    return ten_ket_qua


def dam_bao_thu_muc(duong_dan_thu_muc):
    """Tạo thư mục nếu chưa tồn tại (bao gồm cả thư mục cha).
    
    Tham số:
        duong_dan_thu_muc: Đường dẫn thư mục cần tạo.
    """
    Path(duong_dan_thu_muc).mkdir(parents=True, exist_ok=True)


def lay_kich_thuoc_file(duong_dan):
    """Lấy kích thước file và trả về chuỗi dễ đọc.
    
    Ví dụ: "2.5 MB", "340 KB"
    
    Tham số:
        duong_dan: Đường dẫn file.
        
    Trả về:
        Chuỗi kích thước dễ đọc, hoặc "—" nếu file không tồn tại.
    """
    try:
        kich_thuoc = Path(duong_dan).stat().st_size
        if kich_thuoc < 1024:
            return f"{kich_thuoc} B"
        elif kich_thuoc < 1024 * 1024:
            return f"{kich_thuoc / 1024:.1f} KB"
        else:
            return f"{kich_thuoc / (1024 * 1024):.1f} MB"
    except (OSError, FileNotFoundError):
        return "—"

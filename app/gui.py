"""
Giao diện ứng dụng — Hệ Thống Phát Hiện Làn Đường & Phương Tiện Giao Thông.

Công nghệ:
- Mô hình Học sâu Đa nhiệm AI YOLOPv2 (Phát hiện vạch làn, vùng mặt đường, phương tiện giao thông)

Đặc điểm giao diện hiện đại:
- Bố cục 50:50 đối xứng hoàn hảo giữa Khung Hình Gốc và Kết Quả Nhận Diện.
- Màn hình hiển thị nền tối phong cách hệ thống xe tự hành (Tesla HUD / Autonomous Driving Monitor).
- Thanh tiêu đề thương hiệu chuyên nghiệp kèm chỉ báo trạng thái thời gian thực.
- Bảng điều khiển dạng thẻ (Cards) với hệ màu chuẩn và phân cấp thao tác rõ ràng.
- Tự động co giãn ảnh thông minh (Responsive resizing) khi phóng to / thu nhỏ cửa sổ.
- Terminal nhật ký hoạt động có màu sắc trực quan (Console styling).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import queue
import time
from pathlib import Path

from lane_detection.video_processor import XuLyVideo
from utils.file_utils import (
    kiem_tra_file_anh, kiem_tra_file_video,
    tao_ten_file_ket_qua, dam_bao_thu_muc, lay_kich_thuoc_file,
    DINH_DANG_ANH, DINH_DANG_VIDEO
)


class UngDungPhatHienLan:
    """Giao diện chính của ứng dụng phát hiện làn đường."""

    def __init__(self, root):
        """Khởi tạo giao diện.

        Tham số:
            root: Cửa sổ Tk() chính.
        """
        self.root = root
        self.root.title("Hệ Thống Phát Hiện Làn Đường")
        self.root.geometry("1360x860")
        self.root.minsize(1100, 720)

        # Biến trạng thái
        self.che_do = tk.StringVar(value="anh")                    # "anh" hoặc "video"
        self.phuong_phap_chon = tk.StringVar(value="yolopv2")      # Cố định AI YOLOPv2
        self.trong_so_chon = tk.StringVar(value="yolopv2_multidomain.pt")

        # Tùy chọn hiển thị
        self.hien_xe = tk.BooleanVar(value=False)
        self.hien_lan = tk.BooleanVar(value=True)
        self.hien_duong = tk.BooleanVar(value=True)
        self.khoet_xe = tk.BooleanVar(value=True)

        self.duong_dan_file = None                 # Đường dẫn file hiện tại
        self.anh_goc = None                        # Ảnh gốc (numpy BGR)
        self.anh_ket_qua = None                    # Ảnh kết quả (numpy BGR)
        self.dang_xu_ly = False                    # Đang xử lý?

        # Bộ xử lý AI tích hợp
        self.xu_ly_video = XuLyVideo()

        # Thread và queue cho video
        self.thread_xu_ly = None
        self.su_kien_dung = threading.Event()
        self.hang_doi = queue.Queue()

        # Giữ tham chiếu ảnh để Tkinter không xóa bộ nhớ
        self._anh_goc_tk = None
        self._anh_ket_qua_tk = None
        self._resize_timer = None

        # Cài đặt giao diện & style
        self._thiet_lap_style()
        self._tao_giao_dien()

        # Ghi nhật ký khởi động
        self._ghi_nhat_ky("Hệ thống đã khởi động thành công.", "success")
        self._ghi_nhat_ky("Mô hình AI: YOLOPv2 sẵn sàng.", "info")

        # Xử lý đóng cửa sổ & vòng lặp kiểm tra queue
        self.root.protocol("WM_DELETE_WINDOW", self._dong_ung_dung)
        self._kiem_tra_hang_doi()

    # ==================== THIẾT LẬP STYLE GIAO DIỆN ====================

    def _thiet_lap_style(self):
        """Thiết lập theme hiện đại, màu sắc và typography đồng bộ."""
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except Exception:
            pass

        # Bảng màu chuẩn Modern Slate & Emerald
        self.root.configure(bg="#0f172a")

        # Cấu hình chung cho ttk
        self.style.configure(".", font=("Segoe UI", 9), background="#f8fafc", foreground="#1e293b")

        # Thẻ Card (LabelFrame)
        self.style.configure(
            "Card.TLabelframe",
            background="#ffffff",
            relief="solid",
            borderwidth=1,
            bordercolor="#cbd5e1"
        )
        self.style.configure(
            "Card.TLabelframe.Label",
            font=("Segoe UI", 9, "bold"),
            foreground="#0f172a",
            background="#ffffff"
        )

        # Monitor Card (Dành cho 2 khung ảnh)
        self.style.configure(
            "Monitor.TLabelframe",
            background="#ffffff",
            relief="solid",
            borderwidth=1,
            bordercolor="#cbd5e1"
        )
        self.style.configure(
            "Monitor.TLabelframe.Label",
            font=("Segoe UI", 9, "bold"),
            foreground="#0f172a",
            background="#ffffff"
        )

        # Style Nút Bấm Thao Tác (Color-coded action buttons)
        self.style.configure(
            "Primary.TButton",
            font=("Segoe UI", 9, "bold"),
            background="#2563eb",
            foreground="#ffffff",
            padding=(10, 5),
            borderwidth=0
        )
        self.style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")])

        self.style.configure(
            "Success.TButton",
            font=("Segoe UI", 10, "bold"),
            background="#059669",
            foreground="#ffffff",
            padding=(10, 6),
            borderwidth=0
        )
        self.style.map("Success.TButton", background=[("active", "#047857"), ("disabled", "#94a3b8")])

        self.style.configure(
            "Danger.TButton",
            font=("Segoe UI", 9, "bold"),
            background="#dc2626",
            foreground="#ffffff",
            padding=(10, 5),
            borderwidth=0
        )
        self.style.map("Danger.TButton", background=[("active", "#b91c1c"), ("disabled", "#94a3b8")])

        self.style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 9),
            background="#475569",
            foreground="#ffffff",
            padding=(10, 5),
            borderwidth=0
        )
        self.style.map("Secondary.TButton", background=[("active", "#334155"), ("disabled", "#94a3b8")])

        # Style Checkbox & Radiobutton
        self.style.configure("TCheckbutton", background="#ffffff", font=("Segoe UI", 9))
        self.style.configure("TRadiobutton", background="#ffffff", font=("Segoe UI", 9))

        # Thanh tiến trình (Emerald progressbar)
        self.style.configure(
            "Emerald.Horizontal.TProgressbar",
            troughcolor="#e2e8f0",
            background="#10b981",
            thickness=10,
            borderwidth=0
        )

    # ==================== TẠO BỐ CỤC TỔNG THỂ ====================

    def _tao_giao_dien(self):
        """Tạo cấu trúc layout phân tầng hiện đại."""
        # 1. Thanh tiêu đề Header thương hiệu
        self._tao_thanh_tieu_de()

        # 2. Vùng làm việc chính (Body)
        vung_lam_viec = tk.Frame(self.root, bg="#f1f5f9", padx=10, pady=8)
        vung_lam_viec.pack(fill=tk.BOTH, expand=True)

        # Cột trái: Bảng điều khiển (Sidebar rộng 330px)
        khung_trai = tk.Frame(vung_lam_viec, bg="#f1f5f9", width=330)
        khung_trai.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        khung_trai.pack_propagate(False)

        self._tao_bang_dieu_khien(khung_trai)

        # Cột phải: Khu vực 2 Màn hình đối xứng + Terminal Nhật ký
        khung_phai = tk.Frame(vung_lam_viec, bg="#f1f5f9")
        khung_phai.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._tao_khu_vuc_hien_thi(khung_phai)
        self._tao_khu_vuc_nhat_ky(khung_phai)

    # ==================== HEADER BAR ====================

    def _tao_thanh_tieu_de(self):
        """Tạo thanh tiêu đề tối sang trọng trên cùng."""
        header = tk.Frame(self.root, bg="#0f172a", height=58, padx=16, pady=8)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        # Trái: Tên hệ thống & thông tin đề tài
        tieu_de_frame = tk.Frame(header, bg="#0f172a")
        tieu_de_frame.pack(side=tk.LEFT, fill=tk.Y)

        lbl_title = tk.Label(
            tieu_de_frame,
            text="🚗 HỆ THỐNG PHÁT HIỆN LÀN ĐƯỜNG",
            font=("Segoe UI", 12, "bold"),
            fg="#f8fafc",
            bg="#0f172a"
        )
        lbl_title.pack(anchor=tk.W)

        # Phải: Badge trạng thái thời gian thực
        self.khung_badge = tk.Frame(header, bg="#1e293b", padx=12, pady=4, relief="ridge", bd=1)
        self.khung_badge.pack(side=tk.RIGHT, pady=4)

        self.lbl_badge_status = tk.Label(
            self.khung_badge,
            text="● SẴN SÀNG",
            font=("Segoe UI", 9, "bold"),
            fg="#34d399",
            bg="#1e293b"
        )
        self.lbl_badge_status.pack()

    def _cap_nhat_badge(self, trang_thai):
        """Cập nhật pill badge trạng thái ở header."""
        if trang_thai == "san_sang":
            self.lbl_badge_status.config(text="● SẴN SÀNG", fg="#34d399")
        elif trang_thai == "dang_xu_ly":
            self.lbl_badge_status.config(text="● ĐANG XỬ LÝ...", fg="#38bdf8")
        elif trang_thai == "hoan_thanh":
            self.lbl_badge_status.config(text="● HOÀN THÀNH", fg="#10b981")
        elif trang_thai == "loi":
            self.lbl_badge_status.config(text="● LỖI", fg="#f87171")

    # ==================== BẢNG ĐIỀU KHIỂN BÊN TRÁI ====================

    def _tao_bang_dieu_khien(self, cha):
        """Tạo các Card chức năng gọn gàng, khoa học ở cột trái."""
        # Card 1: Nguồn Dữ Liệu & Mô Hình
        card_nguon = ttk.LabelFrame(cha, text=" 1. Nguồn Dữ Liệu & Trọng Số ", style="Card.TLabelframe", padding=8)
        card_nguon.pack(fill=tk.X, pady=(0, 6))

        khung_radio = tk.Frame(card_nguon, bg="#ffffff")
        khung_radio.pack(fill=tk.X, pady=(0, 6))

        ttk.Radiobutton(
            khung_radio, text="📷 Hình ảnh", variable=self.che_do, value="anh",
            command=self._chuyen_che_do
        ).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Radiobutton(
            khung_radio, text="🎥 Video", variable=self.che_do, value="video",
            command=self._chuyen_che_do
        ).pack(side=tk.LEFT)

        # Thông tin mô hình AI
        badge_ai = tk.Frame(card_nguon, bg="#f1f5f9", padx=8, pady=4, relief="groove", bd=1)
        badge_ai.pack(fill=tk.X, pady=(2, 6))
        tk.Label(
            badge_ai, text="Mô hình AI: YOLOPv2 (Đa nhiệm)",
            font=("Segoe UI", 8, "bold"), fg="#1e293b", bg="#f1f5f9"
        ).pack(anchor=tk.W)

        # File Checkpoint
        tk.Label(card_nguon, text="File Checkpoint:", font=("Segoe UI", 8), bg="#ffffff", fg="#475569").pack(anchor=tk.W)

        thu_muc_weights = Path(__file__).resolve().parent.parent / "weights"
        danh_sach_weights = ["yolopv2_multidomain.pt", "yolopv2.pt"]
        if thu_muc_weights.exists():
            ds_tim_thay = [f.name for f in thu_muc_weights.glob("*.pt")]
            if ds_tim_thay:
                danh_sach_weights = sorted(ds_tim_thay)

        self.combo_weights = ttk.Combobox(
            card_nguon, values=danh_sach_weights,
            textvariable=self.trong_so_chon, state="readonly"
        )
        self.combo_weights.pack(fill=tk.X, pady=(1, 2))
        if "yolopv2_multidomain.pt" in danh_sach_weights:
            self.combo_weights.set("yolopv2_multidomain.pt")
        elif "yolopv2.pt" in danh_sach_weights:
            self.combo_weights.set("yolopv2.pt")
        elif danh_sach_weights:
            self.combo_weights.set(danh_sach_weights[0])

        # Card 2: Tùy Chọn Hiển Thị (Làn đường & Xe)
        card_tuy_chon = ttk.LabelFrame(cha, text=" 2. Tùy Chọn Hiển Thị ", style="Card.TLabelframe", padding=8)
        card_tuy_chon.pack(fill=tk.X, pady=(0, 6))

        self.cb_lan = ttk.Checkbutton(card_tuy_chon, text="🔴 Vạch đường (Lane Line)", variable=self.hien_lan)
        self.cb_lan.pack(anchor=tk.W, pady=2)

        self.cb_duong = ttk.Checkbutton(card_tuy_chon, text="🔵 Làn xe chạy (Ego-Lane)", variable=self.hien_duong)
        self.cb_duong.pack(anchor=tk.W, pady=2)

        self.cb_khoet_xe = ttk.Checkbutton(card_tuy_chon, text="🛡️ Không tô đè lên xe & người", variable=self.khoet_xe)
        self.cb_khoet_xe.pack(anchor=tk.W, pady=2)

        self.cb_xe = ttk.Checkbutton(card_tuy_chon, text="🚗 Khung viền xe (Tùy chọn)", variable=self.hien_xe)
        self.cb_xe.pack(anchor=tk.W, pady=2)

        # Card 3: Thao Tác (Action Buttons)
        card_thao_tac = ttk.LabelFrame(cha, text=" 3. Thao Tác ", style="Card.TLabelframe", padding=8)
        card_thao_tac.pack(fill=tk.X, pady=(0, 6))

        self.nut_mo_tep = ttk.Button(card_thao_tac, text="📂 Mở tệp...", command=self._mo_tep, style="Primary.TButton")
        self.nut_mo_tep.pack(fill=tk.X, pady=2)

        self.nut_xu_ly = ttk.Button(card_thao_tac, text="▶ BẮT ĐẦU XỬ LÝ", command=self._bat_dau_xu_ly, style="Success.TButton", state=tk.DISABLED)
        self.nut_xu_ly.pack(fill=tk.X, pady=4)

        khung_phu = tk.Frame(card_thao_tac, bg="#ffffff")
        khung_phu.pack(fill=tk.X, pady=1)

        self.nut_dung = ttk.Button(khung_phu, text="⏹ Dừng", command=self._dung_xu_ly, style="Danger.TButton", state=tk.DISABLED)
        self.nut_dung.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.nut_luu = ttk.Button(khung_phu, text="💾 Lưu kết quả", command=self._luu_ket_qua, style="Secondary.TButton", state=tk.DISABLED)
        self.nut_luu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # Card 4: Thông Số & Trạng Thái
        card_thong_so = ttk.LabelFrame(cha, text=" 4. Thông Số & Trạng Thái ", style="Card.TLabelframe", padding=8)
        card_thong_so.pack(fill=tk.BOTH, expand=True, pady=(0, 2))

        thong_tin_items = [
            ("Trạng thái:", "trang_thai", "Sẵn sàng"),
            ("Phương pháp:", "phuong_phap", "AI YOLOPv2"),
            ("Tệp:", "tep", "—"),
            ("Kích thước:", "kich_thuoc", "—"),
            ("Phương tiện:", "so_xe", "—"),
            ("Tốc độ / FPS:", "fps", "—"),
        ]

        self.nhan_thong_tin = {}

        bang_tt = tk.Frame(card_thong_so, bg="#ffffff")
        bang_tt.pack(fill=tk.X, pady=1)

        for i, (nhan, key, mac_dinh) in enumerate(thong_tin_items):
            tk.Label(
                bang_tt, text=nhan, font=("Segoe UI", 8, "bold"),
                fg="#475569", bg="#ffffff", anchor=tk.W
            ).grid(row=i, column=0, sticky="w", pady=2)

            lbl = tk.Label(
                bang_tt, text=mac_dinh, font=("Segoe UI", 8),
                fg="#0f172a", bg="#ffffff", anchor=tk.W, wraplength=190, justify=tk.LEFT
            )
            lbl.grid(row=i, column=1, sticky="w", padx=(6, 0), pady=2)
            self.nhan_thong_tin[key] = lbl

        # Thanh tiến trình mượt mà (chỉ hiện khi xử lý video)
        self.khung_tien_trinh = tk.Frame(card_thong_so, bg="#ffffff")
        self.khung_tien_trinh.pack(fill=tk.X, pady=(8, 0))

        tk.Label(self.khung_tien_trinh, text="Tiến trình:", font=("Segoe UI", 8, "bold"), fg="#475569", bg="#ffffff").pack(anchor=tk.W)
        khung_bar = tk.Frame(self.khung_tien_trinh, bg="#ffffff")
        khung_bar.pack(fill=tk.X, pady=(2, 0))

        self.thanh_tien_trinh = ttk.Progressbar(
            khung_bar, mode='determinate', style="Emerald.Horizontal.TProgressbar"
        )
        self.thanh_tien_trinh.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.nhan_phan_tram = tk.Label(khung_bar, text="0%", font=("Segoe UI", 8, "bold"), fg="#059669", bg="#ffffff", width=4)
        self.nhan_phan_tram.pack(side=tk.LEFT)

        self.khung_tien_trinh.pack_forget()

    # ==================== KHU VỰC HIỂN THỊ ĐỐI XỨNG 50:50 ====================

    def _tao_khu_vuc_hien_thi(self, cha):
        """Tạo khu vực hiển thị chia đôi 50:50 tuyệt đối bằng grid uniform."""
        self.khung_anh = ttk.Frame(cha)
        self.khung_anh.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        # KHÓA TỶ LỆ 50% - 50% TUYỆT ĐỐI CHO 2 CỘT HIỂN THỊ
        self.khung_anh.grid_rowconfigure(0, weight=1)
        self.khung_anh.grid_columnconfigure(0, weight=1, uniform="display_monitor")
        self.khung_anh.grid_columnconfigure(1, weight=1, uniform="display_monitor")

        # Cột 1: Khung hình gốc
        khung_goc = ttk.LabelFrame(
            self.khung_anh, text=" 📷 Khung Hình Gốc (Original Input) ",
            style="Monitor.TLabelframe", padding=4
        )
        khung_goc.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
        khung_goc.grid_rowconfigure(0, weight=1)
        khung_goc.grid_columnconfigure(0, weight=1)

        self.nhan_anh_goc = tk.Label(
            khung_goc,
            text="CHƯA CÓ DỮ LIỆU ĐẦU VÀO\n\nNhấn '📂 Mở tệp' để tải ảnh hoặc video",
            font=("Segoe UI", 10, "bold"),
            fg="#64748b",
            bg="#0b0f19",
            anchor=tk.CENTER,
            justify=tk.CENTER
        )
        self.nhan_anh_goc.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # Cột 2: Khung hình kết quả
        khung_ket_qua = ttk.LabelFrame(
            self.khung_anh, text=" 🎯 Kết Quả Nhận Diện (AI Perception Overlay) ",
            style="Monitor.TLabelframe", padding=4
        )
        khung_ket_qua.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=0)
        khung_ket_qua.grid_rowconfigure(0, weight=1)
        khung_ket_qua.grid_columnconfigure(0, weight=1)

        self.nhan_anh_ket_qua = tk.Label(
            khung_ket_qua,
            text="CHƯA XỬ LÝ\n\nNhấn '▶ BẮT ĐẦU XỬ LÝ' để nhận diện làn đường",
            font=("Segoe UI", 10, "bold"),
            fg="#64748b",
            bg="#0b0f19",
            anchor=tk.CENTER,
            justify=tk.CENTER
        )
        self.nhan_anh_ket_qua.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # Lắng nghe sự kiện co giãn cửa sổ để tự động resize ảnh mượt mà
        self.khung_anh.bind("<Configure>", self._on_resize_display)

    def _on_resize_display(self, event=None):
        """Debounce tự động tính toán lại kích thước ảnh khi người dùng kéo giãn cửa sổ."""
        if self._resize_timer is not None:
            self.root.after_cancel(self._resize_timer)
        self._resize_timer = self.root.after(120, self._ve_lai_cac_anh)

    def _ve_lai_cac_anh(self):
        """Vẽ lại cả hai ảnh để luôn giữ kích thước 50-50 cân đối hoàn hảo."""
        if self.anh_goc is not None:
            self._hien_thi_anh(self.anh_goc, self.nhan_anh_goc, "goc")
        if self.anh_ket_qua is not None:
            self._hien_thi_anh(self.anh_ket_qua, self.nhan_anh_ket_qua, "ket_qua")

    # ==================== TERMINAL NHẬT KÝ (LOGS) ====================

    def _tao_khu_vuc_nhat_ky(self, cha):
        """Tạo terminal nhật ký hoạt động có màu sắc chuyên nghiệp."""
        khung_log = ttk.LabelFrame(cha, text=" 📋 Nhật Ký Hoạt Động (System Logs) ", style="Card.TLabelframe", padding=4)
        khung_log.pack(fill=tk.X, pady=(0, 0))

        # Thanh công cụ nhỏ của log
        toolbar_log = tk.Frame(khung_log, bg="#ffffff")
        toolbar_log.pack(fill=tk.X, pady=(0, 2))

        lbl_log_hint = tk.Label(
            toolbar_log, text="Ghi nhận thông tin phân tích mô hình thời gian thực",
            font=("Segoe UI", 8), fg="#64748b", bg="#ffffff"
        )
        lbl_log_hint.pack(side=tk.LEFT)

        btn_clear = tk.Button(
            toolbar_log, text="🧹 Xóa log", font=("Segoe UI", 8),
            bg="#f1f5f9", fg="#475569", relief="flat", bd=0, padx=6, pady=1,
            command=self._xoa_nhat_ky
        )
        btn_clear.pack(side=tk.RIGHT)

        # Khung Text terminal
        khung_text = tk.Frame(khung_log, bg="#0b1120")
        khung_text.pack(fill=tk.X)

        self.text_nhat_ky = tk.Text(
            khung_text, height=5, wrap=tk.WORD, state=tk.DISABLED,
            font=("Consolas", 9), bg="#0b1120", fg="#e2e8f0",
            insertbackground="#38bdf8", selectbackground="#1e293b",
            relief="flat", padx=6, pady=4
        )
        self.text_nhat_ky.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(
            khung_text, orient=tk.VERTICAL, command=self.text_nhat_ky.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_nhat_ky.config(yscrollcommand=scrollbar.set)

        # Định nghĩa các thẻ màu hiển thị log
        self.text_nhat_ky.tag_config("time", foreground="#64748b")
        self.text_nhat_ky.tag_config("success", foreground="#34d399")
        self.text_nhat_ky.tag_config("info", foreground="#38bdf8")
        self.text_nhat_ky.tag_config("warning", foreground="#fbbf24")
        self.text_nhat_ky.tag_config("error", foreground="#f87171")
        self.text_nhat_ky.tag_config("normal", foreground="#e2e8f0")

    def _xoa_nhat_ky(self):
        """Xóa trắng nội dung log."""
        self.text_nhat_ky.config(state=tk.NORMAL)
        self.text_nhat_ky.delete("1.0", tk.END)
        self.text_nhat_ky.config(state=tk.DISABLED)

    # ==================== CÁC HÀM XỬ LÝ SỰ KIỆN ====================

    def _chuyen_che_do(self):
        """Xử lý khi chuyển đổi giữa Ảnh và Video."""
        cd = self.che_do.get()
        self._ghi_nhat_ky(f"Chuyển nguồn dữ liệu sang: {'Video' if cd == 'video' else 'Hình ảnh'}", "info")
        self.duong_dan_file = None
        self.anh_goc = None
        self.anh_ket_qua = None

        self.nhan_anh_goc.config(image='', text="CHƯA CÓ DỮ LIỆU ĐẦU VÀO\n\nNhấn '📂 Mở tệp' để tải ảnh hoặc video")
        self.nhan_anh_ket_qua.config(image='', text="CHƯA XỬ LÝ\n\nNhấn '▶ BẮT ĐẦU XỬ LÝ' để nhận diện làn đường")

        self.nut_xu_ly.config(state=tk.DISABLED)
        self.nut_luu.config(state=tk.DISABLED)
        self.nhan_thong_tin['tep'].config(text="—")
        self.nhan_thong_tin['kich_thuoc'].config(text="—")
        self.nhan_thong_tin['so_xe'].config(text="—")
        self.nhan_thong_tin['fps'].config(text="—")
        self.nhan_thong_tin['trang_thai'].config(text="Sẵn sàng")
        self._cap_nhat_badge("san_sang")

    def _mo_tep(self):
        """Mở hộp thoại chọn file ảnh hoặc video."""
        thu_muc_project = Path(__file__).resolve().parent.parent

        if self.che_do.get() == "anh":
            thu_muc_mac_dinh = thu_muc_project / "data" / "images"
            dinh_dang = [
                ("Ảnh", " ".join(f"*{ext}" for ext in DINH_DANG_ANH)),
                ("Tất cả", "*.*")
            ]
        else:
            thu_muc_mac_dinh = thu_muc_project / "data" / "videos"
            dinh_dang = [
                ("Video", " ".join(f"*{ext}" for ext in DINH_DANG_VIDEO)),
                ("Tất cả", "*.*")
            ]

        if not thu_muc_mac_dinh.exists():
            thu_muc_mac_dinh = thu_muc_project

        duong_dan = filedialog.askopenfilename(
            title="Chọn tệp đầu vào",
            initialdir=str(thu_muc_mac_dinh),
            filetypes=dinh_dang
        )

        if not duong_dan:
            return

        if self.che_do.get() == "anh":
            hop_le, thong_bao = kiem_tra_file_anh(duong_dan)
        else:
            hop_le, thong_bao = kiem_tra_file_video(duong_dan)

        if not hop_le:
            messagebox.showerror("Tệp không hợp lệ", thong_bao)
            self._ghi_nhat_ky(f"Lỗi tệp: {thong_bao}", "error")
            return

        self.duong_dan_file = Path(duong_dan)
        self.nhan_thong_tin['tep'].config(text=self.duong_dan_file.name)
        self._ghi_nhat_ky(f"Đã mở tệp: {self.duong_dan_file.name}", "info")

        if self.che_do.get() == "anh":
            self._hien_thi_anh_dau_vao(self.duong_dan_file)
        else:
            self._hien_thi_video_dau_vao(self.duong_dan_file)

        self.nut_xu_ly.config(state=tk.NORMAL)
        self.nut_luu.config(state=tk.DISABLED)

        self.anh_ket_qua = None
        self.nhan_anh_ket_qua.config(image='', text="CHƯA XỬ LÝ\n\nNhấn '▶ BẮT ĐẦU XỬ LÝ' để nhận diện làn đường")
        self._anh_ket_qua_tk = None

    def _hien_thi_anh_dau_vao(self, duong_dan):
        """Đọc và hiển thị ảnh đầu vào."""
        self.anh_goc = cv2.imread(str(duong_dan))

        if self.anh_goc is None:
            messagebox.showerror("Lỗi", "Không thể nạp file ảnh này.")
            self._ghi_nhat_ky("Lỗi: Không đọc được ảnh.", "error")
            return

        chieu_cao, chieu_rong = self.anh_goc.shape[:2]
        self.nhan_thong_tin['kich_thuoc'].config(text=f"{chieu_rong} × {chieu_cao} px")
        self.nhan_thong_tin['trang_thai'].config(text="Đã tải ảnh")
        self._cap_nhat_badge("san_sang")

        self._hien_thi_anh(self.anh_goc, self.nhan_anh_goc, "goc")

    def _hien_thi_video_dau_vao(self, duong_dan):
        """Đọc frame đầu tiên của video và hiển thị."""
        cap = cv2.VideoCapture(str(duong_dan))

        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể nạp file video này.")
            self._ghi_nhat_ky("Lỗi: Không mở được video.", "error")
            return

        ret, frame = cap.read()
        tong_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        chieu_rong = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        chieu_cao = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if not ret:
            messagebox.showerror("Lỗi", "Không đọc được khung hình đầu tiên của video.")
            return

        self.anh_goc = frame
        self.nhan_thong_tin['kich_thuoc'].config(
            text=f"{chieu_rong} × {chieu_cao} ({tong_frame} frames)"
        )
        self.nhan_thong_tin['fps'].config(text=f"{fps:.1f} FPS gốc")
        self.nhan_thong_tin['trang_thai'].config(text="Đã tải video")
        self._cap_nhat_badge("san_sang")

        self._hien_thi_anh(frame, self.nhan_anh_goc, "goc")

    def _bat_dau_xu_ly(self):
        """Bắt đầu xử lý ảnh hoặc video."""
        if self.duong_dan_file is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở tệp trước khi xử lý.")
            return

        if self.che_do.get() == "anh":
            self._xu_ly_anh()
        else:
            self._xu_ly_video_bat_dau()

    def _xu_ly_anh(self):
        """Xử lý ảnh đơn trên thread riêng biệt (Giao diện hoàn toàn mượt mà)."""
        if self.anh_goc is None or self.dang_xu_ly:
            return

        ve_xe = self.hien_xe.get()
        ve_lan = self.hien_lan.get()
        ve_vung_lai = self.hien_duong.get()

        ten_weights = self.trong_so_chon.get()
        duong_dan_weights = Path(__file__).resolve().parent.parent / "weights" / ten_weights

        self.dang_xu_ly = True
        self.nut_xu_ly.config(state=tk.DISABLED)
        self.nut_mo_tep.config(state=tk.DISABLED)
        self.nhan_thong_tin['trang_thai'].config(text="Đang phân tích...")
        self._cap_nhat_badge("dang_xu_ly")
        self._ghi_nhat_ky(f"Đang phân tích với trọng số: {ten_weights}...", "info")

        def worker():
            thoi_gian_bat_dau = time.time()
            try:
                anh_ket_qua, thong_tin = self.xu_ly_video.xu_ly_mot_anh(
                    self.anh_goc,
                    ve_xe=ve_xe,
                    ve_lan=ve_lan,
                    ve_vung_lai=ve_vung_lai,
                    duong_dan_weights=duong_dan_weights,
                    khoet_phuong_tien=self.khoet_xe.get()
                )
                thoi_gian = time.time() - thoi_gian_bat_dau

                def cap_nhat_gui():
                    self.anh_ket_qua = anh_ket_qua
                    self._hien_thi_anh(anh_ket_qua, self.nhan_anh_ket_qua, "ket_qua")

                    self.nhan_thong_tin['phuong_phap'].config(text="AI YOLOPv2")
                    self.nhan_thong_tin['trang_thai'].config(text="Hoàn thành")
                    self.nhan_thong_tin['fps'].config(text=f"{1.0/max(thoi_gian, 0.001):.1f} FPS ({thoi_gian*1000:.0f} ms)")

                    so_xe = thong_tin.get('so_luong_xe', 0)
                    self.nhan_thong_tin['so_xe'].config(text=f"{so_xe} xe" if ve_xe else "Đã khoét thân xe")

                    self._ghi_nhat_ky(f"✓ Phát hiện thành công ({thoi_gian*1000:.0f}ms): Vạch đỏ + Làn xanh nước biển", "success")
                    self._cap_nhat_badge("hoan_thanh")

                    self.nut_luu.config(state=tk.NORMAL)
                    self.nut_xu_ly.config(state=tk.NORMAL)
                    self.nut_mo_tep.config(state=tk.NORMAL)
                    self.dang_xu_ly = False

                self.root.after(0, cap_nhat_gui)

            except Exception as e:
                def bao_loi():
                    self.nhan_thong_tin['trang_thai'].config(text="Lỗi xử lý")
                    self._cap_nhat_badge("loi")
                    self._ghi_nhat_ky(f"Lỗi khi xử lý: {str(e)}", "error")
                    messagebox.showerror("Lỗi", f"Xảy ra lỗi trong quá trình xử lý:\n{str(e)}")
                    self.nut_xu_ly.config(state=tk.NORMAL)
                    self.nut_mo_tep.config(state=tk.NORMAL)
                    self.dang_xu_ly = False

                self.root.after(0, bao_loi)

        threading.Thread(target=worker, daemon=True).start()

    def _xu_ly_video_bat_dau(self):
        """Bắt đầu xử lý video trên thread riêng biệt."""
        if self.dang_xu_ly:
            return

        thu_muc_project = Path(__file__).resolve().parent.parent
        thu_muc_ket_qua = thu_muc_project / "results" / "videos"
        dam_bao_thu_muc(thu_muc_ket_qua)

        self.duong_dan_ket_qua_video = tao_ten_file_ket_qua(
            self.duong_dan_file, thu_muc_ket_qua, "_ketqua"
        ).with_suffix('.mp4')

        ve_xe = self.hien_xe.get()
        ve_lan = self.hien_lan.get()
        ve_vung_lai = self.hien_duong.get()

        ten_weights = self.trong_so_chon.get()
        duong_dan_weights = thu_muc_project / "weights" / ten_weights

        self._ghi_nhat_ky("Bắt đầu xử lý video bằng AI YOLOPv2...", "info")
        self._ghi_nhat_ky(f"Tệp kết quả: {self.duong_dan_ket_qua_video.name}", "normal")

        self.dang_xu_ly = True
        self.su_kien_dung.clear()

        # Cập nhật trạng thái nút
        self.nut_xu_ly.config(state=tk.DISABLED)
        self.nut_dung.config(state=tk.NORMAL)
        self.nut_mo_tep.config(state=tk.DISABLED)
        self.nut_luu.config(state=tk.DISABLED)

        # Hiện thanh tiến trình
        self.khung_tien_trinh.pack(fill=tk.X, pady=(8, 0))
        self.thanh_tien_trinh['value'] = 0
        self.nhan_phan_tram.config(text="0%")
        self.nhan_thong_tin['trang_thai'].config(text="Đang xử lý video...")
        self._cap_nhat_badge("dang_xu_ly")

        khoet_xe = self.khoet_xe.get()

        self.thread_xu_ly = threading.Thread(
            target=self._thread_xu_ly_video,
            args=(ve_xe, ve_lan, ve_vung_lai, duong_dan_weights, khoet_xe),
            daemon=True
        )
        self.thread_xu_ly.start()

    def _thread_xu_ly_video(self, ve_xe, ve_lan, ve_vung_lai, duong_dan_weights, khoet_xe=True):
        """Hàm chạy trên thread riêng — xử lý video bằng YOLOPv2."""
        try:
            thanh_cong, thong_bao = self.xu_ly_video.xu_ly_video(
                self.duong_dan_file,
                self.duong_dan_ket_qua_video,
                ham_cap_nhat_tien_trinh=self._callback_tien_trinh,
                su_kien_dung=self.su_kien_dung,
                ve_xe=ve_xe,
                ve_lan=ve_lan,
                ve_vung_lai=ve_vung_lai,
                frame_skip=2,
                duong_dan_weights=duong_dan_weights,
                khoet_phuong_tien=khoet_xe
            )
            self.hang_doi.put(('hoan_thanh', thanh_cong, thong_bao))
        except Exception as e:
            self.hang_doi.put(('hoan_thanh', False, f"Lỗi: {str(e)}"))

    def _callback_tien_trinh(self, frame_hien_tai, tong_frame, fps, thong_tin, frame_ket_qua):
        """Callback từ worker thread gửi tiến trình về GUI qua queue."""
        frame_gui = frame_ket_qua if (frame_hien_tai % 2 == 0 or frame_hien_tai == tong_frame) else None
        self.hang_doi.put((
            'tien_trinh', frame_hien_tai, tong_frame, fps, thong_tin, frame_gui
        ))

    def _kiem_tra_hang_doi(self):
        """Kiểm tra queue định kỳ để cập nhật GUI mượt mà."""
        try:
            while True:
                msg = self.hang_doi.get_nowait()

                if msg[0] == 'tien_trinh':
                    _, frame_ht, tong, fps, thong_tin, frame_anh = msg
                    phan_tram = int(frame_ht / max(tong, 1) * 100)

                    self.thanh_tien_trinh['value'] = phan_tram
                    self.nhan_phan_tram.config(text=f"{phan_tram}%")
                    self.nhan_thong_tin['fps'].config(text=f"{fps:.1f} FPS")

                    so_xe = thong_tin.get('so_luong_xe', 0)
                    self.nhan_thong_tin['so_xe'].config(text=f"{so_xe} xe" if so_xe > 0 else "0")

                    if frame_anh is not None:
                        self.anh_ket_qua = frame_anh
                        self._hien_thi_anh(frame_anh, self.nhan_anh_ket_qua, "ket_qua")

                elif msg[0] == 'hoan_thanh':
                    _, thanh_cong, thong_bao = msg
                    self.dang_xu_ly = False

                    self.nut_xu_ly.config(state=tk.NORMAL)
                    self.nut_dung.config(state=tk.DISABLED)
                    self.nut_mo_tep.config(state=tk.NORMAL)

                    if thanh_cong:
                        self.nhan_thong_tin['trang_thai'].config(text="Hoàn thành")
                        self._cap_nhat_badge("hoan_thanh")
                        self._ghi_nhat_ky(f"✓ Video hoàn thành: {thong_bao}", "success")
                        self.nut_luu.config(state=tk.NORMAL)
                        messagebox.showinfo("Hoàn thành", f"Xử lý video hoàn tất!\n{thong_bao}")
                    else:
                        self.nhan_thong_tin['trang_thai'].config(text="Đã dừng/Lỗi")
                        self._cap_nhat_badge("warning")
                        self._ghi_nhat_ky(f"Xử lý video: {thong_bao}", "warning")

        except queue.Empty:
            pass

        self.root.after(50, self._kiem_tra_hang_doi)

    def _dung_xu_ly(self):
        """Dừng tiến trình xử lý video."""
        if self.dang_xu_ly:
            self.su_kien_dung.set()
            self._ghi_nhat_ky("Đang dừng tiến trình xử lý video...", "warning")
            self.nhan_thong_tin['trang_thai'].config(text="Đang dừng...")

    def _luu_ket_qua(self):
        """Lưu ảnh hoặc thông báo vị trí video kết quả."""
        if self.che_do.get() == "anh":
            self._luu_anh_ket_qua()
        else:
            self._luu_video_ket_qua()

    def _luu_anh_ket_qua(self):
        """Lưu ảnh kết quả vào file."""
        if self.anh_ket_qua is None:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để lưu.")
            return

        thu_muc_project = Path(__file__).resolve().parent.parent
        thu_muc_ket_qua = thu_muc_project / "results" / "images"
        dam_bao_thu_muc(thu_muc_ket_qua)

        ten_mac_dinh = tao_ten_file_ket_qua(self.duong_dan_file, thu_muc_ket_qua)

        duong_dan = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả",
            initialdir=str(thu_muc_ket_qua),
            initialfile=ten_mac_dinh.name,
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("BMP", "*.bmp")
            ]
        )

        if not duong_dan:
            return

        thanh_cong = cv2.imwrite(str(duong_dan), self.anh_ket_qua)
        if thanh_cong:
            self._ghi_nhat_ky(f"Đã lưu ảnh: {Path(duong_dan).name}", "success")
            messagebox.showinfo("Thành công", f"Đã lưu ảnh kết quả tại:\n{duong_dan}")
        else:
            messagebox.showerror("Lỗi", "Không thể ghi file ảnh.")

    def _luu_video_ket_qua(self):
        """Thông báo vị trí video kết quả đã lưu."""
        if hasattr(self, 'duong_dan_ket_qua_video') and self.duong_dan_ket_qua_video:
            if self.duong_dan_ket_qua_video.exists():
                kich_thuoc = lay_kich_thuoc_file(self.duong_dan_ket_qua_video)
                self._ghi_nhat_ky(f"Video kết quả: {self.duong_dan_ket_qua_video.name} ({kich_thuoc})", "success")
                messagebox.showinfo(
                    "Thành công",
                    f"Video kết quả đã được lưu an toàn tại:\n{self.duong_dan_ket_qua_video}"
                )
            else:
                messagebox.showwarning("Cảnh báo", "File video kết quả không tồn tại.")
        else:
            messagebox.showwarning("Cảnh báo", "Chưa có video kết quả.")

    # ==================== HIỂN THỊ ẢNH ĐỐI XỨNG CHUẨN KÍCH THƯỚC ====================

    def _hien_thi_anh(self, anh_bgr, nhan_label, loai):
        """Chuyển ảnh BGR sang PhotoImage và hiển thị trên Label căn giữa tuyệt đối.
        
        Đảm bảo cả hai khung hiển thị luôn luôn áp dụng cùng kích thước scale,
        tránh hiện tượng bên to bên nhỏ.
        """
        self.root.update_idletasks()

        # Lấy kích thước khả dụng của màn hình hiển thị
        w_goc = self.nhan_anh_goc.winfo_width()
        w_kq = self.nhan_anh_ket_qua.winfo_width()
        h_goc = self.nhan_anh_goc.winfo_height()
        h_kq = self.nhan_anh_ket_qua.winfo_height()

        # Dùng kích thước lớn nhất thực tế của cả 2 cột để 2 ảnh luôn có cùng tỷ lệ scale
        kha_dung_w = max(w_goc, w_kq)
        kha_dung_h = max(h_goc, h_kq)

        if kha_dung_w < 80:
            kha_dung_w = 480
        if kha_dung_h < 80:
            kha_dung_h = 360

        # Trừ nhẹ padding 8px để ảnh không chạm sát mép viền
        target_w = max(10, kha_dung_w - 8)
        target_h = max(10, kha_dung_h - 8)

        chieu_cao, chieu_rong = anh_bgr.shape[:2]
        ty_le = min(target_w / chieu_rong, target_h / chieu_cao)
        kich_thuoc_moi = (max(1, int(chieu_rong * ty_le)), max(1, int(chieu_cao * ty_le)))

        anh_resize = cv2.resize(anh_bgr, kich_thuoc_moi, interpolation=cv2.INTER_AREA)
        anh_rgb = cv2.cvtColor(anh_resize, cv2.COLOR_BGR2RGB)
        anh_pil = Image.fromarray(anh_rgb)
        anh_tk = ImageTk.PhotoImage(anh_pil)

        nhan_label.config(image=anh_tk, text="")

        if loai == "goc":
            self._anh_goc_tk = anh_tk
        else:
            self._anh_ket_qua_tk = anh_tk

    # ==================== NHẬT KÝ VỚI MÀU SẮC ====================

    def _ghi_nhat_ky(self, noi_dung, tag="normal"):
        """Ghi một dòng vào khung nhật ký với timestamp và highlight màu sắc."""
        thoi_gian = time.strftime("%H:%M:%S")
        dong_time = f"[{thoi_gian}] "
        dong_msg = f"{noi_dung}\n"

        self.text_nhat_ky.config(state=tk.NORMAL)
        self.text_nhat_ky.insert(tk.END, dong_time, "time")
        self.text_nhat_ky.insert(tk.END, dong_msg, tag)
        self.text_nhat_ky.see(tk.END)
        self.text_nhat_ky.config(state=tk.DISABLED)

    # ==================== ĐÓNG ỨNG DỤNG ====================

    def _dong_ung_dung(self):
        """Đóng ứng dụng an toàn — dừng thread và giải phóng tài nguyên."""
        if self.dang_xu_ly:
            self.su_kien_dung.set()
            if self.thread_xu_ly is not None:
                self.thread_xu_ly.join(timeout=1.5)

        self.xu_ly_video.giai_phong()
        self.root.destroy()

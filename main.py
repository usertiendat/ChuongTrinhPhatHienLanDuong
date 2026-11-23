"""
Hệ Thống Phát Hiện Làn Đường & Phương Tiện Giao Thông — Điểm khởi chạy.

Ứng dụng phát hiện làn đường và phân tích giao thông sử dụng mô hình học sâu YOLOPv2.
Hỗ trợ xử lý hình ảnh và video mượt mà trên nền tảng desktop.

Chạy: python main.py
"""

import tkinter as tk
from app.gui import UngDungPhatHienLan


def main():
    """Khởi chạy ứng dụng desktop."""
    root = tk.Tk()
    app = UngDungPhatHienLan(root)  # Giao diện tiếng Việt
    root.mainloop()


if __name__ == "__main__":
    main()

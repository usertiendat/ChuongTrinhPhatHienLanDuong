# 🚗 HỆ THỐNG PHÁT HIỆN LÀN ĐƯỜNG & PHƯƠNG TIỆN GIAO THÔNG
### (LANE & VEHICLE DETECTION SYSTEM — AI YOLOPv2)

Dự án phát hiện làn đường, phân vùng mặt đường xe chạy và nhận diện phương tiện giao thông thời gian thực sử dụng **Mô hình Học sâu Đa nhiệm YOLOPv2 (Panoptic Driving Perception)** kết hợp giao diện Desktop Tkinter tiếng Việt hiện đại.

> **Đề tài**: Bài tập lớn môn Xử lý ảnh và Thị giác máy tính  
> **Nhóm thực hiện**: Nhóm 1 — Lớp 24DHTT02 | HK7 (Năm học 2026 – 2027)

---

## 🌟 TÍNH NĂNG NỔI BẬT

- **Mô hình Học sâu Đa nhiệm YOLOPv2 (3 nhánh đầu ra đồng thời)**:
  - 🔴 **Vạch kẻ đường (Lane Line)**: Phân đoạn chính xác vạch kẻ nét đứt, nét liền, phân làn đường cong.
  - 🔵 **Làn xe đang chạy (Drivable Area / Ego-Lane)**: Xác định không gian mặt đường xe có thể di chuyển, phân biệt làn xe chạy với vỉa hè và dải phân cách.
  - 🛡️ **Khoét thân xe thông minh (Vehicle Carving)**: Tự động khoét vùng thân xe và người đi đường, tránh việc lớp màu mặt đường bị tô đè lên các phương tiện phía trước.
  - 🚗 **Nhận diện phương tiện giao thông (Vehicle Detection)**: Bounding box và confidence score (tùy chọn bật/tắt linh hoạt).
- **Giao diện Desktop hiện đại (Modern Tkinter GUI)**:
  - Bố cục đối xứng **50:50 hoàn hảo** giữa Khung hình gốc và Kết quả nhận diện.
  - Màn hình hiển thị nền tối chuyên nghiệp phong cách xe tự hành (Tesla HUD / Autonomous Driving Monitor).
  - Tự động co giãn ảnh thông minh (Responsive Auto-Resize) khi kéo giãn cửa sổ.
  - Console nhật ký hoạt động có màu sắc trực quan (System Logs).
- **Tối ưu hóa suy luận (CPU Optimization)**:
  - Cơ chế **Frame-Skipping** thông minh giúp xử lý video mượt mà trên phần cứng phổ thông.
  - Xử lý đa luồng an toàn (Multi-threading + Queue), giao diện không bao giờ bị đơ/treo khi tải nặng.
- **Quy trình Huấn luyện & Tinh chỉnh linh hoạt (Colab 1-Click)**:
  - Cung cấp sẵn Notebooks huấn luyện trên Google Colab với GPU T4 miễn phí trong thư mục `notebooks/`.

---

## 📁 CẤU TRÚC THƯ MỤC DỰ ÁN

```
ChuongTrinhPhatHienLanDuong/
├── main.py                     # Điểm khởi chạy ứng dụng chính
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── README.md                   # Tài liệu hướng dẫn dự án
├── .gitignore                  # Cấu hình loại trừ file nặng / cache
│
├── app/                        # Tầng giao diện người dùng (GUI)
│   ├── __init__.py
│   └── gui.py                  # Giao diện Tkinter hiện đại 50:50, đa luồng & log
│
├── lane_detection/             # Tầng suy luận AI YOLOPv2 lõi
│   ├── __init__.py
│   ├── yolopv2_detector.py     # Lớp suy luận YOLOPv2, tiền xử lý ảnh & vẽ kết quả
│   └── video_processor.py      # Xử lý video, frame-skipping & callback tiến trình
│
├── models/                     # Kiến trúc mạng nơ-ron YOLOPv2
│   ├── __init__.py
│   └── yolopv2_utils.py        # Hậu xử lý Non-Maximum Suppression (NMS), anchor boxes
│
├── utils/                      # Tiện ích bổ trợ
│   ├── __init__.py
│   └── file_utils.py           # Kiểm tra định dạng tệp tin, tạo tên file kết quả
│
├── training/                   # Pipeline huấn luyện & fine-tuning
│   ├── train_yolopv2.py        # Script huấn luyện Two-Phase Fine-Tuning
│   ├── losses.py               # Hàm mất mát đa nhiệm (Focal Loss + Dice Loss)
│   ├── dataset.py              # PyTorch Dataset & DataLoader
│   └── prepare_tusimple_dataset.py # Script trích xuất và chuẩn hóa nhãn TuSimple
│
├── notebooks/                  # Notebooks huấn luyện trên Google Colab
│   ├── YOLOPv2_TuSimple_Training_Colab.ipynb      # Huấn luyện trên bộ dữ liệu TuSimple
│   └── YOLOPv2_MultiDomain_Training_Colab.ipynb   # Huấn luyện thích ứng đa miền Việt Nam
│
├── data/                       # Dữ liệu thử nghiệm mẫu
│   ├── images/                 # Ảnh mẫu kiểm thử (Vo-Chi-Cong-06.jpg, ...)
│   └── videos/                 # Video mẫu kiểm thử (video_test_traffic.mp4, ...)
│
├── weights/                    # Thư mục chứa trọng số mô hình (.pt)
│   └── .gitkeep
│
├── results/                    # Thư mục lưu kết quả xuất ra (ảnh & video)
│   ├── images/
│   └── videos/
│
└── tests/                      # Kiểm thử tự động
    ├── test_phase3_integration.py # Kiểm thử tích hợp toàn diện hệ thống
    ├── test_yolopv2_standalone.py # Kiểm thử độc lập mô hình YOLOPv2
    └── run_benchmark.py           # Đo đạc hiệu năng FPS và thời gian suy luận
```

---

## ⚙️ CÀI ĐẶT VÀ KHỞI CHẠY

### 1. Yêu cầu môi trường
- Python 3.10 - 3.12 (64-bit)
- Hệ điều hành: Windows 10/11 hoặc Linux / macOS

### 2. Cài đặt thư viện
```bash
# Di chuyển vào thư mục dự án
cd ChuongTrinhPhatHienLanDuong

# Tạo và kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\activate

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

### 3. Tải file Trọng số (Weights Checkpoint)
Đặt file trọng số vào thư mục `weights/`:
- **`weights/yolopv2.pt`** (156 MB): Trọng số gốc tải từ [YOLOPv2 Official Release](https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt)
- **`weights/yolopv2_multidomain.pt`** (156 MB): Trọng số đã tinh chỉnh sau khi chạy Colab.

*(Lưu ý: Các file `.pt` có dung lượng >100MB được loại trừ trong `.gitignore` theo quy định của GitHub).*

### 4. Khởi chạy ứng dụng
```bash
python main.py
```

### 5. Chạy kiểm thử tự động
```bash
python tests/test_phase3_integration.py
```

---

## 📖 HƯỚNG DẪN SỬ DỤNG GIAO DIỆN

1. **Xử lý Ảnh đơn**:
   - Chọn chế độ **📷 Hình ảnh** $\rightarrow$ Bấm **"📂 Mở tệp..."** và chọn ảnh trong `data/images/`.
   - Tùy chọn hiển thị: Tích chọn vạch đường, làn xe, hoặc khung viền xe.
   - Bấm **"▶ BẮT ĐẦU XỬ LÝ"** để nhận diện làn đường.
   - Bấm **"💾 Lưu kết quả"** để lưu ảnh đã nhận diện vào `results/images/`.

2. **Xử lý Video**:
   - Chọn chế độ **🎥 Video** $\rightarrow$ Bấm **"📂 Mở tệp..."** và chọn video trong `data/videos/`.
   - Bấm **"▶ BẮT ĐẦU XỬ LÝ"** để phát hiện làn đường liên tục theo thời gian thực.
   - Theo dõi thanh tiến trình %, số lượng phương tiện và FPS ở cột bên trái.
   - Bấm **"⏹ Dừng"** bất kỳ lúc nào nếu muốn dừng sớm. Video hoàn thành sẽ tự động lưu trong `results/videos/`.

---

## 🛠️ CÔNG NGHỆ ÁP DỤNG

- **Ngôn ngữ**: Python 3.12 (64-bit)
- **Deep Learning Framework**: PyTorch 2.x & TorchScript JIT, TorchVision
- **Kiến trúc AI**: YOLOPv2 (E-ELAN Backbone, BiFPN Neck, Multi-Task Detection Heads)
- **Hàm mất mát huấn luyện**: Focal Loss, Multi-Class Dice Loss, Binary Cross-Entropy
- **Thị giác máy tính**: OpenCV (tối ưu hóa tiền xử lý và hậu xử lý hình ảnh)
- **Giao diện người dùng**: Tkinter & ttk (Giao diện chuẩn hiện đại, cân đối 50:50)

---

## 👥 THÀNH VIÊN THỰC HIỆN

- **Nhóm**: Nhóm 1  
- **Lớp**: 24DHTT02  
- **Học kỳ**: HK7 (Năm học 2026 – 2027)  
- **Môn học**: Xử lý ảnh và Thị giác máy tính  

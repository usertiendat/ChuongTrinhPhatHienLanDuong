# -*- coding: utf-8 -*-
"""
gt_chuan_xac.py — Thay thế hàm `tao_ground_truth_chuan_xac(ten_anh, img)` trong
prepare_multidomain_dataset.py.

Ý tưởng chính
-------------
1. MẶT ĐƯỜNG (Drivable Area)
   * Mỗi ranh giới đường là một "cạnh" (edge) được mô tả bằng vài mốc neo (toạ độ chuẩn hoá 0..1):
       - kind="line"  : cạnh đường thẳng -> khớp bình phương tối thiểu tổng (TLS) thành MỘT đường thẳng
                        duy nhất (trong phối cảnh, cạnh đường thẳng luôn là đường thẳng; hết "gãy khúc").
       - kind="curve" : cạnh cong -> spline Catmull-Rom hướng tâm (centripetal) rồi làm trơn theo cung.
       - kind="raw"   : điểm góc khung ảnh (không làm trơn).
   * Vẽ polygon ở độ phân giải 1/16 pixel (fillPoly shift=4 + LINE_AA) rồi nhị phân hoá -> biên mịn.
   * (refine=True) Tinh chỉnh bám pixel thật: dọc pháp tuyến của cạnh, tìm vị trí mà phía trong giống
     nhựa đường (mô hình Gauss trong Lab) và phía ngoài khác (cây xanh, gạch vỉa hè...). Dùng quy hoạch
     động Viterbi + phạt lệch khỏi mốc neo + phạt nhảy -> bám mép thật nhưng không "run".

2. VẠCH KẺ LÀN (Lane Line) theo luật viễn cận 3D
   * Mỗi vạch là đường thẳng qua điểm tụ (xh, vy). Với s = (y-vy)/(h-vy) (s=1 ở đáy ảnh):
         độ sâu tương đối  u = 1/s ,   y = vy + (h-vy)/u ,   x = xh + (xb-xh)*s ,   bề rộng = wb*s
     Vạch đứt chia đều theo u (chiều dài thực đều nhau) => ảnh xa thì mỏng, ngắn, sát nhau;
     gần thì dày, dài. Mỗi vạch đứt là hình thang đúng phối cảnh.
   * (refine=True) Tìm (xb, xh, pha, chu kỳ) khớp nhất với "bằng chứng sơn trắng" (top-hat độ sáng,
     độ bão hoà thấp, chỉ trong lòng đường, ngoài hộp xe). Vạch không có sơn thật -> bị loại
     (trừ vạch gắn keep_if_unsupported=True).

3. OCCLUSION: vạch bị xoá dưới thân ô tô / bus / tải (cả hộp) và phần thân xe máy (hộp thu nhỏ).

4. BOUNDING BOX: giữ NGUYÊN 100% logic cũ (cùng ngưỡng, cùng bộ lọc, cùng NMS, cùng hộp thủ công).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

# --- giữ nguyên cách nạp YOLO của file gốc (lười nạp để test không cần ultralytics) ---
ROOT = Path(__file__).resolve().parent.parent
_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(str(ROOT / "yolov8x.pt"))
    return _yolo_model


# =============================================================================
# 0. CẤU HÌNH HÌNH HỌC TỪNG ẢNH  (toạ độ CHUẨN HOÁ u=x/w, v=y/h  -> không phụ thuộc độ phân giải)
# =============================================================================
SPECS = {
    # ---- 1. Đại lộ Cộng Hòa (góc cao, hai chiều xe, dải phân cách giữa) ----
    "images (1).jpg": dict(
        horizon=0.010,
        band=0.030,
        edges=[
            dict(kind="line", refine=True, ext=(0.0, 0.05),            # trái: xa -> gần
                 pts=[(0.4871, 0.0496), (0.4086, 0.1830), (0.3657, 0.3227), (0.3300, 0.4752),
                      (0.2800, 0.6531), (0.2300, 0.8056), (0.1600, 0.9962)]),
            dict(kind="line", refine=True, ext=(0.05, 0.0),            # phải: gần -> xa
                 pts=[(0.8443, 0.9962), (0.7871, 0.8183), (0.7300, 0.6531), (0.6800, 0.5260),
                      (0.6371, 0.4117), (0.6014, 0.3100), (0.5693, 0.2084), (0.5443, 0.1321),
                      (0.5286, 0.0813), (0.5171, 0.0496)]),
        ],
        lanes=[
            dict(xb=0.3464, xh=0.5007, dash=0.07, gap=0.11, wb=0.0045, keep_if_unsupported=True),
            dict(xb=0.8150, xh=0.5007, dash=0.07, gap=0.11, wb=0.0045, keep_if_unsupported=True),
        ],
        # dải phân cách giữa hai chiều xe: KHÔNG phải mặt đường xe chạy
        exclude=[dict(xb=0.6000, xh=0.5007, wb=0.010)],
    ),
    # ---- 2. Nội thành TP.HCM (ngã tư, dải phân cách lan can bên trái) ----
    "Giao-Thong-Tphcm-10.jpg": dict(
        horizon=0.140,
        band=0.025,
        edges=[
            dict(kind="curve", refine=True, ext=(0.0, 0.05),           # trái (chân lan can): xa -> gần
                 pts=[(0.6836, 0.2156), (0.6017, 0.2453), (0.5651, 0.3009), (0.5433, 0.3697),
                      (0.5214, 0.4408), (0.4887, 0.4941), (0.4302, 0.5474), (0.1333, 0.7820),
                      (0.0000, 0.9988)]),
            dict(kind="curve", refine=True, ext=(0.05, 0.0),           # phải (mép bó vỉa): gần -> xa
                 pts=[(0.9189, 0.9988), (0.8940, 0.8555), (0.8550, 0.6481), (0.8316, 0.5592),
                      (0.8161, 0.4882), (0.8098, 0.3815), (0.8083, 0.3341), (0.7911, 0.2334)]),
        ],
        lanes=[
            dict(xb=0.3242, xh=0.88, dash=0.030, gap=0.090, wb=0.009, keep_if_unsupported=False),
            dict(xb=0.4388, xh=0.88, dash=0.030, gap=0.090, wb=0.009, keep_if_unsupported=True),
            dict(xb=0.5963, xh=0.88, dash=0.030, gap=0.090, wb=0.009, keep_if_unsupported=False),
            dict(xb=0.7319, xh=0.88, dash=0.030, gap=0.090, wb=0.009, keep_if_unsupported=True),
        ],
        exclude=[],
    ),
    # ---- 3. Võ Chí Công (bó vỉa cong bên trái, dải phân cách cây xanh bên phải) ----
    "Vo-Chi-Cong-06.jpg": dict(
        horizon=0.090,
        band=0.020,
        edges=[
            dict(kind="curve", refine=True, ext=(0.0, 0.04),           # trái: xa -> gần
                 pts=[(0.4690, 0.0938), (0.4310, 0.1876), (0.2875, 0.3750), (0.1250, 0.5630),
                      (0.0310, 0.7500), (0.0000, 1.0000)]),
            dict(kind="raw", pts=[(-0.06, 1.08), (1.08, 1.08)]),        # đáy khung ảnh
            dict(kind="curve", refine=True, ext=(0.04, 0.0),           # phải: gần -> xa
                 pts=[(1.0000, 0.7970), (0.9500, 0.7040), (0.8875, 0.5630), (0.8250, 0.4690),
                      (0.7500, 0.3750), (0.6810, 0.2810), (0.6125, 0.1876), (0.5500, 0.0938)]),
        ],
        lanes=[   # vạch trắng thật: ranh giới làn xe máy | làn ô tô, và 2 vạch giữa các làn ô tô
            dict(xb=0.2188, xh=0.4062, dash=0.17, gap=0.42, wb=0.009, keep_if_unsupported=True),
            dict(xb=0.5130, xh=0.5134, dash=0.18, gap=0.42, wb=0.009, keep_if_unsupported=True),
            dict(xb=0.7306, xh=0.5134, dash=0.18, gap=0.42, wb=0.009, keep_if_unsupported=False),
        ],
        exclude=[],
    ),
    # ---- 4. Tuyến đường có biển phân làn (bồn hoa bên phải, dải phân cách bên trái) ----
    "images.jpg": dict(
        horizon=0.566,
        band=0.030,
        edges=[
            dict(kind="line", refine=True, ext=(0.0, 0.07),            # trái = mép dải phân cách: xa -> biên trái
                 pts=[(0.5166, 0.5640), (0.3902, 0.5806), (0.2954, 0.5924), (0.1848, 0.6031),
                      (0.0900, 0.6137), (0.0000, 0.6244)]),
            dict(kind="raw", pts=[(-0.08, 1.08)]),                      # góc dưới-trái ngoài khung
            dict(kind="line", refine=True, ext=(0.06, 0.0),            # phải = mép nhựa sát gờ vỉa hè
                 pts=[(0.8942, 0.9988), (0.8483, 0.9455), (0.8088, 0.8981), (0.7536, 0.8329),
                      (0.6983, 0.7654), (0.6430, 0.7062), (0.5924, 0.6517), (0.5403, 0.5960),
                      (0.5166, 0.5640)]),
        ],
        lanes=[   # vạch đứt giữa lòng đường; KHÔNG vẽ vạch đỏ lên gờ vỉa hè sơn trắng-đỏ
            dict(xb=0.1216, xh=0.3925, dash=0.06, gap=0.14, wb=0.007, keep_if_unsupported=True),
        ],
        exclude=[],
    ),
}

# Đường đèo (images (2).jpg): chưa có ảnh minh chứng -> giữ hình học gốc nhưng làm trơn bằng spline
DEO_POLY = [(0.0, 1.0), (0.05, 0.85), (0.4, 0.48), (0.1, 0.25), (0.4, 0.20), (0.65, 0.38), (1.0, 0.65), (1.0, 1.0)]
DEO_LANE = [(0.0, 0.95), (0.5, 0.45), (0.2, 0.25)]

# =============================================================================
# 1. HÌNH HỌC CƠ BẢN
# =============================================================================
SHIFT = 4
FACT = 1 << SHIFT


def _fill_poly(mask, pts, thr=100):
    """Vẽ polygon sub-pixel (1/16 px) vào mask 0/255 rồi nhị phân hoá."""
    tmp = np.zeros_like(mask)
    p = np.round(np.asarray(pts, np.float64) * FACT).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(tmp, [p], 255, lineType=cv2.LINE_AA, shift=SHIFT)
    mask[tmp >= thr] = 255


def _dedupe(P, eps=1e-6):
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > eps:
            keep.append(i)
    return P[keep]


def _catmull_rom(P, step=1.0, alpha=0.5):
    """Spline Catmull-Rom hướng tâm đi qua mọi mốc, lấy mẫu ~step pixel."""
    P = _dedupe(np.asarray(P, np.float64))
    if len(P) < 3:
        return P
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** alpha
        t2 = t1 + np.linalg.norm(p2 - p1) ** alpha
        t3 = t2 + np.linalg.norm(p3 - p2) ** alpha
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        t = np.linspace(t1, t2, n, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    out.append(P[-1][None])
    return np.vstack(out)


def _catmull_rom_closed(P, step=2.0, alpha=0.5):
    P = _dedupe(np.asarray(P, np.float64))
    n = len(P)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** alpha
        t2 = t1 + np.linalg.norm(p2 - p1) ** alpha
        t3 = t2 + np.linalg.norm(p3 - p2) ** alpha
        m = max(2, int(np.linalg.norm(p2 - p1) / step))
        t = np.linspace(t1, t2, m, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    return np.vstack(out)


def _resample(P, step=1.0):
    P = _dedupe(np.asarray(P, np.float64))
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    n = max(2, int(s[-1] / step) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], 1)


def _smooth_polyline(P, sigma):
    """Làm trơn theo cung (giữ nguyên 2 đầu mút)."""
    if len(P) < 5 or sigma <= 0:
        return P
    Q = np.stack([gaussian_filter1d(P[:, k], sigma, mode="nearest") for k in range(2)], 1)
    Q[0], Q[-1] = P[0], P[-1]
    return Q


def _fit_line(P):
    """Đường thẳng TLS: trả (tâm, hướng đơn vị)."""
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    return c, vt[0]


def _line_segment(fit_pts, ext_px, step=1.0):
    """Đoạn thẳng TLS nối hình chiếu của mốc đầu và mốc cuối, kéo dài ext_px=(đầu, cuối)."""
    c, d = _fit_line(fit_pts)
    a = c + d * ((fit_pts[0] - c) @ d)
    b = c + d * ((fit_pts[-1] - c) @ d)
    u = (b - a) / (np.linalg.norm(b - a) + 1e-9)
    a = a - u * ext_px[0]
    b = b + u * ext_px[1]
    return _resample(np.vstack([a, b]), step)


def _edge_points(edge, w, h):
    """Điểm dày (~1px) của một cạnh, đơn vị pixel."""
    diag = float(np.hypot(w, h))
    P = np.asarray(edge["pts"], np.float64) * [w, h]
    kind = edge["kind"]
    ext = edge.get("ext", (0.0, 0.0))
    if kind == "raw":
        return P
    if kind == "line":
        return _line_segment(P, (ext[0] * diag, ext[1] * diag))
    # curve
    if ext[0] > 0:
        d = P[0] - P[1]
        P = np.vstack([P[0] + d / np.linalg.norm(d) * ext[0] * diag, P])
    if ext[1] > 0:
        d = P[-1] - P[-2]
        P = np.vstack([P, P[-1] + d / np.linalg.norm(d) * ext[1] * diag])
    C = _catmull_rom(P, 1.0)
    return _smooth_polyline(C, sigma=max(2.0, 0.006 * diag))


def _assemble(edges_px):
    return np.vstack(edges_px)


# =============================================================================
# 2. TINH CHỈNH BIÊN MẶT ĐƯỜNG THEO PIXEL (xác suất nhựa đường + Viterbi)
# =============================================================================
def _veh_mask(h, w, boxes_px, pad=3):
    m = np.zeros((h, w), np.uint8)
    for (x1, y1, x2, y2) in boxes_px:
        m[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)] = 255
    return m


def _asphalt_likelihood(img, da_prior, veh):
    """A(x,y) in [0,1]: giống màu nhựa đường (Gauss Lab, ước lượng bền) và không phải cây xanh."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    core = cv2.erode(da_prior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    sel = (core > 0) & (veh == 0)
    if np.any(core > 0):
        sel &= lab[:, :, 0] < np.percentile(lab[:, :, 0][core > 0], 92)
    pix = lab[sel]
    if pix.shape[0] < 800:
        return None
    mu = np.median(pix, 0)
    cov = np.cov(pix.T) + np.eye(3) * 4.0
    for _ in range(2):
        inv = np.linalg.inv(cov)
        d = pix - mu
        d2 = np.einsum("ni,ij,nj->n", d, inv, d)
        keep = d2 < np.percentile(d2, 90)
        mu = pix[keep].mean(0)
        cov = np.cov(pix[keep].T) + np.eye(3) * 4.0
    inv = np.linalg.inv(cov)
    d = lab - mu
    D2 = np.einsum("hwi,ij,hwj->hw", d, inv, d)
    A = np.exp(-D2 / 8.0).astype(np.float32)
    green = (hsv[:, :, 0] >= 33) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] >= 45)
    A[green] *= 0.15
    A = cv2.GaussianBlur(A, (0, 0), 1.2)
    A[veh > 0] = 0.5
    return A


def _viterbi(U, lam):
    n, K = U.shape
    cost = U[0].copy()
    back = np.zeros((n, K), np.int8)
    for i in range(1, n):
        best = cost.copy()
        arg = np.zeros(K, np.int8)
        up = np.full(K, np.inf)
        up[1:] = cost[:-1] + lam
        m = up < best
        best = np.where(m, up, best)
        arg = np.where(m, 1, arg)
        dn = np.full(K, np.inf)
        dn[:-1] = cost[1:] + lam
        m = dn < best
        best = np.where(m, dn, best)
        arg = np.where(m, -1, arg)
        cost = best + U[i]
        back[i] = arg
    k = int(np.argmin(cost))
    out = np.empty(n, np.int32)
    for i in range(n - 1, -1, -1):
        out[i] = k
        k -= int(back[i][k])
    return out


def _refine_edge(P, kind, A, da_prior, B, win, mu=0.5, lam=0.08, min_conf=0.10):
    h, w = A.shape
    n = len(P)
    if n < 20:
        return P
    T = np.gradient(_smooth_polyline(P, 4.0), axis=0)
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    N = np.stack([-T[:, 1], T[:, 0]], 1)

    def samp(M, pts):
        return map_coordinates(M, [pts[..., 1], pts[..., 0]], order=1, mode="nearest")

    pin = samp(da_prior.astype(np.float32), P + N * 6)
    pout = samp(da_prior.astype(np.float32), P - N * 6)
    if pin.mean() < pout.mean():
        N = -N

    d = np.arange(-B, B + 1)
    K = len(d)
    t_in = np.arange(1, win + 1)
    base = P[:, None, None, :] + N[:, None, None, :] * (d[None, :, None, None] + t_in[None, None, :, None])
    base_o = P[:, None, None, :] + N[:, None, None, :] * (d[None, :, None, None] - t_in[None, None, :, None])
    a_in = samp(A, base).mean(-1)
    a_out = samp(A, base_o).mean(-1)
    score = a_in - a_out
    inside_frame = (P[:, 0] > win + 2) & (P[:, 0] < w - win - 3) & (P[:, 1] > win + 2) & (P[:, 1] < h - win - 3)
    score[~inside_frame] = 0.0
    U = -score + mu * (d[None, :] / float(B)) ** 2
    idx = _viterbi(U.astype(np.float32), lam)
    off = d[idx].astype(np.float64)
    conf = score[np.arange(n), idx] >= min_conf
    conf &= inside_frame
    if conf.mean() < 0.25:
        return P
    Q = P + N * off[:, None]
    if kind == "line":
        pts = Q[conf]
        for _ in range(3):
            c, dd = _fit_line(pts)
            r = np.abs((pts - c) @ np.array([-dd[1], dd[0]]))
            pts = pts[r <= max(1.5, 2.5 * r.std())] if len(pts) > 20 else pts
        c, dd = _fit_line(pts)
        a = c + dd * ((P[0] - c) @ dd)
        b = c + dd * ((P[-1] - c) @ dd)
        return _resample(np.vstack([a, b]), 1.0)
    off = np.where(conf, off, 0.0)
    off = gaussian_filter1d(off, sigma=max(6.0, 0.02 * n), mode="nearest")
    return P + N * off[:, None]


# =============================================================================
# 3. VẠCH KẺ THEO LUẬT VIỄN CẬN
# =============================================================================
def _dash_intervals(dash, gap, phase, scale, u_max):
    P = (dash + gap) * scale
    d = dash * scale
    u = 1.0 - (phase % 1.0) * P
    out = []
    while u < u_max:
        a, b = max(u, 0.97), min(u + d, u_max)
        if b > a:
            out.append((a, b))
        u += P
    return out


def _quad(xb, xh, vy, h, wb, u1, u2):
    pts = []
    for u, sign in ((u2, -1), (u1, +1)):
        s = 1.0 / u
        y = vy + (h - vy) * s
        x = xh + (xb - xh) * s
        t = max(wb * s, 1.3)
        pts.append((x - t / 2, y))
        pts.append((x + t / 2, y))
    (xa, ya), (xb_, yb), (xc, yc), (xd, yd) = pts
    return np.array([(xa, ya), (xb_, yb), (xd, yd), (xc, yc)], np.float64)


def _draw_lane(mask, lane, vy_px, ytop_px, w, h, solid=False, phase=0.0, scale=1.0):
    xb, xh, wb = lane["xb"] * w, lane["xh"] * w, lane["wb"] * w
    s_min = max((ytop_px - vy_px) / (h - vy_px), 0.02)
    u_max = 1.0 / s_min
    if solid:
        ivs = [(0.97, u_max)]
    else:
        ivs = _dash_intervals(lane["dash"], lane["gap"], phase, scale, u_max)
    for (u1, u2) in ivs:
        _fill_poly(mask, _quad(xb, xh, vy_px, h, wb, u1, u2), thr=90)


def _paint_evidence(img, road, veh):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    k = max(9, int(0.014 * w) | 1)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    E = th.astype(np.float32) * (hsv[:, :, 1] < 80)
    roi = (cv2.dilate(road, np.ones((5, 5), np.uint8)) > 0) & (veh == 0)
    E[~roi] = 0
    vals = E[roi]
    if vals.size < 200:
        return None
    ref = max(float(np.percentile(vals, 99.0)), 20.0)
    E = np.clip(E / ref, 0, 1)
    return cv2.GaussianBlur(E, (0, 0), 1.5)


def _line_score(E, xb, xh, vy, ys, w):
    s = (ys - vy) / (E.shape[0] - vy)
    x = xh + (xb - xh) * s
    ok = (x >= 1) & (x < w - 2)
    if ok.sum() < 20:
        return -1.0
    v = E[ys[ok], np.round(x[ok]).astype(int)]
    return float((v * s[ok]).sum() / s[ok].sum())


def _snap_lane(E, lane, vy_px, ytop_px, w, h, B):
    ys = np.arange(int(ytop_px) + 3, h - 1, 2)
    xb0, xh0 = lane["xb"] * w, lane["xh"] * w
    cand = {}
    for dxb in range(-B, B + 1, 2):
        for dxh in range(-3 * B, 3 * B + 1, 6):
            cand[(dxb, dxh)] = _line_score(E, xb0 + dxb, xh0 + dxh, vy_px, ys, w)
    best = max(cand, key=cand.get)
    for dxb in range(best[0] - 2, best[0] + 3):
        for dxh in range(best[1] - 4, best[1] + 5, 2):
            cand[(dxb, dxh)] = _line_score(E, xb0 + dxb, xh0 + dxh, vy_px, ys, w)
    best = max(cand, key=cand.get)
    sc = np.array(list(cand.values()))
    supported = cand[best] >= 0.10 and cand[best] >= 1.4 * float(np.median(sc)) + 0.02
    new = dict(lane)
    new["xb"] = (xb0 + best[0]) / w
    new["xh"] = (xh0 + best[1]) / w
    if not supported:
        return lane, 0.0, 1.0, False
    xb, xh = new["xb"] * w, new["xh"] * w
    s_min = max((ytop_px - vy_px) / (h - vy_px), 0.02)
    u_max = 1.0 / s_min
    best_c, best_v = (0.0, 1.0), -9.0
    for q in (0.8, 0.9, 1.0, 1.1, 1.25):
        for ph in np.linspace(0, 1, 16, endpoint=False):
            ivs = _dash_intervals(lane["dash"], lane["gap"], ph, q, u_max)
            if not ivs:
                continue
            dsamp, gsamp = [], []
            for i, (a, b) in enumerate(ivs):
                dsamp += list(np.linspace(a, b, 6))
                if i + 1 < len(ivs):
                    gsamp.append(0.5 * (b + ivs[i + 1][0]))
            def val(us):
                us = np.asarray(us)
                s = 1.0 / us
                y = np.clip(np.round(vy_px + (h - vy_px) * s).astype(int), 0, h - 1)
                x = np.clip(np.round(xh + (xb - xh) * s).astype(int), 0, w - 1)
                return E[y, x].mean() if len(us) else 0.0
            v = val(dsamp) - 0.6 * (val(gsamp) if gsamp else 0.0)
            if v > best_v:
                best_v, best_c = v, (ph, q)
    return new, best_c[0], best_c[1], True


# =============================================================================
# 4. BOUNDING BOX — GIỮ NGUYÊN 100% LOGIC CỦA FILE GỐC
# =============================================================================
_COCO_TO_CLS = {2: 0, 3: 1, 5: 2, 7: 3}


def _reject_cong_hoa(x1, y1, x2, y2, xc, yc, cls):
    return (xc < 160 and yc > 300) or (yc < 45 or xc > 700)


def _reject_tphcm(x1, y1, x2, y2, xc, yc, cls):
    return x2 > 870 or yc < 150


def _reject_vcc(x1, y1, x2, y2, xc, yc, cls):
    return (x1 < 20 and y1 < 200) or (cls == 1 and xc > 550) or (xc < 550)


_DET_RULES = {
    "images (1).jpg": _reject_cong_hoa,
    "Giao-Thong-Tphcm-10.jpg": _reject_tphcm,
    "Vo-Chi-Cong-06.jpg": _reject_vcc,
}

_MOTOS_VCC = [(220, 640, 315, 810), (500, 560, 590, 710), (640, 490, 710, 620)]
_MOTOS_HW = [(250, 370, 275, 415), (225, 355, 238, 380), (212, 355, 222, 375)]
_CARS_HW = [(3, 0, 320, 50, 425), (0, 80, 360, 110, 395), (0, 115, 360, 145, 395),
            (3, 150, 350, 185, 395), (3, 175, 350, 200, 390)]


def _detect_boxes(ten_anh, img, w, h):
    boxes = []
    if ten_anh in _DET_RULES:
        reject = _DET_RULES[ten_anh]
        res = _get_yolo().predict(img, conf=0.18, classes=[2, 3, 5, 7], verbose=False)[0]
        raw_boxes = []
        for box in res.boxes:
            c = int(box.cls[0])
            if c not in _COCO_TO_CLS:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if reject(x1, y1, x2, y2, xc, yc, _COCO_TO_CLS[c]):
                continue
            raw_boxes.append((_COCO_TO_CLS[c], conf, x1, y1, x2, y2))
        raw_boxes.sort(key=lambda x: x[1], reverse=True)
        keep = []
        for b in raw_boxes:
            overlap = False
            for kb in keep:
                ix1, iy1 = max(b[2], kb[2]), max(b[3], kb[3])
                ix2, iy2 = min(b[4], kb[4]), min(b[5], kb[5])
                if ix2 > ix1 and iy2 > iy1:
                    i_area = (ix2 - ix1) * (iy2 - iy1)
                    a1 = (b[4] - b[2]) * (b[5] - b[3])
                    a2 = (kb[4] - kb[2]) * (kb[5] - kb[3])
                    if i_area / float(a1 + a2 - i_area) > 0.50:
                        overlap = True
                        break
            if not overlap:
                keep.append(b)
        for b in keep:
            boxes.append([b[0], ((b[2] + b[4]) / 2.0) / w, ((b[3] + b[5]) / 2.0) / h,
                          (b[4] - b[2]) / w, (b[5] - b[3]) / h])
    if ten_anh == "Vo-Chi-Cong-06.jpg":
        for mx1, my1, mx2, my2 in _MOTOS_VCC:
            boxes.append([1, ((mx1 + mx2) / 2.0) / w, ((my1 + my2) / 2.0) / h, (mx2 - mx1) / w, (my2 - my1) / h])
    elif ten_anh == "images.jpg":
        for mx1, my1, mx2, my2 in _MOTOS_HW:
            boxes.append([1, ((mx1 + mx2) / 2.0) / w, ((my1 + my2) / 2.0) / h, (mx2 - mx1) / w, (my2 - my1) / h])
        for c_id, cx1, cy1, cx2, cy2 in _CARS_HW:
            boxes.append([c_id, ((cx1 + cx2) / 2.0) / w, ((cy1 + cy2) / 2.0) / h, (cx2 - cx1) / w, (cy2 - cy1) / h])
    return boxes


def _boxes_px(boxes, w, h):
    out = []
    for b in boxes:
        cls_id, xc, yc, bw, bh = b
        x1 = int(max(0, (xc - bw / 2.0) * w)); y1 = int(max(0, (yc - bh / 2.0) * h))
        x2 = int(min(w, (xc + bw / 2.0) * w)); y2 = int(min(h, (yc + bh / 2.0) * h))
        out.append((int(cls_id), x1, y1, x2, y2))
    return out


# =============================================================================
# 5. LÕI: DỰNG MẶT NẠ TỪ SPEC
# =============================================================================
def build_masks(img, spec, boxes, refine=True, occlude_motorcycle=True):
    h, w = img.shape[:2]
    da = np.zeros((h, w), np.uint8)
    ll = np.zeros((h, w), np.uint8)
    bpx = _boxes_px(boxes, w, h)
    veh = _veh_mask(h, w, [b[1:] for b in bpx])

    # ---- 5.1 mặt đường từ các cạnh ----
    edge_pts = [_edge_points(e, w, h) for e in spec["edges"]]
    prior = np.zeros((h, w), np.uint8)
    _fill_poly(prior, _assemble(edge_pts))
    da[:] = prior

    if refine:
        A = _asphalt_likelihood(img, prior, veh)
        if A is not None:
            B = max(6, int(round(spec.get("band", 0.03) * w)))
            win = 6 + int(0.006 * w)
            for i, e in enumerate(spec["edges"]):
                if e.get("refine"):
                    edge_pts[i] = _refine_edge(edge_pts[i], e["kind"], A, prior, B, win)
            da[:] = 0
            _fill_poly(da, _assemble(edge_pts))

    # ---- 5.2 vạch kẻ làn ----
    vy_px = spec["horizon"] * h
    ys_top = np.where(da.any(1))[0]
    ytop_px = float(ys_top.min()) if len(ys_top) else 0.0
    E = _paint_evidence(img, da, veh) if refine else None
    Bl = max(4, int(round(0.025 * w)))
    for lane in spec["lanes"]:
        phase, scale, use = 0.0, 1.0, lane
        if E is not None:
            use, phase, scale, ok = _snap_lane(E, lane, vy_px, ytop_px, w, h, Bl)
            if not ok and not lane.get("keep_if_unsupported", False):
                continue
        _draw_lane(ll, use, vy_px, ytop_px, w, h, solid=False, phase=phase, scale=scale)

    # ---- 5.3 loại dải phân cách / vật cản khỏi mặt đường ----
    for ex in spec.get("exclude", []):
        strip = np.zeros((h, w), np.uint8)
        _draw_lane(strip, dict(xb=ex["xb"], xh=ex["xh"], wb=ex["wb"]), vy_px, ytop_px, w, h, solid=True)
        da[strip > 0] = 0
        ll[strip > 0] = 0

    # vạch chỉ nằm trong (lòng đường nở nhẹ 3px)
    ll &= cv2.dilate(da, np.ones((3, 3), np.uint8))

    # ---- 5.4 Occlusion: vạch lặn dưới thân xe ----
    for cls_id, x1, y1, x2, y2 in bpx:
        if cls_id in (0, 2, 3):
            ll[y1:y2, x1:x2] = 0
        elif cls_id == 1 and occlude_motorcycle:
            bw, bh = x2 - x1, y2 - y1
            ll[y1 + int(0.10 * bh):y2, x1 + int(0.12 * bw):x2 - int(0.12 * bw)] = 0
    return da, ll


def _build_deo(img):
    h, w = img.shape[:2]
    da = np.zeros((h, w), np.uint8)
    ll = np.zeros((h, w), np.uint8)
    P = np.array(DEO_POLY, np.float64) * [w, h]
    _fill_poly(da, _catmull_rom_closed(P, 2.0))
    C = _catmull_rom(np.array(DEO_LANE, np.float64) * [w, h], 1.0)
    for i in range(len(C) - 1):
        t = float(np.interp(C[i, 1], [0.25 * h, 0.95 * h], [1.5, 4.0]))
        cv2.line(ll, tuple(np.round(C[i]).astype(int)), tuple(np.round(C[i + 1]).astype(int)), 255,
                 max(1, int(round(t))), cv2.LINE_AA)
    ll = (ll >= 100).astype(np.uint8) * 255
    ll &= cv2.dilate(da, np.ones((3, 3), np.uint8))
    return da, ll


# =============================================================================
# 6. HÀM CÔNG KHAI (thay thế hàm cũ, cùng chữ ký trả về)
# =============================================================================
def tao_ground_truth_chuan_xac(ten_anh, img, refine=True, debug_path=None):
    h, w = img.shape[:2]
    boxes = _detect_boxes(ten_anh, img, w, h)
    if ten_anh in SPECS:
        da, ll = build_masks(img, SPECS[ten_anh], boxes, refine=refine)
    elif ten_anh == "images (2).jpg":
        da, ll = _build_deo(img)
    else:
        da = np.zeros((h, w), np.uint8)
        ll = np.zeros((h, w), np.uint8)

    if debug_path:
        vis = img.copy()
        cnts, _ = cv2.findContours(da, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        vis[ll > 0] = (0, 0, 255)
        for cls_id, x1, y1, x2, y2 in _boxes_px(boxes, w, h):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 1)
        cv2.imwrite(str(debug_path), vis)
    return boxes, da, ll

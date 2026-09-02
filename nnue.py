"""Mạng lượng giá kiểu NNUE.

Bộ đặc trưng "HalfKP rút gọn": với MỖI bên (Trắng/Đen) ta mã hoá thế cờ theo
góc nhìn của bên đó, chỉ số đặc trưng gồm 3 thành phần:

    (ô vua của bên đó -> nhóm)  x  (loại quân tương đối)  x  (ô của quân)
             KING_BUCKETS       x           10            x      64

"Loại quân tương đối" = 0..4 cho quân MÌNH (T,M,T,X,H) và 5..9 cho quân ĐỊCH;
vua không nằm trong đặc trưng vì đã dùng làm khoá nhóm. Toạ độ được lật dọc
cho Đen nên mạng học một biểu diễn duy nhất dùng chung cho cả hai bên.

Kiến trúc:
    features -> 256 (accumulator, chung trọng số cho 2 góc nhìn)
    [acc_bên_đi | acc_bên_kia] = 512 -> ClippedReLU -> 32 -> 32 -> 1

Đầu ra được nhân CP_SCALE để ra centipawn theo góc nhìn BÊN ĐANG ĐI.

torch chỉ cần khi HUẤN LUYỆN. Khi suy luận chỉ cần numpy (nhanh hơn nhiều
trong vòng lặp tìm kiếm) - xem lớp NNUEEvaluator.
"""

import numpy as np

from chess_core import WHITE, BLACK, KING

import os

HIDDEN = int(os.environ.get("NNUE_HIDDEN", 256))
L1, L2 = 32, 32
CP_SCALE = 410.0        # đầu ra mạng (đơn vị ~ "pawn") -> centipawn

# Số nhóm ô vua. Càng nhiều nhóm mạng càng biểu đạt mạnh nhưng càng cần nhiều
# dữ liệu: 1 -> 640 đặc trưng, 4 -> 2560, 64 -> 40960 (HalfKP đầy đủ).
#
# Đo thực tế trên 34k thế cờ, kiểm trên tập giữ lại (tương quan với nhãn
# tìm kiếm - càng cao càng tốt):
#     KB=1 HIDDEN=256  r=0.856      <- mặc định
#     KB=1 HIDDEN=128  r=0.849
#     KB=4 HIDDEN=256  r=0.804
#     KB=4 HIDDEN=128  r=0.802
# Với ít dữ liệu, mạng nhỏ tổng quát tốt hơn. Khi có vài trăm nghìn thế cờ
# hãy thử lại KB=4, và vài triệu thì mới đáng dùng KB=64.
KING_BUCKETS = int(os.environ.get("NNUE_KING_BUCKETS", 1))

# ô vua (đã lật theo góc nhìn) -> nhóm
if KING_BUCKETS == 1:
    KB = [0] * 64
elif KING_BUCKETS == 4:
    KB = [((sq & 7) >= 4) * 2 + ((sq >> 3) >= 4) for sq in range(64)]
elif KING_BUCKETS == 64:
    KB = list(range(64))
else:
    raise ValueError("KING_BUCKETS phải là 1, 4 hoặc 64")

NUM_FEATURES = KING_BUCKETS * 10 * 64


# Với mỗi góc nhìn: chỉ số quân -> độ lệch trong khối 640 của một nhóm vua.
# Bỏ qua vua (đã dùng làm khoá nhóm).
_PIECES = [p for p in range(12) if p % 6 != KING]
_OFF = [[0] * 12, [0] * 12]
for _persp in (WHITE, BLACK):
    for _p in _PIECES:
        _OFF[_persp][_p] = ((0 if (_p // 6) == _persp else 5) + _p % 6) * 64


def features(pos):
    """Trả về (idx_trắng, idx_đen): danh sách chỉ số đặc trưng đang bật.

    Đây là hàm nóng nhất khi tìm kiếm nên viết vòng lặp bit trực tiếp thay vì
    dùng generator bits() - tránh chi phí dựng khung yield cho mỗi quân cờ.
    """
    kw = pos.king_sq(WHITE)
    kb = pos.king_sq(BLACK)
    if kw < 0 or kb < 0:
        return [], []
    base_w = KB[kw] * 640
    base_b = KB[kb ^ 56] * 640
    off_w, off_b = _OFF[WHITE], _OFF[BLACK]
    pieces = pos.pieces
    idx_w, idx_b = [], []
    aw, ab = idx_w.append, idx_b.append
    for p in _PIECES:
        bb = pieces[p]
        if not bb:
            continue
        ow = base_w + off_w[p]
        ob = base_b + off_b[p]
        while bb:
            low = bb & -bb
            sq = low.bit_length() - 1
            aw(ow + sq)
            ab(ob + (sq ^ 56))
            bb ^= low
    return idx_w, idx_b


class NNUEEvaluator:
    """Suy luận thuần numpy. Dùng làm eval_fn cho Searcher."""

    def __init__(self, path, cache_bits=18):
        z = np.load(path)
        self.w0 = z["w0"].astype(np.float32)     # (NUM_FEATURES, HIDDEN)
        self.b0 = z["b0"].astype(np.float32)
        self.w1 = z["w1"].astype(np.float32)     # (2*HIDDEN, L1)
        self.b1 = z["b1"].astype(np.float32)
        self.w2 = z["w2"].astype(np.float32)
        self.b2 = z["b2"].astype(np.float32)
        self.w3 = z["w3"].astype(np.float32)
        self.b3 = z["b3"].astype(np.float32)
        self.cp_scale = float(z["cp_scale"]) if "cp_scale" in z else CP_SCALE

        # Kiến trúc phải khớp, nếu không mạng vẫn chạy nhưng cho ra số vô nghĩa
        kb = int(z["king_buckets"]) if "king_buckets" in z else KING_BUCKETS
        hid = int(z["hidden"]) if "hidden" in z else HIDDEN
        if (kb, hid) != (KING_BUCKETS, HIDDEN):
            raise ValueError(
                "Trọng số {} được huấn luyện với KING_BUCKETS={}, HIDDEN={} "
                "nhưng nnue.py đang đặt {} và {}.\nĐặt biến môi trường "
                "NNUE_KING_BUCKETS={} NNUE_HIDDEN={} rồi chạy lại."
                .format(path, kb, hid, KING_BUCKETS, HIDDEN, kb, hid))
        self.w0 = np.ascontiguousarray(self.w0)
        # bộ đệm dùng lại giữa các lần gọi: ma trận ở đây quá nhỏ nên chi phí
        # cấp phát của numpy ngang ngửa chi phí tính toán
        self._buf = np.empty(2 * HIDDEN, dtype=np.float32)
        self._b0x2 = np.concatenate((self.b0, self.b0))
        self._h1 = np.empty(self.w1.shape[1], dtype=np.float32)
        self._h2 = np.empty(self.w2.shape[1], dtype=np.float32)
        self.mask = (1 << cache_bits) - 1
        self.cache_key = np.zeros(self.mask + 1, dtype=np.uint64)
        self.cache_val = np.zeros(self.mask + 1, dtype=np.int32)
        self.hits = self.calls = 0

    def raw(self, pos):
        wi, bi = features(pos)
        w0 = self.w0
        acc_w = w0.take(wi, axis=0).sum(axis=0)
        acc_b = w0.take(bi, axis=0).sum(axis=0)
        buf = self._buf
        if pos.side == WHITE:
            buf[:HIDDEN] = acc_w
            buf[HIDDEN:] = acc_b
        else:
            buf[:HIDDEN] = acc_b
            buf[HIDDEN:] = acc_w
        buf += self._b0x2
        np.clip(buf, 0.0, 1.0, out=buf)
        h1 = self._h1
        np.dot(buf, self.w1, out=h1)
        h1 += self.b1
        np.clip(h1, 0.0, 1.0, out=h1)
        h2 = self._h2
        np.dot(h1, self.w2, out=h2)
        h2 += self.b2
        np.clip(h2, 0.0, 1.0, out=h2)
        return float((h2 @ self.w3)[0] + self.b3[0])

    def __call__(self, pos):
        """Điểm cp theo góc nhìn bên đang đi (khớp giao diện evaluate_stm)."""
        self.calls += 1
        k = pos.key
        slot = k & self.mask
        if self.cache_key[slot] == np.uint64(k) and self.cache_val[slot] != 0:
            self.hits += 1
            return int(self.cache_val[slot])
        cp = int(round(self.raw(pos) * self.cp_scale))
        cp = max(-3000, min(3000, cp)) or 1
        self.cache_key[slot] = np.uint64(k)
        self.cache_val[slot] = cp
        return cp

    def white_cp(self, pos):
        cp = self(pos)
        return cp if pos.side == WHITE else -cp


# --------------------------------------------------------- phần cần PyTorch

def build_torch_model():
    """Tạo mô hình PyTorch (chỉ gọi khi huấn luyện)."""
    import torch
    import torch.nn as nn

    class NNUEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ft = nn.EmbeddingBag(NUM_FEATURES, HIDDEN, mode="sum", include_last_offset=True)
            self.ft_bias = nn.Parameter(torch.zeros(HIDDEN))
            self.l1 = nn.Linear(2 * HIDDEN, L1)
            self.l2 = nn.Linear(L1, L2)
            self.l3 = nn.Linear(L2, 1)
            nn.init.normal_(self.ft.weight, std=0.01)

        def forward(self, idx_w, off_w, idx_b, off_b, stm):
            """stm: (B,1) = 1.0 nếu Trắng đi, 0.0 nếu Đen đi."""
            aw = self.ft(idx_w, off_w) + self.ft_bias
            ab = self.ft(idx_b, off_b) + self.ft_bias
            own = stm * aw + (1.0 - stm) * ab
            opp = stm * ab + (1.0 - stm) * aw
            x = torch.clamp(torch.cat([own, opp], dim=1), 0.0, 1.0)
            x = torch.clamp(self.l1(x), 0.0, 1.0)
            x = torch.clamp(self.l2(x), 0.0, 1.0)
            return self.l3(x)

    return NNUEModel()


def export_npz(model, path, cp_scale=CP_SCALE):
    """Xuất trọng số sang .npz để suy luận bằng numpy."""
    import torch
    with torch.no_grad():
        np.savez_compressed(
            path,
            w0=model.ft.weight.detach().cpu().numpy().astype(np.float16),
            b0=model.ft_bias.detach().cpu().numpy().astype(np.float32),
            w1=model.l1.weight.detach().cpu().numpy().T.astype(np.float32),
            b1=model.l1.bias.detach().cpu().numpy().astype(np.float32),
            w2=model.l2.weight.detach().cpu().numpy().T.astype(np.float32),
            b2=model.l2.bias.detach().cpu().numpy().astype(np.float32),
            w3=model.l3.weight.detach().cpu().numpy().T.astype(np.float32),
            b3=model.l3.bias.detach().cpu().numpy().astype(np.float32),
            cp_scale=np.float32(cp_scale),
            king_buckets=np.int32(KING_BUCKETS),
            hidden=np.int32(HIDDEN),
        )

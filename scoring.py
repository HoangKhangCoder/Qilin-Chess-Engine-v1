"""Ánh xạ điểm nội bộ (centipawn) -> thang 0..1000 cho bên Trắng.

Ý nghĩa của thang điểm: **S = 1000 x kỳ vọng kết quả ván cờ của Trắng**
    S = 1000  -> Trắng chiếu hết ngay trong nước đi này
    S = 500   -> hoàn toàn cân bằng
    S = 0     -> Đen chiếu hết ngay trong nước đi này

Vì Trắng đi trước nên vị trí ban đầu KHÔNG cân bằng: lợi thế tiên thủ được
quy đổi thành đúng 5 điểm -> 505. Con số 5 này không phải hằng số cộng thêm
một cách tuỳ tiện, nó là một *hằng số hiệu chỉnh* CALIB đặt trong hàm sigmoid
sao cho hàm lượng giá của chính engine, khi chấm vị trí ban đầu, ra đúng 505.
Nhờ vậy đổi hàm lượng giá (thủ công <-> mạng nơ-ron) thì mốc 505 vẫn giữ nguyên.
"""

import math

# ------------------------------------------------------------------ hằng số

SCALE = 310.0          # cp tương ứng 1 đơn vị logit; hiệu chỉnh từ dữ liệu ván thật
START_SCORE = 505.0    # điểm chuẩn của vị trí xuất phát
NORMAL_LO, NORMAL_HI = 10, 990   # dải cho thế cờ không có chiếu hết ép buộc
MATE_FLOOR = 991                 # chiếu hết xa nhất vẫn > mọi ưu thế vật chất

MATE_VALUE = 1_000_000           # điểm cp nội bộ cho chiếu hết
MATE_BOUND = MATE_VALUE - 1000   # |cp| lớn hơn ngưỡng này => là điểm chiếu hết

_CALIB = 0.0    # bù trừ hiệu chỉnh, gán bởi calibrate()


def calibrate(cp_start):
    """Đặt CALIB sao cho cp_start (điểm engine chấm vị trí đầu) -> đúng 505."""
    global _CALIB
    target_logit = math.log(START_SCORE / (1000.0 - START_SCORE))
    _CALIB = SCALE * target_logit - cp_start
    return _CALIB


def win_expectancy(cp):
    """Kỳ vọng kết quả của Trắng (thắng=1, hoà=0.5, thua=0) từ điểm cp."""
    x = (cp + _CALIB) / SCALE
    if x > 40:
        return 1.0
    if x < -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def mate_distance(cp):
    """Số NƯỚC ĐI tới chiếu hết, dương = Trắng chiếu, âm = Đen chiếu, 0 = không có."""
    if cp > MATE_BOUND:
        plies = MATE_VALUE - cp
        return (plies + 1) // 2
    if cp < -MATE_BOUND:
        plies = MATE_VALUE + cp
        return -((plies + 1) // 2)
    return 0


def cp_to_score(cp):
    """Điểm cp (góc nhìn Trắng) -> điểm 0..1000."""
    n = mate_distance(cp)
    if n > 0:                                   # Trắng chiếu hết sau n nước
        return max(MATE_FLOOR, 1000 - (n - 1))
    if n < 0:                                   # Đen chiếu hết sau -n nước
        return min(1000 - MATE_FLOOR, 0 + (-n - 1))
    s = round(1000.0 * win_expectancy(cp))
    return min(NORMAL_HI, max(NORMAL_LO, s))


def score_to_cp(score):
    """Nghịch đảo gần đúng: điểm 0..1000 -> cp (chỉ cho dải không chiếu hết)."""
    s = min(NORMAL_HI, max(NORMAL_LO, score)) / 1000.0
    return SCALE * math.log(s / (1.0 - s)) - _CALIB


def explain(score, cp=None):
    """Diễn giải điểm số bằng tiếng Việt."""
    n = mate_distance(cp) if cp is not None else 0
    if n > 0:
        return "Trắng chiếu hết sau {} nước".format(n) if n > 1 else "Trắng CHIẾU HẾT ngay nước này"
    if n < 0:
        return "Đen chiếu hết sau {} nước".format(-n) if n < -1 else "Đen CHIẾU HẾT ngay nước này"
    d = score - 500
    a = abs(d)
    who = "Trắng" if d > 0 else "Đen"
    if a < 8:
        return "Cân bằng"
    if a < 30:
        return "{} nhỉnh hơn chút ít".format(who)
    if a < 80:
        return "{} hơi ưu thế".format(who)
    if a < 180:
        return "{} ưu thế rõ".format(who)
    if a < 330:
        return "{} ưu thế lớn".format(who)
    if a < 460:
        return "{} thắng thế".format(who)
    return "{} thắng gần như chắc chắn".format(who)


def bar(score, width=40):
    """Thanh trực quan cho điểm số."""
    filled = int(round(score / 1000.0 * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


calibrate(0.0)   # mặc định; main.py sẽ hiệu chỉnh lại theo hàm lượng giá thật

"""Hàm lượng giá thủ công (tapered eval kiểu PeSTO) - làm mốc so sánh
và sinh nhãn khởi động cho mạng nơ-ron."""

from chess_core import (
    WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    WP, BP,
    FILE_A, bits, popcount, KNIGHT_ATT, KING_ATT, bishop_attacks, rook_attacks,
)

MG_VAL = (82, 337, 365, 477, 1025, 0)
EG_VAL = (94, 281, 297, 512, 936, 0)
PHASE_W = (0, 1, 1, 2, 4, 0)
TOTAL_PHASE = 24

# Bảng vị trí ô, viết theo thứ tự a8..h1 (nhìn từ phía Trắng)
_MG = {}
_EG = {}
_MG[PAWN] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0]
_EG[PAWN] = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0]
_MG[KNIGHT] = [
   -167, -89, -34, -49,  61, -97, -15, -107,
    -73, -41,  72,  36,  23,  62,   7,  -17,
    -47,  60,  37,  65,  84, 129,  73,   44,
     -9,  17,  19,  53,  37,  69,  18,   22,
    -13,   4,  16,  13,  28,  19,  21,   -8,
    -23,  -9,  12,  10,  19,  17,  25,  -16,
    -29, -53, -12,  -3,  -1,  18, -14,  -19,
   -105, -21, -58, -33, -17, -28, -19,  -23]
_EG[KNIGHT] = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64]
_MG[BISHOP] = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21]
_EG[BISHOP] = [
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17]
_MG[ROOK] = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26]
_EG[ROOK] = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20]
_MG[QUEEN] = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50]
_EG[QUEEN] = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41]
_MG[KING] = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14]
_EG[KING] = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43]

# PST[piece_index][square]  (square theo quy ước a1=0); đã cộng sẵn giá trị quân
PST_MG = [[0] * 64 for _ in range(12)]
PST_EG = [[0] * 64 for _ in range(12)]
for _t in range(6):
    for _sq in range(64):
        _rank, _file = _sq >> 3, _sq & 7
        _idx_white = (7 - _rank) * 8 + _file          # bảng viết từ a8
        PST_MG[_t][_sq] = MG_VAL[_t] + _MG[_t][_idx_white]
        PST_EG[_t][_sq] = EG_VAL[_t] + _EG[_t][_idx_white]
        _idx_black = _rank * 8 + _file                 # lật dọc cho Đen
        PST_MG[_t + 6][_sq] = MG_VAL[_t] + _MG[_t][_idx_black]
        PST_EG[_t + 6][_sq] = EG_VAL[_t] + _EG[_t][_idx_black]

FILES = [FILE_A << f for f in range(8)]
# Mặt nạ "phía trước + hai cột kề" để phát hiện tốt thông
PASSED_MASK = [[0] * 64 for _ in range(2)]
ISOLATED_MASK = [0] * 64
for _sq in range(64):
    _f, _r = _sq & 7, _sq >> 3
    _adj = FILES[_f]
    if _f > 0:
        _adj |= FILES[_f - 1]
    if _f < 7:
        _adj |= FILES[_f + 1]
    _wm = _bm = 0
    for _rr in range(_r + 1, 8):
        _wm |= _adj & (0xFF << (_rr * 8))
    for _rr in range(0, _r):
        _bm |= _adj & (0xFF << (_rr * 8))
    PASSED_MASK[WHITE][_sq] = _wm
    PASSED_MASK[BLACK][_sq] = _bm
    ISOLATED_MASK[_sq] = _adj & ~FILES[_f]

PASSED_BONUS_MG = (0, 5, 10, 20, 35, 60, 100, 0)
PASSED_BONUS_EG = (0, 15, 25, 45, 75, 120, 180, 0)
MOBILITY_MG = {KNIGHT: 4, BISHOP: 5, ROOK: 2, QUEEN: 1}
MOBILITY_EG = {KNIGHT: 4, BISHOP: 5, ROOK: 4, QUEEN: 2}
KING_ATTACK_W = {KNIGHT: 20, BISHOP: 20, ROOK: 40, QUEEN: 80}
BISHOP_PAIR_MG, BISHOP_PAIR_EG = 30, 45
DOUBLED_MG, DOUBLED_EG = -10, -22
ISOLATED_MG, ISOLATED_EG = -14, -18
ROOK_OPEN_MG, ROOK_SEMI_MG = 26, 12
TEMPO = 12


def game_phase(pos):
    ph = 0
    for t in (KNIGHT, BISHOP, ROOK, QUEEN):
        ph += PHASE_W[t] * (popcount(pos.pieces[t]) + popcount(pos.pieces[t + 6]))
    return min(ph, TOTAL_PHASE)


def evaluate(pos):
    """Điểm centipawn theo GÓC NHÌN TRẮNG (dương = Trắng tốt hơn)."""
    mg = eg = 0
    occ = pos.all_bb
    wp, bp = pos.pieces[WP], pos.pieces[BP]

    for p in range(12):
        bb = pos.pieces[p]
        if not bb:
            continue
        sign = 1 if p < 6 else -1
        tbl_mg, tbl_eg = PST_MG[p], PST_EG[p]
        for sq in bits(bb):
            mg += sign * tbl_mg[sq]
            eg += sign * tbl_eg[sq]

    for color in (WHITE, BLACK):
        base = color * 6
        sign = 1 if color == WHITE else -1
        own = pos.color_bb[color]
        enemy_king = pos.king_sq(color ^ 1)
        king_zone = KING_ATT[enemy_king] | (1 << enemy_king) if enemy_king >= 0 else 0
        king_pressure = 0

        # cơ động + áp lực lên vua đối phương
        for t in (KNIGHT, BISHOP, ROOK, QUEEN):
            for sq in bits(pos.pieces[base + t]):
                if t == KNIGHT:
                    att = KNIGHT_ATT[sq]
                elif t == BISHOP:
                    att = bishop_attacks(sq, occ)
                elif t == ROOK:
                    att = rook_attacks(sq, occ)
                else:
                    att = bishop_attacks(sq, occ) | rook_attacks(sq, occ)
                free = att & ~own
                n = popcount(free)
                mg += sign * MOBILITY_MG[t] * (n - 4)
                eg += sign * MOBILITY_EG[t] * (n - 4)
                if free & king_zone:
                    king_pressure += KING_ATTACK_W[t]
        mg += sign * (king_pressure * king_pressure) // 400

        # cặp tượng
        if popcount(pos.pieces[base + BISHOP]) >= 2:
            mg += sign * BISHOP_PAIR_MG
            eg += sign * BISHOP_PAIR_EG

        # cấu trúc tốt
        pawns = pos.pieces[base + PAWN]
        foes = pos.pieces[(color ^ 1) * 6 + PAWN]
        for f in range(8):
            n = popcount(pawns & FILES[f])
            if n > 1:
                mg += sign * DOUBLED_MG * (n - 1)
                eg += sign * DOUBLED_EG * (n - 1)
        for sq in bits(pawns):
            if not (pawns & ISOLATED_MASK[sq]):
                mg += sign * ISOLATED_MG
                eg += sign * ISOLATED_EG
            if not (foes & PASSED_MASK[color][sq]):
                rel = (sq >> 3) if color == WHITE else 7 - (sq >> 3)
                mg += sign * PASSED_BONUS_MG[rel]
                eg += sign * PASSED_BONUS_EG[rel]

        # xe trên cột mở / nửa mở
        for sq in bits(pos.pieces[base + ROOK]):
            f = FILES[sq & 7]
            if not (pawns & f):
                mg += sign * (ROOK_OPEN_MG if not (foes & f) else ROOK_SEMI_MG)

    ph = game_phase(pos)
    num = mg * ph + eg * (TOTAL_PHASE - ph)
    # cắt về 0 (không phải làm tròn xuống) để lượng giá đối xứng tuyệt đối
    # theo màu: eval(thế cờ) == -eval(thế cờ lật màu)
    score = -((-num) // TOTAL_PHASE) if num < 0 else num // TOTAL_PHASE
    score += TEMPO if pos.side == WHITE else -TEMPO
    return score


def evaluate_stm(pos):
    """Điểm theo góc nhìn BÊN ĐANG ĐI (dùng trong negamax)."""
    e = evaluate(pos)
    return e if pos.side == WHITE else -e

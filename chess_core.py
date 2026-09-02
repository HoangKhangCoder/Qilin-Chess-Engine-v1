"""Lõi cờ vua thuần Python (bitboard) - không dùng thư viện cờ ngoài.

Quy ước ô: 0 = a1, 1 = b1, ..., 7 = h1, 8 = a2, ..., 63 = h8.
Chỉ số quân: color*6 + type, type: 0=P 1=N 2=B 3=R 4=Q 5=K
"""

import random

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK = range(12)

FULL = (1 << 64) - 1
FILE_A = 0x0101010101010101
FILE_H = FILE_A << 7
NOT_A = FULL ^ FILE_A
NOT_H = FULL ^ FILE_H
RANK_1 = 0xFF
RANK_3 = RANK_1 << 16
RANK_6 = RANK_1 << 40
RANK_8 = RANK_1 << 56

PIECE_CHARS = "PNBRQKpnbrqk"
CHAR_TO_PIECE = {c: i for i, c in enumerate(PIECE_CHARS)}

# ---------------------------------------------------------------- bit helpers

try:
    (0).bit_count()

    def popcount(b):
        return b.bit_count()
except AttributeError:                                       # Python < 3.10
    def popcount(b):
        return bin(b).count("1")


def lsb(b):
    return (b & -b).bit_length() - 1


def msb(b):
    return b.bit_length() - 1


def bits(b):
    """Duyệt từng ô đang bật."""
    while b:
        low = b & -b
        yield low.bit_length() - 1
        b ^= low


def sq_name(sq):
    return "abcdefgh"[sq & 7] + str((sq >> 3) + 1)


def name_sq(s):
    return (int(s[1]) - 1) * 8 + "abcdefgh".index(s[0])


# ------------------------------------------------------------ attack tables

KNIGHT_ATT = [0] * 64
KING_ATT = [0] * 64
PAWN_ATT = [[0] * 64, [0] * 64]

for _sq in range(64):
    _f, _r = _sq & 7, _sq >> 3
    _n = _k = 0
    for _df, _dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
        _nf, _nr = _f + _df, _r + _dr
        if 0 <= _nf < 8 and 0 <= _nr < 8:
            _n |= 1 << (_nr * 8 + _nf)
    for _df in (-1, 0, 1):
        for _dr in (-1, 0, 1):
            if _df == _dr == 0:
                continue
            _nf, _nr = _f + _df, _r + _dr
            if 0 <= _nf < 8 and 0 <= _nr < 8:
                _k |= 1 << (_nr * 8 + _nf)
    KNIGHT_ATT[_sq] = _n
    KING_ATT[_sq] = _k
    _w = _b = 0
    for _df in (-1, 1):
        _nf = _f + _df
        if 0 <= _nf < 8:
            if _r + 1 < 8:
                _w |= 1 << ((_r + 1) * 8 + _nf)
            if _r - 1 >= 0:
                _b |= 1 << ((_r - 1) * 8 + _nf)
    PAWN_ATT[WHITE][_sq] = _w
    PAWN_ATT[BLACK][_sq] = _b

# 8 hướng tia: 0=N 1=NE 2=E 3=SE 4=S 5=SW 6=W 7=NW
_DIRS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
IS_POS = (True, True, True, False, False, False, False, True)
RAYS = [[0] * 64 for _ in range(8)]
for _d, (_df, _dr) in enumerate(_DIRS):
    for _sq in range(64):
        _f, _r = _sq & 7, _sq >> 3
        _bb = 0
        while True:
            _f += _df
            _r += _dr
            if not (0 <= _f < 8 and 0 <= _r < 8):
                break
            _bb |= 1 << (_r * 8 + _f)
        RAYS[_d][_sq] = _bb

def ray_attacks(d, sq, occ):
    a = RAYS[d][sq]
    blockers = a & occ
    if blockers:
        b = lsb(blockers) if IS_POS[d] else msb(blockers)
        a ^= RAYS[d][b]
    return a


def rook_attacks(sq, occ):
    return (ray_attacks(0, sq, occ) | ray_attacks(2, sq, occ)
            | ray_attacks(4, sq, occ) | ray_attacks(6, sq, occ))


def bishop_attacks(sq, occ):
    return (ray_attacks(1, sq, occ) | ray_attacks(3, sq, occ)
            | ray_attacks(5, sq, occ) | ray_attacks(7, sq, occ))


# ------------------------------------------------------------------- zobrist

_rng = random.Random(20240815)
Z_PIECE = [[_rng.getrandbits(64) for _ in range(64)] for _ in range(12)]
Z_SIDE = _rng.getrandbits(64)
Z_CASTLE = [_rng.getrandbits(64) for _ in range(16)]
Z_EP = [_rng.getrandbits(64) for _ in range(8)]

# ---------------------------------------------------------------------- move
# from(6) | to(6)<<6 | promo(3)<<12 | flag(3)<<15
F_NORMAL, F_DOUBLE, F_EP, F_CASTLE, F_PROMO = 0, 1, 2, 3, 4


def mk_move(frm, to, promo=0, flag=F_NORMAL):
    return frm | (to << 6) | (promo << 12) | (flag << 15)


def move_str(m):
    s = sq_name(m & 63) + sq_name((m >> 6) & 63)
    if (m >> 15) & 7 == F_PROMO:
        s += "nbrq"[((m >> 12) & 7) - KNIGHT]
    return s


CASTLE_MASK = [15] * 64
CASTLE_MASK[0] = 15 & ~2      # a1 -> mất nhập thành dài Trắng
CASTLE_MASK[4] = 15 & ~3      # e1 -> mất cả hai bên Trắng
CASTLE_MASK[7] = 15 & ~1      # h1
CASTLE_MASK[56] = 15 & ~8     # a8
CASTLE_MASK[60] = 15 & ~12    # e8
CASTLE_MASK[63] = 15 & ~4     # h8

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class Position:
    __slots__ = ("pieces", "color_bb", "all_bb", "board", "side",
                 "castling", "ep", "halfmove", "fullmove", "key")

    def __init__(self, fen=START_FEN):
        self.set_fen(fen)

    # ------------------------------------------------------------------ FEN
    def set_fen(self, fen):
        parts = fen.split()
        rows = parts[0].split("/")
        self.pieces = [0] * 12
        self.board = [-1] * 64
        for i, row in enumerate(rows):
            rank = 7 - i
            file = 0
            for ch in row:
                if ch.isdigit():
                    file += int(ch)
                else:
                    p = CHAR_TO_PIECE[ch]
                    sq = rank * 8 + file
                    self.pieces[p] |= 1 << sq
                    self.board[sq] = p
                    file += 1
        self.side = WHITE if parts[1] == "w" else BLACK
        self.castling = 0
        if len(parts) > 2 and parts[2] != "-":
            for ch, bit in (("K", 1), ("Q", 2), ("k", 4), ("q", 8)):
                if ch in parts[2]:
                    self.castling |= bit
        self.ep = name_sq(parts[3]) if len(parts) > 3 and parts[3] != "-" else -1
        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove = int(parts[5]) if len(parts) > 5 else 1
        self._refresh()
        # Nhiều FEN ghi ô ep vô điều kiện sau mỗi nước tốt đi 2 ô. Bỏ ô ep không
        # bắt được thật, để cùng một thế cờ luôn cho cùng một khoá Zobrist dù
        # đến từ FEN hay từ chuỗi nước đi.
        if self.ep >= 0 and not self._has_legal_ep(self.ep):
            self.ep = -1
            self.key = self.compute_key()

    def _refresh(self):
        w = self.pieces[0] | self.pieces[1] | self.pieces[2] | self.pieces[3] \
            | self.pieces[4] | self.pieces[5]
        b = self.pieces[6] | self.pieces[7] | self.pieces[8] | self.pieces[9] \
            | self.pieces[10] | self.pieces[11]
        self.color_bb = [w, b]
        self.all_bb = w | b
        self.key = self.compute_key()

    def compute_key(self):
        k = 0
        for p in range(12):
            for sq in bits(self.pieces[p]):
                k ^= Z_PIECE[p][sq]
        if self.side == BLACK:
            k ^= Z_SIDE
        k ^= Z_CASTLE[self.castling]
        if self.ep >= 0:
            k ^= Z_EP[self.ep & 7]
        return k

    def fen(self):
        rows = []
        for rank in range(7, -1, -1):
            row, empty = "", 0
            for file in range(8):
                p = self.board[rank * 8 + file]
                if p < 0:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += PIECE_CHARS[p]
            if empty:
                row += str(empty)
            rows.append(row)
        cast = "".join(c for c, b in (("K", 1), ("Q", 2), ("k", 4), ("q", 8))
                       if self.castling & b) or "-"
        return "{} {} {} {} {} {}".format(
            "/".join(rows), "w" if self.side == WHITE else "b", cast,
            sq_name(self.ep) if self.ep >= 0 else "-", self.halfmove, self.fullmove)

    def copy(self):
        p = Position.__new__(Position)
        p.pieces = self.pieces[:]
        p.color_bb = self.color_bb[:]
        p.all_bb = self.all_bb
        p.board = self.board[:]
        p.side = self.side
        p.castling = self.castling
        p.ep = self.ep
        p.halfmove = self.halfmove
        p.fullmove = self.fullmove
        p.key = self.key
        return p

    # -------------------------------------------------------------- tấn công
    def attackers_to(self, sq, by):
        """Bitboard các quân màu `by` đang tấn công ô `sq`."""
        occ = self.all_bb
        base = by * 6
        res = PAWN_ATT[by ^ 1][sq] & self.pieces[base + PAWN]
        res |= KNIGHT_ATT[sq] & self.pieces[base + KNIGHT]
        res |= KING_ATT[sq] & self.pieces[base + KING]
        ba = bishop_attacks(sq, occ)
        res |= ba & (self.pieces[base + BISHOP] | self.pieces[base + QUEEN])
        ra = rook_attacks(sq, occ)
        res |= ra & (self.pieces[base + ROOK] | self.pieces[base + QUEEN])
        return res

    def is_attacked(self, sq, by):
        occ = self.all_bb
        base = by * 6
        if PAWN_ATT[by ^ 1][sq] & self.pieces[base + PAWN]:
            return True
        if KNIGHT_ATT[sq] & self.pieces[base + KNIGHT]:
            return True
        if KING_ATT[sq] & self.pieces[base + KING]:
            return True
        if bishop_attacks(sq, occ) & (self.pieces[base + BISHOP] | self.pieces[base + QUEEN]):
            return True
        if rook_attacks(sq, occ) & (self.pieces[base + ROOK] | self.pieces[base + QUEEN]):
            return True
        return False

    def king_sq(self, color):
        k = self.pieces[color * 6 + KING]
        return (k & -k).bit_length() - 1 if k else -1

    def in_check(self, color=None):
        if color is None:
            color = self.side
        ks = self.king_sq(color)
        return ks >= 0 and self.is_attacked(ks, color ^ 1)

    # ------------------------------------------------------- sinh nước đi
    def generate(self, captures_only=False):
        moves = []
        add = moves.append
        side = self.side
        base = side * 6
        us = self.color_bb[side]
        them = self.color_bb[side ^ 1]
        occ = self.all_bb
        empty = FULL ^ occ
        targets = them if captures_only else (FULL ^ us)

        pawns = self.pieces[base + PAWN]
        if side == WHITE:
            promo_rank, push, dbl_rank = RANK_8, 8, RANK_3
            capl_sh, capr_sh = 7, 9
        else:
            promo_rank, push, dbl_rank = RANK_1, -8, RANK_6
            capl_sh, capr_sh = -9, -7

        # --- tốt: ăn quân
        if side == WHITE:
            capl = ((pawns & NOT_A) << 7) & them
            capr = ((pawns & NOT_H) << 9) & them
        else:
            capl = ((pawns & NOT_A) >> 9) & them
            capr = ((pawns & NOT_H) >> 7) & them
        for tobb, sh in ((capl, capl_sh), (capr, capr_sh)):
            for to in bits(tobb):
                frm = to - sh
                if (1 << to) & promo_rank:
                    for pr in (QUEEN, ROOK, BISHOP, KNIGHT):
                        add(mk_move(frm, to, pr, F_PROMO))
                else:
                    add(mk_move(frm, to))

        # --- tốt: bắt tốt qua đường
        if self.ep >= 0:
            for frm in bits(PAWN_ATT[side ^ 1][self.ep] & pawns):
                add(mk_move(frm, self.ep, 0, F_EP))

        # --- tốt: đi thẳng
        one = ((pawns << 8) if side == WHITE else (pawns >> 8)) & empty & FULL
        promos = one & promo_rank
        for to in bits(promos):
            frm = to - push
            for pr in (QUEEN, ROOK, BISHOP, KNIGHT):
                add(mk_move(frm, to, pr, F_PROMO))
        if not captures_only:
            for to in bits(one & ~promo_rank):
                add(mk_move(to - push, to))
            two = ((one & dbl_rank) << 8 if side == WHITE else (one & dbl_rank) >> 8) & empty
            for to in bits(two):
                add(mk_move(to - 2 * push, to, 0, F_DOUBLE))

        # --- mã
        for frm in bits(self.pieces[base + KNIGHT]):
            for to in bits(KNIGHT_ATT[frm] & targets):
                add(mk_move(frm, to))
        # --- tượng / hậu (chéo)
        for frm in bits(self.pieces[base + BISHOP] | self.pieces[base + QUEEN]):
            for to in bits(bishop_attacks(frm, occ) & targets):
                add(mk_move(frm, to))
        # --- xe / hậu (thẳng)
        for frm in bits(self.pieces[base + ROOK] | self.pieces[base + QUEEN]):
            for to in bits(rook_attacks(frm, occ) & targets):
                add(mk_move(frm, to))
        # --- vua
        ksq = self.king_sq(side)
        if ksq >= 0:
            for to in bits(KING_ATT[ksq] & targets):
                add(mk_move(ksq, to))
            # --- nhập thành
            if not captures_only and self.castling:
                enemy = side ^ 1
                if side == WHITE:
                    if (self.castling & 1) and not (occ & 0x60) \
                            and not self.is_attacked(4, enemy) \
                            and not self.is_attacked(5, enemy) and not self.is_attacked(6, enemy):
                        add(mk_move(4, 6, 0, F_CASTLE))
                    if (self.castling & 2) and not (occ & 0x0E) \
                            and not self.is_attacked(4, enemy) \
                            and not self.is_attacked(3, enemy) and not self.is_attacked(2, enemy):
                        add(mk_move(4, 2, 0, F_CASTLE))
                else:
                    if (self.castling & 4) and not (occ & (0x60 << 56)) \
                            and not self.is_attacked(60, enemy) \
                            and not self.is_attacked(61, enemy) and not self.is_attacked(62, enemy):
                        add(mk_move(60, 62, 0, F_CASTLE))
                    if (self.castling & 8) and not (occ & (0x0E << 56)) \
                            and not self.is_attacked(60, enemy) \
                            and not self.is_attacked(59, enemy) and not self.is_attacked(58, enemy):
                        add(mk_move(60, 58, 0, F_CASTLE))
        return moves

    def legal_moves(self):
        out = []
        for m in self.generate():
            u = self.make_move(m)
            if not self.is_attacked(self.king_sq(self.side ^ 1), self.side):
                out.append(m)
            self.unmake_move(m, u)
        return out

    def has_legal_move(self):
        for m in self.generate():
            u = self.make_move(m)
            ok = not self.is_attacked(self.king_sq(self.side ^ 1), self.side)
            self.unmake_move(m, u)
            if ok:
                return True
        return False

    # -------------------------------------------------------- make / unmake
    def _has_legal_ep(self, ep_sq):
        """Bên ĐANG ĐI có nước bắt tốt qua đường hợp lệ tại ep_sq không?

        Chỉ khi có thì ô ep mới được ghi vào thế cờ. Nếu ghi vô điều kiện, ô ep
        "ma" sẽ lọt vào khoá Zobrist và khiến hai thế cờ giống hệt nhau bị coi
        là khác - hỏng phát hiện lặp 3 lần. Đây cũng là quy ước của python-chess
        và của luật FIDE (quyền bắt qua đường chỉ tồn tại khi bắt được thật).
        """
        side = self.side
        base = side * 6
        srcs = PAWN_ATT[side ^ 1][ep_sq] & self.pieces[base + PAWN]
        if not srcs:
            return False
        ksq = self.king_sq(side)
        if ksq < 0:
            return False
        them = side ^ 1
        tb = them * 6
        cap_sq = ep_sq - 8 if side == WHITE else ep_sq + 8
        cap_bb = 1 << cap_sq
        ep_bb = 1 << ep_sq
        e_pawns = self.pieces[tb + PAWN] ^ cap_bb        # tốt bị bắt biến mất
        e_diag = self.pieces[tb + BISHOP] | self.pieces[tb + QUEEN]
        e_line = self.pieces[tb + ROOK] | self.pieces[tb + QUEEN]
        e_knights = self.pieces[tb + KNIGHT]
        e_king = self.pieces[tb + KING]
        for frm in bits(srcs):
            occ = (self.all_bb ^ (1 << frm) ^ cap_bb) | ep_bb
            if PAWN_ATT[side][ksq] & e_pawns:
                continue
            if KNIGHT_ATT[ksq] & e_knights:
                continue
            if KING_ATT[ksq] & e_king:
                continue
            if bishop_attacks(ksq, occ) & e_diag:
                continue
            if rook_attacks(ksq, occ) & e_line:
                continue
            return True
        return False

    def make_move(self, m):
        frm = m & 63
        to = (m >> 6) & 63
        flag = (m >> 15) & 7
        side = self.side
        enemy = side ^ 1
        piece = self.board[frm]
        captured = self.board[to]
        undo = (captured, self.castling, self.ep, self.halfmove, self.key)
        key = self.key

        if self.ep >= 0:
            key ^= Z_EP[self.ep & 7]
        key ^= Z_CASTLE[self.castling]

        # nhấc quân khỏi ô xuất phát
        fb = 1 << frm
        tb = 1 << to
        self.pieces[piece] ^= fb
        self.color_bb[side] ^= fb
        self.board[frm] = -1
        key ^= Z_PIECE[piece][frm]

        if captured >= 0:
            self.pieces[captured] ^= tb
            self.color_bb[enemy] ^= tb
            key ^= Z_PIECE[captured][to]
        elif flag == F_EP:
            csq = to - 8 if side == WHITE else to + 8
            cp = enemy * 6 + PAWN
            self.pieces[cp] ^= 1 << csq
            self.color_bb[enemy] ^= 1 << csq
            self.board[csq] = -1
            key ^= Z_PIECE[cp][csq]

        moved_type = piece % 6          # LƯU trước khi phong cấp đổi loại quân
        if flag == F_PROMO:
            piece = side * 6 + ((m >> 12) & 7)
        self.pieces[piece] |= tb
        self.color_bb[side] |= tb
        self.board[to] = piece
        key ^= Z_PIECE[piece][to]

        if flag == F_CASTLE:
            if to == 6:
                rf, rt = 7, 5
            elif to == 2:
                rf, rt = 0, 3
            elif to == 62:
                rf, rt = 63, 61
            else:
                rf, rt = 56, 59
            rp = side * 6 + ROOK
            self.pieces[rp] ^= (1 << rf) | (1 << rt)
            self.color_bb[side] ^= (1 << rf) | (1 << rt)
            self.board[rf] = -1
            self.board[rt] = rp
            key ^= Z_PIECE[rp][rf] ^ Z_PIECE[rp][rt]

        self.castling &= CASTLE_MASK[frm] & CASTLE_MASK[to]
        key ^= Z_CASTLE[self.castling]

        # Phong cấp cũng là nước ĐI TỐT nên phải reset đồng hồ 50 nước;
        # dùng piece ở đây sẽ sai vì nó đã thành hậu/xe/tượng/mã.
        if moved_type == PAWN or captured >= 0 or flag == F_EP:
            self.halfmove = 0
        else:
            self.halfmove += 1
        if side == BLACK:
            self.fullmove += 1
        self.side = enemy
        self.all_bb = self.color_bb[0] | self.color_bb[1]

        # đặt ô bắt qua đường SAU khi đổi lượt: cần biết bên kia có bắt được không
        self.ep = -1
        if flag == F_DOUBLE:
            ep_sq = (frm + to) >> 1
            if self._has_legal_ep(ep_sq):
                self.ep = ep_sq
                key ^= Z_EP[ep_sq & 7]
        self.key = key ^ Z_SIDE
        return undo

    def unmake_move(self, m, undo):
        captured, castling, ep, halfmove, key = undo
        frm = m & 63
        to = (m >> 6) & 63
        flag = (m >> 15) & 7
        side = self.side ^ 1
        enemy = self.side
        fb = 1 << frm
        tb = 1 << to

        piece = self.board[to]
        self.pieces[piece] ^= tb
        self.color_bb[side] ^= tb
        if flag == F_PROMO:
            piece = side * 6 + PAWN
        self.pieces[piece] |= fb
        self.color_bb[side] |= fb
        self.board[frm] = piece
        self.board[to] = -1

        if captured >= 0:
            self.pieces[captured] |= tb
            self.color_bb[enemy] |= tb
            self.board[to] = captured
        elif flag == F_EP:
            csq = to - 8 if side == WHITE else to + 8
            cp = enemy * 6 + PAWN
            self.pieces[cp] |= 1 << csq
            self.color_bb[enemy] |= 1 << csq
            self.board[csq] = cp

        if flag == F_CASTLE:
            if to == 6:
                rf, rt = 7, 5
            elif to == 2:
                rf, rt = 0, 3
            elif to == 62:
                rf, rt = 63, 61
            else:
                rf, rt = 56, 59
            rp = side * 6 + ROOK
            self.pieces[rp] ^= (1 << rf) | (1 << rt)
            self.color_bb[side] ^= (1 << rf) | (1 << rt)
            self.board[rt] = -1
            self.board[rf] = rp

        self.castling = castling
        self.ep = ep
        self.halfmove = halfmove
        self.key = key
        if side == BLACK:
            self.fullmove -= 1
        self.side = side
        self.all_bb = self.color_bb[0] | self.color_bb[1]

    def make_null(self):
        undo = (self.ep, self.key)
        if self.ep >= 0:
            self.key ^= Z_EP[self.ep & 7]
        self.ep = -1
        self.side ^= 1
        self.key ^= Z_SIDE
        return undo

    def unmake_null(self, undo):
        self.ep, self.key = undo
        self.side ^= 1

    # ------------------------------------------------------------ trạng thái
    def is_checkmate(self):
        return self.in_check() and not self.has_legal_move()

    def is_stalemate(self):
        return not self.in_check() and not self.has_legal_move()

    def has_non_pawn_material(self, color):
        base = color * 6
        return bool(self.pieces[base + KNIGHT] | self.pieces[base + BISHOP]
                    | self.pieces[base + ROOK] | self.pieces[base + QUEEN])

    def is_insufficient_material(self):
        if self.pieces[WP] or self.pieces[BP] or self.pieces[WR] or self.pieces[BR] \
                or self.pieces[WQ] or self.pieces[BQ]:
            return False
        minors = popcount(self.pieces[WN] | self.pieces[WB] | self.pieces[BN] | self.pieces[BB])
        return minors <= 1

    def __str__(self):
        out = []
        for rank in range(7, -1, -1):
            row = [str(rank + 1), " "]
            for file in range(8):
                p = self.board[rank * 8 + file]
                row.append("." if p < 0 else PIECE_CHARS[p])
                row.append(" ")
            out.append("".join(row))
        out.append("  a b c d e f g h")
        out.append("  " + ("Trắng" if self.side == WHITE else "Đen") + " đi")
        return "\n".join(out)


# ---------------------------------------------------------------------- perft

def perft(pos, depth):
    if depth == 0:
        return 1
    total = 0
    for m in pos.generate():
        u = pos.make_move(m)
        if not pos.is_attacked(pos.king_sq(pos.side ^ 1), pos.side):
            total += perft(pos, depth - 1) if depth > 1 else 1
        pos.unmake_move(m, u)
    return total


def parse_uci(pos, s):
    """Chuyển 'e2e4', 'e7e8q' thành nước đi hợp lệ, hoặc None."""
    for m in pos.legal_moves():
        if move_str(m) == s:
            return m
    return None


# ------------------------------------------------------------- luật hoà cờ

class Game:
    """Position + lịch sử khoá Zobrist, đủ để áp dụng các luật hoà cần lịch sử.

    Các luật hoà được xử lý:
      - hết nước đi mà không bị chiếu  -> pat (stalemate)
      - luật 50 nước (100 nửa nước không bắt quân và không đi tốt)
      - lặp lại thế cờ 3 lần (kể cả thủ hoà bằng chiếu liên tục)
      - không đủ lực chiếu hết (vua trơ, vua + 1 mã/tượng)
    Đồng hồ nửa nước reset khi ĐI TỐT (bao gồm phong cấp và bắt tốt qua đường)
    hoặc khi BẮT QUÂN - đúng theo luật FIDE 9.3.
    """

    def __init__(self, fen=START_FEN):
        self.pos = Position(fen)
        self.keys = [self.pos.key]
        self.stack = []

    def push(self, move):
        self.stack.append((move, self.pos.make_move(move)))
        self.keys.append(self.pos.key)
        return self

    def push_uci(self, s):
        m = parse_uci(self.pos, s)
        if m is None:
            raise ValueError("Nước không hợp lệ: {}".format(s))
        return self.push(m)

    def pop(self):
        if not self.stack:
            return None
        m, u = self.stack.pop()
        self.pos.unmake_move(m, u)
        self.keys.pop()
        return m

    def repetitions(self):
        """Số lần thế cờ hiện tại đã xuất hiện (tính cả lần này)."""
        k = self.pos.key
        n = 0
        # cùng bên đi => cách nhau 2 nửa nước; xa nhất là mốc reset đồng hồ
        span = min(self.pos.halfmove, len(self.keys) - 1)
        for i in range(len(self.keys) - 1, len(self.keys) - 2 - span, -2):
            if i >= 0 and self.keys[i] == k:
                n += 1
        return n

    def draw_reason(self):
        """Lý do hoà (chuỗi tiếng Việt) hoặc None nếu ván chưa hoà."""
        pos = self.pos
        if not pos.has_legal_move():
            return None if pos.in_check() else "hết nước đi mà không bị chiếu - PAT"
        if pos.is_insufficient_material():
            return "không bên nào đủ lực chiếu hết"
        if pos.halfmove >= 100:
            return "luật 50 nước ({} nửa nước không bắt quân, không đi tốt)".format(pos.halfmove)
        r = self.repetitions()
        if r >= 3:
            return "lặp lại thế cờ {} lần".format(r)
        return None

    def outcome(self):
        """('1-0'|'0-1'|'1/2-1/2'|None, mô tả)."""
        pos = self.pos
        if not pos.has_legal_move():
            if pos.in_check():
                w = "Đen" if pos.side == WHITE else "Trắng"
                return ("0-1" if pos.side == WHITE else "1-0",
                        "{} thắng bằng chiếu hết".format(w))
            return "1/2-1/2", "hoà - hết nước đi mà không bị chiếu (pat)"
        d = self.draw_reason()
        if d:
            return "1/2-1/2", "hoà - " + d
        return None, "ván đang tiếp diễn"

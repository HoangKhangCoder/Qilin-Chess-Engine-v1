"""Tìm kiếm alpha-beta (negamax) với hàm lượng giá cắm ngoài.

Đây là phần "đánh giá sâu": hàm lượng giá chỉ nhìn thế cờ tĩnh, còn tìm kiếm
mới cho biết điều gì thật sự xảy ra sau vài nước ép buộc. Điểm trả về luôn
quy về GÓC NHÌN TRẮNG (dương = Trắng hơn).
"""

import time

from chess_core import (
    WHITE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, F_PROMO, F_EP, move_str,
    KNIGHT_ATT, KING_ATT, PAWN_ATT, bishop_attacks, rook_attacks,
)
from evaluate import evaluate_stm
from scoring import MATE_VALUE, MATE_BOUND

PIECE_VAL = (100, 320, 330, 500, 900, 20000)
MAX_PLY = 128
QS_MAX_DEPTH = 24      # trần độ sâu quiescence
QS_CHECK_DEPTH = 6     # chỉ nối tiếp chuỗi chiếu trong ngần này tầng

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2



# Giá trị quân dùng riêng cho SEE - phải là thang đơn giản, nhất quán, vì SEE
# chỉ cộng trừ dọc chuỗi ăn quân chứ không hiểu gì về thế cờ.
SEE_VAL = (100, 320, 330, 500, 900, 20000)


def see(pos, move):
    """Static Exchange Evaluation: cộng dồn chuỗi ăn qua lại trên ô đích.

    Trả về số centipawn LỜI (dương) hay LỖ (âm) nếu thực hiện nước ăn quân này
    rồi hai bên thay nhau ăn lại bằng quân rẻ nhất cho tới khi không ai muốn ăn.

    Đây là phép tính TĨNH - nó không biết gì về chiếu, ghim, hay đòn phối hợp.
    Vì vậy nó chỉ được dùng để SẮP XẾP thứ tự nước đi ở mọi nơi, và chỉ được
    phép CẮT BỎ hẳn nước đi trong quiescence khi không bị chiếu (xem quiesce).
    """
    frm = move & 63
    to = (move >> 6) & 63
    flag = (move >> 15) & 7
    board = pos.board

    if flag == F_EP:
        captured_val = SEE_VAL[PAWN]
    else:
        victim = board[to]
        if victim < 0:
            return 0                      # không phải nước ăn quân
        captured_val = SEE_VAL[victim % 6]

    attacker = board[frm]
    if attacker < 0:
        return 0
    att_type = attacker % 6

    # gain[i] = điểm ròng nếu chuỗi ăn dừng lại sau lượt thứ i
    gain = [captured_val]
    occ = pos.all_bb ^ (1 << frm)
    if flag == F_EP:
        occ ^= 1 << (to - 8 if pos.side == WHITE else to + 8)
    occ |= 1 << to

    side = pos.side ^ 1
    on_square = att_type          # quân đang đứng trên ô đích, sẽ bị ăn tiếp

    # Quân trượt có thể lộ ra sau khi quân trước rời đi, nên phải tính lại
    # attackers theo `occ` hiện tại ở mỗi lượt thay vì tính một lần.
    for _ in range(31):
        att = _attackers_to(pos, to, side, occ) & occ
        if not att:
            break
        bb, t = _least_valuable(pos, att, side)
        if not bb:
            break
        gain.append(SEE_VAL[on_square] - gain[-1])
        occ ^= bb
        on_square = t
        side ^= 1

    # lùi ngược: mỗi bên chỉ ăn tiếp nếu việc đó có lợi cho mình
    for i in range(len(gain) - 2, -1, -1):
        gain[i] = -max(-gain[i], gain[i + 1])
    return gain[0]


def _attackers_to(pos, sq, by, occ):
    """Như Position.attackers_to nhưng theo occupancy TÙY Ý (đang mô phỏng)."""
    base = by * 6
    res = PAWN_ATT[by ^ 1][sq] & pos.pieces[base + PAWN]
    res |= KNIGHT_ATT[sq] & pos.pieces[base + KNIGHT]
    res |= KING_ATT[sq] & pos.pieces[base + KING]
    res |= bishop_attacks(sq, occ) & (pos.pieces[base + BISHOP] | pos.pieces[base + QUEEN])
    res |= rook_attacks(sq, occ) & (pos.pieces[base + ROOK] | pos.pieces[base + QUEEN])
    return res


def _least_valuable(pos, attackers, color):
    """Quân RẺ NHẤT của `color` - luôn ăn bằng quân rẻ trước."""
    base = color * 6
    for t in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
        bb = attackers & pos.pieces[base + t]
        if bb:
            return (bb & -bb), t
    return 0, -1


class SearchAbort(Exception):
    pass


class Searcher:
    def __init__(self, eval_fn=None, tt_bits=20):
        self.eval_fn = eval_fn or evaluate_stm
        self.tt_mask = (1 << tt_bits) - 1
        self.tt = {}
        self.nodes = 0
        self.deadline = None
        self.max_nodes = None
        self.killers = [[0, 0] for _ in range(MAX_PLY)]
        self.history = [[0] * 64 for _ in range(12)]
        self.path_keys = []
        self.seldepth = 0

    # ---------------------------------------------------------------- tiện ích
    def _check_time(self):
        if self.deadline is not None and time.time() > self.deadline:
            raise SearchAbort()
        if self.max_nodes is not None and self.nodes > self.max_nodes:
            raise SearchAbort()

    def _is_repetition(self, pos):
        """path_keys[-1] chính là thế cờ hiện tại, nên phải dò lùi từ -3 và
        bước 2 (cùng bên đi). Chỉ cần lặp lại 1 lần là coi như hoà."""
        k = pos.key
        for i in range(len(self.path_keys) - 3, -1, -2):
            if self.path_keys[i] == k:
                return True
        return False

    def _order(self, pos, moves, ply, tt_move):
        board = pos.board
        killers = self.killers[ply] if ply < MAX_PLY else (0, 0)
        hist = self.history
        scored = []
        for m in moves:
            if m == tt_move:
                scored.append((1 << 30, m))
                continue
            to = (m >> 6) & 63
            frm = m & 63
            victim = board[to]
            flag = (m >> 15) & 7
            if victim >= 0 or flag == F_EP:
                # MVV-LVA xếp "ăn quân to bằng quân nhỏ" lên đầu, nhưng không
                # biết ô đó có được bảo vệ hay không. SEE biết. Nước ăn LỖ bị
                # đẩy xuống DƯỚI cả nước im lặng (chứ không bị loại - đòn hy
                # sinh vẫn phải được thử, chỉ là thử sau).
                mvv = PIECE_VAL[victim % 6] if victim >= 0 else PIECE_VAL[PAWN]
                lva = PIECE_VAL[board[frm] % 6]
                # SEE trong Python thuần đắt (đo được: bật vô điều kiện làm
                # CHẬM 15% dù giảm 8% số nút). Ăn quân to bằng quân nhỏ thì
                # chắc chắn không lỗ - khỏi cần tính. Chỉ tính khi quân ăn
                # đắt hơn quân bị ăn, tức là mới CÓ THỂ lỗ.
                if lva <= mvv or see(pos, m) >= 0:
                    s = 1_000_000 + mvv * 16 - lva
                else:
                    s = -1_000_000 + mvv * 16 - lva
            elif flag == F_PROMO:
                s = 900_000 + PIECE_VAL[(m >> 12) & 7]
            elif m == killers[0]:
                s = 800_000
            elif m == killers[1]:
                s = 790_000
            else:
                s = hist[board[frm]][to]
            scored.append((s, m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored]

    # ------------------------------------------------------------- quiescence
    def quiesce(self, pos, alpha, beta, ply, qdepth=0):
        self.nodes += 1
        if self.nodes & 2047 == 0:
            self._check_time()
        if ply > self.seldepth:
            self.seldepth = ply

        in_check = pos.in_check()
        # chặn nổ đệ quy: chuỗi chiếu liên tiếp có thể kéo dài vô hạn
        if ply >= MAX_PLY - 2 or qdepth >= QS_MAX_DEPTH or (in_check and qdepth >= QS_CHECK_DEPTH):
            return self.eval_fn(pos)

        if not in_check:
            stand = self.eval_fn(pos)
            if stand >= beta:
                return stand
            if stand > alpha:
                alpha = stand
            best = stand
        else:
            best = -MATE_VALUE + ply

        moves = pos.generate(captures_only=not in_check)
        moves = self._order(pos, moves, ply, 0)
        legal = 0
        for m in moves:
            if not in_check:
                # CHỈ cắt tỉa khi KHÔNG bị chiếu. Đòn hy sinh thiên tài luôn
                # đi kèm chuỗi chiếu ép buộc, và mọi nước trong chuỗi đó rơi
                # vào nhánh in_check bên dưới - nên không bao giờ bị SEE loại.
                # Trong search chính (negamax) SEE cũng không cắt gì cả, chỉ
                # dùng để sắp xếp, nên nước hy sinh vẫn được tìm kiếm đầy đủ.
                _v = pos.board[(m >> 6) & 63]
                if ((m >> 15) & 7 != F_PROMO and _v >= 0
                        and PIECE_VAL[pos.board[m & 63] % 6] > PIECE_VAL[_v % 6]
                        and see(pos, m) < 0):
                    continue

                # delta pruning: bỏ qua nước ăn quân không thể kéo lại thế cờ
                victim = pos.board[(m >> 6) & 63]
                gain = PIECE_VAL[victim % 6] if victim >= 0 else PIECE_VAL[PAWN]
                if (m >> 15) & 7 == F_PROMO:
                    gain += PIECE_VAL[QUEEN]
                if best + gain + 200 < alpha:
                    continue
            u = pos.make_move(m)
            if pos.is_attacked(pos.king_sq(pos.side ^ 1), pos.side):
                pos.unmake_move(m, u)
                continue
            legal += 1
            score = -self.quiesce(pos, -beta, -alpha, ply + 1, qdepth + 1)
            pos.unmake_move(m, u)
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        if in_check and legal == 0:
            return -MATE_VALUE + ply
        return best

    # ---------------------------------------------------------------- negamax
    def negamax(self, pos, depth, alpha, beta, ply, allow_null=True):
        self.nodes += 1
        if self.nodes & 2047 == 0:
            self._check_time()

        if ply > 0:
            if pos.halfmove >= 100 or pos.is_insufficient_material() or self._is_repetition(pos):
                return 0
            # mate distance pruning
            alpha = max(alpha, -MATE_VALUE + ply)
            beta = min(beta, MATE_VALUE - ply - 1)
            if alpha >= beta:
                return alpha

        if ply >= MAX_PLY - 4:
            return self.quiesce(pos, alpha, beta, ply)
        in_check = pos.in_check()
        if in_check and ply < MAX_PLY - 16:
            depth += 1                      # kéo dài khi bị chiếu
        if depth <= 0:
            return self.quiesce(pos, alpha, beta, ply)

        alpha0 = alpha
        key = pos.key
        slot = self.tt.get(key & self.tt_mask)
        tt_move = 0
        if slot is not None and slot[0] == key:
            _, tt_depth, tt_score, tt_flag, tt_move = slot
            if tt_depth >= depth and ply > 0:
                if tt_score > MATE_BOUND:
                    tt_score -= ply
                elif tt_score < -MATE_BOUND:
                    tt_score += ply
                if tt_flag == TT_EXACT:
                    return tt_score
                if tt_flag == TT_LOWER and tt_score >= beta:
                    return tt_score
                if tt_flag == TT_UPPER and tt_score <= alpha:
                    return tt_score

        # null-move pruning
        if (allow_null and not in_check and depth >= 3 and ply > 0
                and pos.has_non_pawn_material(pos.side) and beta < MATE_BOUND):
            r = 2 + depth // 6
            nu = pos.make_null()
            self.path_keys.append(pos.key)
            score = -self.negamax(pos, depth - 1 - r, -beta, -beta + 1, ply + 1, False)
            self.path_keys.pop()
            pos.unmake_null(nu)
            if score >= beta:
                return beta if score > MATE_BOUND else score

        moves = self._order(pos, pos.generate(), ply, tt_move)
        best = -MATE_VALUE - 1
        best_move = 0
        legal = 0

        for i, m in enumerate(moves):
            u = pos.make_move(m)
            if pos.is_attacked(pos.king_sq(pos.side ^ 1), pos.side):
                pos.unmake_move(m, u)
                continue
            legal += 1
            is_quiet = u[0] < 0 and (m >> 15) & 7 not in (F_PROMO, F_EP)
            self.path_keys.append(pos.key)

            if legal == 1:
                score = -self.negamax(pos, depth - 1, -beta, -alpha, ply + 1)
            else:
                # late move reduction
                red = 0
                if depth >= 3 and is_quiet and i >= 4 and not in_check:
                    red = 1 + (i >= 10) + (depth >= 6)
                    red = min(red, depth - 2)
                score = -self.negamax(pos, depth - 1 - red, -alpha - 1, -alpha, ply + 1)
                if score > alpha and red:
                    score = -self.negamax(pos, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -self.negamax(pos, depth - 1, -beta, -alpha, ply + 1)

            self.path_keys.pop()
            pos.unmake_move(m, u)

            if score > best:
                best = score
                best_move = m
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if is_quiet:
                            k = self.killers[ply]
                            if k[0] != m:
                                k[1] = k[0]
                                k[0] = m
                            self.history[pos.board[m & 63]][(m >> 6) & 63] += depth * depth
                        break

        if legal == 0:
            return -MATE_VALUE + ply if in_check else 0

        store = best
        if store > MATE_BOUND:
            store += ply
        elif store < -MATE_BOUND:
            store -= ply
        flag = TT_EXACT if alpha0 < best < beta else (TT_LOWER if best >= beta else TT_UPPER)
        self.tt[key & self.tt_mask] = (key, depth, store, flag, best_move)
        return best

    # ------------------------------------------------------------------- API
    def search(self, pos, depth=6, time_limit=None, max_nodes=None, history_keys=None,
               on_iteration=None):
        """Trả về dict: score (cp, góc nhìn Trắng), move, pv, depth, nodes."""
        self.nodes = 0
        self.seldepth = 0
        self.killers = [[0, 0] for _ in range(MAX_PLY)]
        self.history = [[0] * 64 for _ in range(12)]
        self.path_keys = list(history_keys or [])
        self.deadline = (time.time() + time_limit) if time_limit else None
        self.max_nodes = max_nodes

        work = pos.copy()
        legal = work.legal_moves()
        if not legal:
            cp = -MATE_VALUE if work.in_check() else 0
            return {"score": cp if work.side == WHITE else -cp, "move": None, "pv": [],
                    "depth": 0, "nodes": 0, "mate": work.in_check(), "legal": 0}

        best_move, best_cp, reached = legal[0], 0, 0
        alpha, beta = -MATE_VALUE, MATE_VALUE
        for d in range(1, depth + 1):
            try:
                if d >= 4:
                    window = 40
                    while True:
                        s = self.negamax(work, d, best_cp - window, best_cp + window, 0)
                        if best_cp - window < s < best_cp + window:
                            break
                        window *= 4
                        if window > 2000:
                            s = self.negamax(work, d, -MATE_VALUE, MATE_VALUE, 0)
                            break
                else:
                    s = self.negamax(work, d, alpha, beta, 0)
            except SearchAbort:
                # Ngoại lệ bay xuyên qua các khung negamax nên unmake_move bị bỏ
                # qua: `work` vẫn giữ nguyên chuỗi nước đang thử dở, thậm chí
                # đang tới lượt bên kia. Không dựng lại thì extract_pv bên dưới
                # sẽ đọc một thế cờ KHÁC và trả về nước bất hợp lệ ở thế cờ gốc.
                work = pos.copy()
                self.path_keys = list(history_keys or [])
                break
            slot = self.tt.get(work.key & self.tt_mask)
            if slot and slot[0] == work.key and slot[4]:
                best_move = slot[4]
            best_cp, reached = s, d
            if on_iteration:
                on_iteration(d, s, best_move, self.nodes)
            if abs(s) > MATE_BOUND:      # đã tìm ra chiếu hết ép buộc
                break

        pv = self.extract_pv(work, reached + 4)
        if pv:
            best_move = pv[0]
        # Lưới an toàn cuối: search KHÔNG BAO GIỜ được trả về nước bất hợp lệ.
        # match.py và play_stockfish.py đánh thẳng nước này lên bàn cờ.
        if best_move not in legal:
            best_move = legal[0]
            pv = []
        white_cp = best_cp if pos.side == WHITE else -best_cp
        return {"score": white_cp, "move": best_move, "pv": pv, "depth": reached,
                "nodes": self.nodes, "seldepth": self.seldepth, "legal": len(legal)}

    def extract_pv(self, pos, max_len=16):
        work = pos.copy()
        pv, seen, undo = [], set(), []
        for _ in range(max_len):
            slot = self.tt.get(work.key & self.tt_mask)
            if not slot or slot[0] != work.key or not slot[4]:
                break
            m = slot[4]
            if m not in work.legal_moves() or work.key in seen:
                break
            seen.add(work.key)
            pv.append(m)
            undo.append((m, work.make_move(m)))
        for m, u in reversed(undo):
            work.unmake_move(m, u)
        return pv


def pv_string(pos, pv):
    return " ".join(move_str(m) for m in pv)


def find_mate_in_one(pos):
    """Kiểm tra trực tiếp: bên đi có nước chiếu hết ngay không?"""
    for m in pos.legal_moves():
        u = pos.make_move(m)
        mate = pos.is_checkmate()
        pos.unmake_move(m, u)
        if mate:
            return m
    return None

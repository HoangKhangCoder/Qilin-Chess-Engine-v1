"""Đấu đối kháng hai hàm lượng giá để biết mạng có thực sự khá hơn không.

Loss huấn luyện giảm KHÔNG đảm bảo cờ đánh hay hơn: mạng có thể chỉ đang bắt
chước chính hàm lượng giá đã sinh ra dữ liệu. Thước đo thật là kết quả ván đấu
ở CÙNG số nút tìm kiếm (công bằng về thời gian tính toán).

    python match.py --nnue weights/nnue.npz --games 40 --nodes 4000
"""

import argparse
import math
import os
import random
import sys
import time

from chess_core import Game, Position, START_FEN, WHITE, move_str
from search import Searcher
from scoring import MATE_BOUND

MAX_PLIES = 250


def make_eval(nnue_path):
    if nnue_path:
        from nnue import NNUEEvaluator
        return NNUEEvaluator(nnue_path), "NNUE"
    import evaluate
    return evaluate.evaluate_stm, "PeSTO"


def play(eval_a, eval_b, opening_moves, nodes, seconds=None):
    """eval_a cầm Trắng. Trả về 1.0 / 0.5 / 0.0 theo góc nhìn eval_a."""
    game = Game(START_FEN)
    for m in opening_moves:
        game.push(m)
    engines = {WHITE: Searcher(eval_fn=eval_a), 1: Searcher(eval_fn=eval_b)}

    for _ in range(MAX_PLIES):
        res, _ = game.outcome()
        if res:
            return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[res]
        r = engines[game.pos.side].search(game.pos, depth=64, max_nodes=nodes,
                                          time_limit=seconds, history_keys=game.keys)
        if r["move"] is None:
            break
        # Phòng vệ nhiều lớp: search đã được vá để không trả nước bất hợp lệ,
        # nhưng game.push() không kiểm tra gì cả - một nước sai lọt qua đây sẽ
        # làm hỏng âm thầm toàn bộ kết quả đấu, đúng như đã từng xảy ra.
        if r["move"] not in game.pos.legal_moves():
            raise RuntimeError("search tra ve nuoc bat hop le {} tai {}".format(
                move_str(r["move"]), game.pos.fen()))
        game.push(r["move"])
    return 0.5


def random_opening(rng, plies):
    """Sinh khai cuộc ngẫu nhiên; cùng một khai cuộc dùng cho cả hai lượt màu."""
    pos = Position(START_FEN)
    moves = []
    for _ in range(plies):
        legal = pos.legal_moves()
        if not legal:
            return None
        m = rng.choice(legal)
        moves.append(m)
        pos.make_move(m)
    return moves if pos.legal_moves() else None


def elo(score, n):
    """Chênh lệch Elo ước lượng, kèm sai số chuẩn."""
    if n == 0 or score in (0.0, 1.0):
        return None, None
    p = score / n
    e = -400 * math.log10(1 / p - 1)
    se = math.sqrt(p * (1 - p) / n)
    lo = p - 1.96 * se
    hi = p + 1.96 * se
    band = None
    if 0 < lo and hi < 1:
        band = (-400 * math.log10(1 / lo - 1), -400 * math.log10(1 / hi - 1))
    return e, band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnue", required=True, help="trọng số .npz cho phe A")
    ap.add_argument("--baseline", default=None, help="trọng số cho phe B (mặc định: PeSTO)")
    ap.add_argument("--games", type=int, default=40, help="số cặp ván (mỗi cặp đổi màu)")
    ap.add_argument("--nodes", type=int, default=4000)
    ap.add_argument("--time", type=float, default=None,
                    help="giây mỗi nước cho CẢ HAI bên (thay cho --nodes)")
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ea, na = make_eval(args.nnue)
    eb, nb = make_eval(args.baseline)
    rng = random.Random(args.seed)
    print("{} vs {} | {} nút/nước | {} cặp ván".format(na, nb, args.nodes, args.games))

    score = w = d = l = 0.0
    n = 0
    t0 = time.time()
    for i in range(args.games):
        op = random_opening(rng, args.opening_plies)
        if op is None:
            continue
        # cùng khai cuộc, chơi hai lượt đổi màu để triệt tiêu lợi thế tiên thủ
        for a_is_white in (True, False):
            nodes = None if args.time else args.nodes
            r = play(ea, eb, op, nodes, args.time) if a_is_white \
                else 1.0 - play(eb, ea, op, nodes, args.time)
            score += r
            n += 1
            w += r == 1.0
            d += r == 0.5
            l += r == 0.0
            sys.stderr.write("\r{} ván: {:+.1f} (T{} H{} B{})  {:.0f}s".format(
                n, score - n / 2, int(w), int(d), int(l), time.time() - t0))
    print()
    e, band = elo(score, n)
    print("{} ghi {:.1f}/{} = {:.1%}".format(na, score, n, score / n if n else 0))
    if e is None:
        print("chưa đủ dữ liệu để ước lượng Elo")
    elif band:
        print("Elo chênh: {:+.0f}  (khoảng tin cậy 95%: {:+.0f} .. {:+.0f})".format(
            e, band[0], band[1]))
    else:
        print("Elo chênh: {:+.0f}  (mẫu quá nhỏ, khoảng tin cậy không xác định)".format(e))


if __name__ == "__main__":
    main()

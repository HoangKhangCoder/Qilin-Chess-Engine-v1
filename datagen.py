"""Sinh dữ liệu huấn luyện bằng tự chơi (self-play).

Mỗi dòng dữ liệu: FEN | điểm cp của tìm kiếm nông (góc nhìn Trắng) | kết quả ván
Kết quả: 1.0 Trắng thắng, 0.5 hoà, 0.0 Đen thắng.

Nhãn kết hợp hai nguồn này chính là mấu chốt: điểm tìm kiếm cho tín hiệu dày,
ít nhiễu; kết quả ván cho sự thật cuối cùng, chống việc mạng chỉ học lại đúng
hàm lượng giá thủ công đã dùng để sinh dữ liệu.

    python datagen.py --games 2000 --out data/selfplay.txt --workers 8
"""

import argparse
import os
import random
import sys
import time

from chess_core import Position, START_FEN, F_PROMO, F_EP
from search import Searcher
from scoring import MATE_BOUND

RESIGN_CP = 2000
RESIGN_PLIES = 6
MAX_PLIES = 300


BALANCE_CP = 200      # ngưỡng loại khai cuộc đã lệch hẳn


def random_opening(rng, plies, tries=12, eval_fn=None):
    """Khai cuộc ngẫu nhiên nhưng LỌC theo độ cân bằng.

    Đi bừa vài nước tạo ra rất nhiều thế cờ đã thua hẳn hoặc phi lý - học từ
    chúng vừa phí vừa lệch phân phối so với ván cờ thật. Ở đây ta thử lại tới
    khi thế cờ còn tương đối cân bằng, nhờ đó dữ liệu tập trung vào vùng mà
    độ chính xác của hàm lượng giá thực sự quyết định kết quả.
    """
    probe = Searcher(eval_fn=eval_fn)
    for _ in range(tries):
        pos = Position(START_FEN)
        ok = True
        for _ in range(plies):
            moves = pos.legal_moves()
            if not moves:
                ok = False
                break
            pos.make_move(rng.choice(moves))
        if not ok or not pos.legal_moves():
            continue
        if abs(probe.search(pos, depth=64, max_nodes=1500)["score"]) <= BALANCE_CP:
            return pos
    return None


def play_game(rng, nodes_per_move, opening_plies, eval_fn=None):
    """Chơi một ván, trả về danh sách (fen, cp_trắng) và kết quả."""
    pos = random_opening(rng, opening_plies, eval_fn=eval_fn)
    if pos is None:
        return [], None
    searcher = Searcher(eval_fn=eval_fn)
    samples, keys = [], []
    result = None
    high_streak = 0

    for ply in range(MAX_PLIES):
        keys.append(pos.key)
        if pos.halfmove >= 100 or pos.is_insufficient_material():
            result = 0.5
            break
        legal = pos.legal_moves()
        if not legal:
            result = 0.5 if not pos.in_check() else (0.0 if pos.side == 0 else 1.0)
            break

        r = searcher.search(pos, depth=64, max_nodes=nodes_per_move, history_keys=keys)
        cp, best = r["score"], r["move"]
        if best is None:
            result = 0.5
            break

        # lọc mẫu: bỏ thế đang bị chiếu, bỏ nước tốt nhất là ăn quân (nhiễu)
        noisy = pos.in_check() or pos.board[(best >> 6) & 63] >= 0 \
            or (best >> 15) & 7 in (F_PROMO, F_EP)
        if not noisy and abs(cp) < MATE_BOUND:
            samples.append((pos.fen(), cp))

        if abs(cp) > RESIGN_CP or abs(cp) > MATE_BOUND:
            high_streak += 1
            if high_streak >= RESIGN_PLIES:
                result = 1.0 if cp > 0 else 0.0
                break
        else:
            high_streak = 0

        # Không chèn nước đi bừa giữa ván: một nước thí quân ngẫu nhiên làm
        # hỏng nhãn KẾT QUẢ của mọi thế cờ đứng trước nó. Sự đa dạng đã đến
        # từ khai cuộc ngẫu nhiên có lọc ở trên.
        pos.make_move(best)
    if result is None:
        result = 0.5
    return samples, result


_WORKER_EVAL = {}


def _get_eval_fn(nnue_path):
    """Nạp mạng MỘT LẦN cho mỗi tiến trình con rồi dùng lại (trọng số nặng)."""
    if not nnue_path:
        return None
    if nnue_path not in _WORKER_EVAL:
        from nnue import NNUEEvaluator
        _WORKER_EVAL[nnue_path] = NNUEEvaluator(nnue_path)
    return _WORKER_EVAL[nnue_path]


def _worker(args):
    seed, n_games, nodes, opening_plies, nnue_path = args
    eval_fn = _get_eval_fn(nnue_path)
    # Worker của Pool là daemon nên chỉ tự chết khi cha THOÁT BÌNH THƯỜNG.
    # Nếu cha bị kill -9 thì chúng thành mồ côi và ăn hết CPU, nên tự canh:
    # ppid đổi (thường thành 1) nghĩa là cha đã chết -> dừng ngay.
    ppid = os.getppid()
    rng = random.Random(seed)
    out = []
    for _ in range(n_games):
        if os.getppid() != ppid:
            break
        samples, result = play_game(rng, nodes, rng.randint(*opening_plies), eval_fn)
        for fen, cp in samples:
            out.append((fen, cp, result))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--nodes", type=int, default=4000, help="số nút mỗi nước đi")
    ap.add_argument("--out", default="data/selfplay.txt")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opening-min", type=int, default=4)
    ap.add_argument("--opening-max", type=int, default=10)
    ap.add_argument("--nnue", default=None,
                    help="sinh dữ liệu VÒNG SAU bằng mạng đã train (.npz) "
                         "thay vì hàm lượng giá thủ công")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    total = 0
    op = (args.opening_min, args.opening_max)

    with open(args.out, "a") as f:
        if args.workers > 1:
            import multiprocessing as mp
            chunk = max(1, min(4, args.games // (args.workers * 4)))
            tasks = []
            remaining = args.games
            s = args.seed
            while remaining > 0:
                n = min(chunk, remaining)
                tasks.append((s, n, args.nodes, op, args.nnue))
                s += 1
                remaining -= n
            pool = mp.Pool(args.workers)
            try:
                done = 0
                for rows in pool.imap_unordered(_worker, tasks):
                    for fen, cp, res in rows:
                        f.write("{}|{}|{}\n".format(fen, cp, res))
                    total += len(rows)
                    done += 1
                    f.flush()
                    sys.stderr.write("\r{}/{} lô, {} thế cờ, {:.0f}s".format(
                        done, len(tasks), total, time.time() - t0))
            except KeyboardInterrupt:
                sys.stderr.write("\nDừng theo yêu cầu.\n")
            finally:
                pool.terminate()
                pool.join()
        else:
            rng = random.Random(args.seed)
            eval_fn = _get_eval_fn(args.nnue)
            for g in range(args.games):
                samples, result = play_game(rng, args.nodes, rng.randint(*op), eval_fn)
                for fen, cp in samples:
                    f.write("{}|{}|{}\n".format(fen, cp, result))
                total += len(samples)
                sys.stderr.write("\r{}/{} ván, {} thế cờ, {:.0f}s".format(
                    g + 1, args.games, total, time.time() - t0))
    sys.stderr.write("\nĐã ghi {} thế cờ vào {}\n".format(total, args.out))


if __name__ == "__main__":
    main()

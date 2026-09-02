"""Tạo sách khai cuộc bằng chính Stockfish (chỉ dùng khi huấn luyện).

Vấn đề: bảo Stockfish đánh nước tốt nhất từ thế xuất phát thì lần nào cũng ra
đúng một biến. Đi bừa thì đa dạng nhưng cấu trúc tốt vô nghĩa.

Giải pháp - đi bộ bằng MultiPV: ở mỗi nước, hỏi Stockfish K nước tốt nhất, giữ
lại những nước chênh không quá MARGIN centipawn so với nước hay nhất, rồi bốc
ngẫu nhiên. Mọi nước trong sách đều là nước Stockfish chấp nhận được, nhưng
đường đi thì rẽ nhánh liên tục.

Sách chỉ cần tạo MỘT LẦN rồi dùng lại cho hàng trăm nghìn ván, nên chi phí
MultiPV được chia đều gần như bằng không.

    python make_book.py --count 20000 --out data/book.epd
    python datagen_sf.py --games 400000 --epd data/book.epd
"""

import argparse
import os
import random
import sys
import time

import chess
import chess.engine

SF = os.environ.get("STOCKFISH_PATH", "/opt/homebrew/bin/stockfish")


def walk(eng, rng, plies, depth, multipv, margin, early_bonus):
    """Đi bộ một biến khai cuộc, trả về Board hoặc None."""
    b = chess.Board()
    for ply in range(plies):
        if b.is_game_over():
            return None
        infos = eng.analyse(b, chess.engine.Limit(depth=depth), multipv=multipv)
        if not isinstance(infos, list):
            infos = [infos]
        cands, best_cp = [], None
        for info in infos:
            pv = info.get("pv")
            if not pv:
                continue
            sc = info["score"].pov(b.turn)
            cp = sc.score(mate_score=100000)
            if best_cp is None:
                best_cp = cp
            # nới rộng ngưỡng ở vài nước đầu để cây rẽ nhánh mạnh hơn,
            # nếu không mọi biến sẽ chụm lại quanh 1.e4 / 1.d4
            m = margin + (early_bonus if ply < 4 else 0)
            if best_cp - cp <= m:
                cands.append(pv[0])
        if not cands:
            return None
        b.push(rng.choice(cands))
    return b if not b.is_game_over() else None


def _worker(args):
    seed, count, plies_lo, plies_hi, depth, multipv, margin, early, out = args
    ppid = os.getppid()
    rng = random.Random(seed)
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 32})
    n = 0
    try:
        with open(out, "a") as f:
            while n < count:
                if os.getppid() != ppid:
                    break
                b = walk(eng, rng, rng.randint(plies_lo, plies_hi),
                         depth, multipv, margin, early)
                if b is None:
                    continue
                f.write(b.fen() + "\n")
                f.flush()
                n += 1
    finally:
        eng.quit()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20000)
    ap.add_argument("--out", default="data/book.epd")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--multipv", type=int, default=8)
    ap.add_argument("--margin", type=int, default=40,
                    help="chênh cp tối đa so với nước hay nhất thì vẫn nhận")
    ap.add_argument("--early-bonus", type=int, default=40,
                    help="nới thêm ngưỡng ở 4 nước đầu để rẽ nhánh mạnh")
    ap.add_argument("--plies-min", type=int, default=8)
    ap.add_argument("--plies-max", type=int, default=16)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    per = max(1, args.count // args.workers)
    tasks = [(args.seed + i, per, args.plies_min, args.plies_max, args.depth,
              args.multipv, args.margin, args.early_bonus, args.out)
             for i in range(args.workers)]
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        pool = mp.Pool(args.workers)
        try:
            for _ in pool.imap_unordered(_worker, tasks):
                pass
        except KeyboardInterrupt:
            sys.stderr.write("\nDừng.\n")
        finally:
            pool.terminate()
            pool.join()
    else:
        _worker(tasks[0])

    # khử trùng lặp theo thế cờ (bỏ hai số đếm nước cuối FEN)
    seen, kept = set(), []
    for line in open(args.out):
        line = line.strip()
        if not line:
            continue
        k = " ".join(line.split()[:4])
        if k not in seen:
            seen.add(k)
            kept.append(line)
    with open(args.out, "w") as f:
        f.write("\n".join(kept) + "\n")
    print("\n{:,} thế cờ khai cuộc duy nhất -> {}  ({:.0f}s)".format(
        len(kept), args.out, time.time() - t0), file=sys.stderr)


if __name__ == "__main__":
    main()

"""Engine của ta đấu Stockfish nhiều ván, ghi PGN đầy đủ.

Công cụ ĐO, được phép dùng thư viện ngoài. Engine của ta vẫn chỉ dùng
chess_core/search/nnue; python-chess ở đây để ghi SAN/PGN, Stockfish là đối thủ.

Hai điểm về tính công bằng:

* Khai cuộc BẮT CẶP: mỗi thế cờ trong sách được đấu ĐÚNG HAI LẦN, một lần ta
  cầm Trắng một lần ta cầm Đen. Nhờ vậy số ván mỗi màu bằng nhau tuyệt đối, và
  phương sai giảm mạnh - nếu một khai cuộc vốn đã lợi cho Trắng thì cả hai bên
  đều được hưởng nó đúng một lần.
* Cùng ngân sách thời gian mỗi nước cho cả hai bên.

    python play_stockfish.py --games 1000 --time 0.5 --workers 4 \
        --nnue weights/cap_kb64_h256.npz --book data/book.epd
"""

import argparse
import datetime
import os
import random
import sys
import time

import chess
import chess.engine
import chess.pgn

from chess_core import Game, Position, START_FEN, move_str, parse_uci
from search import Searcher
import evaluate
import scoring

SF = os.environ.get("STOCKFISH_PATH", "/opt/homebrew/bin/stockfish")
MAX_PLIES = 300
_W = {}                       # tài nguyên mỗi tiến trình con, tạo một lần


def make_eval(nnue_path):
    if nnue_path:
        from nnue import NNUEEvaluator
        ev = NNUEEvaluator(nnue_path)
        return ev, "NNUE-" + os.path.basename(nnue_path).replace(".npz", "")
    return evaluate.evaluate_stm, "PeSTO"


def _setup(cfg):
    """Nạp engine + Stockfish MỘT LẦN cho mỗi tiến trình con."""
    key = id(cfg)
    if key not in _W:
        our_eval, our_name = make_eval(cfg["nnue"])
        scoring.calibrate(Searcher(eval_fn=our_eval).search(
            Position(START_FEN), depth=8)["score"])
        eng = chess.engine.SimpleEngine.popen_uci(SF)
        opts = {"Threads": 1, "Hash": 32}
        if cfg["sf_elo"]:
            opts["UCI_LimitStrength"] = True
            opts["UCI_Elo"] = cfg["sf_elo"]
        eng.configure(opts)
        _W[key] = (our_eval, our_name, eng)
    return _W[key]


def play_one(cfg, opening_fen, our_is_white, round_no):
    our_eval, our_name, eng = _setup(cfg)
    limit = chess.engine.Limit(time=cfg["seconds"])
    searcher = Searcher(eval_fn=our_eval)

    board = chess.Board(opening_fen) if opening_fen else chess.Board()
    ours = Game(opening_fen or START_FEN)
    start_board = board.copy()
    notes = {}

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < MAX_PLIES:
        our_turn = board.turn == (chess.WHITE if our_is_white else chess.BLACK)
        if our_turn:
            r = searcher.search(ours.pos, depth=64, time_limit=cfg["seconds"],
                                history_keys=ours.keys)
            if r["move"] is None:
                break
            uci = move_str(r["move"])
            notes[len(board.move_stack)] = "{}/1000 d{} {}n".format(
                scoring.cp_to_score(r["score"]), r["depth"], r["nodes"])
        else:
            res = eng.play(board, limit)
            if res.move is None:
                break
            uci = res.move.uci()

        mv = parse_uci(ours.pos, uci)
        if mv is None:
            raise RuntimeError("engine ta coi {} khong hop le tai {}".format(
                uci, ours.pos.fen()))
        ours.push(mv)
        board.push_uci(uci)
        if ours.pos.fen() != board.fen():
            raise RuntimeError("lech FEN sau {}: {} vs {}".format(
                uci, ours.pos.fen(), board.fen()))

    game = chess.pgn.Game()
    if opening_fen:
        game.setup(start_board)
    sf_name = "Stockfish-{}".format(cfg["sf_elo"]) if cfg["sf_elo"] else "Stockfish"
    game.headers["Event"] = "{} vs {} ({}s/nuoc)".format(our_name, sf_name, cfg["seconds"])
    game.headers["Site"] = "localhost"
    game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
    game.headers["Round"] = str(round_no)
    game.headers["White"] = our_name if our_is_white else sf_name
    game.headers["Black"] = sf_name if our_is_white else our_name
    node = game
    for i, mv in enumerate(board.move_stack):
        node = node.add_variation(mv)
        if i in notes:
            node.comment = notes[i]
    res = board.result(claim_draw=True)
    game.headers["Result"] = res
    game.headers["Termination"] = ("checkmate" if board.is_checkmate() else
                                   "draw" if board.is_game_over(claim_draw=True)
                                   else "adjudicated")
    score = 0.5 if res == "1/2-1/2" else (
        1.0 if (res == "1-0") == our_is_white else 0.0)
    return str(game), score, res, our_is_white


def _job(args):
    cfg, fen, white, rnd = args
    try:
        return play_one(cfg, fen, white, rnd)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, "ERROR: {}".format(e), white


def elo(score, n):
    if n == 0 or score in (0, n):
        return None, None
    p = score / n
    import math
    e = -400 * math.log10(1 / p - 1)
    se = math.sqrt(p * (1 - p) / n)
    lo, hi = max(1e-9, p - 1.96 * se), min(1 - 1e-9, p + 1.96 * se)
    return e, (-400 * math.log10(1 / lo - 1), -400 * math.log10(1 / hi - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--nnue", default=None)
    ap.add_argument("--time", type=float, default=0.5,
                    help="giây mỗi nước, ÁP DỤNG CHO CẢ HAI BÊN")
    ap.add_argument("--sf-elo", type=int, default=2000,
                    help="0 = Stockfish sức đầy đủ")
    ap.add_argument("--ladder", default=None,
                    help="danh sách Elo cách nhau bởi dấu phẩy, vd 1600,1800,2000,2200; "
                         "chia đều số ván cho từng mức")
    ap.add_argument("--book", default="data/book.epd")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="games/vs_sf_1000.pgn")
    args = ap.parse_args()

    levels = ([int(x) for x in args.ladder.split(",")] if args.ladder
              else [args.sf_elo if args.sf_elo else None])
    per_level = args.games // len(levels)
    print("THANG ELO: {} | {} van moi muc | {}s/nuoc CA HAI BEN\n".format(
        ", ".join(str(l) if l else "day du" for l in levels), per_level, args.time),
        flush=True)
    summary = []
    for lvl in levels:
        summary.append(run_level(args, lvl, per_level))
    print("\n" + "=" * 62)
    print("{:<14} {:>10} {:>9} {:>22}".format("Doi thu", "Diem", "Ti le", "Elo chenh"))
    for lvl, tot, n, e, ci in summary:
        name = "SF-{}".format(lvl) if lvl else "SF day du"
        el = "{:+.0f} [{:+.0f}..{:+.0f}]".format(e, *ci) if e is not None else "n/a"
        print("{:<14} {:>4.1f}/{:<5.0f} {:>8.1f}% {:>22}".format(
            name, tot, n, tot / max(n, 1) * 100, el))
    print("\nUoc luong Elo cua engine: xem muc nao gan 50% nhat.")
    return


def run_level(args, lvl, ngames):
    cfg = {"nnue": args.nnue, "seconds": args.time, "sf_elo": lvl}
    out = args.out.replace(".pgn", "_elo{}.pgn".format(lvl or "full"))

    # Chạy tiếp: đếm ván đã có trong file để bỏ qua, phòng khi tiến trình bị
    # giết giữa chừng (đã xảy ra một lần, mất 12 tiếng).
    done = 0
    if os.path.exists(out):
        done = open(out).read().count("[Event ")
        if done:
            print("  {}: da co {} van, chay tiep".format(out, done), flush=True)

    rng = random.Random(args.seed)
    pairs = ngames // 2
    book = []
    if args.book and os.path.exists(args.book):
        book = [l.strip() for l in open(args.book) if l.strip()]
        rng.shuffle(book)
    jobs = []
    for i in range(pairs):
        fen = book[i % len(book)] if book else None
        jobs.append((cfg, fen, True, i + 1))     # ta cầm Trắng
        jobs.append((cfg, fen, False, i + 1))    # cùng khai cuộc, ta cầm Đen
    jobs = jobs[done:]                            # bỏ những ván đã chơi

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print("--- SF Elo {} | con {} van ---".format(lvl or "day du", len(jobs)), flush=True)
    if not jobs:
        return (lvl, 0.0, 0, None, None)

    t0 = time.time()
    tot = n = nw = nb = 0.0
    ww = wb = 0.0
    f = open(out, "a")
    try:
        if args.workers > 1:
            import multiprocessing as mp
            pool = mp.Pool(args.workers)
            it = pool.imap_unordered(_job, jobs)
        else:
            pool = None
            it = map(_job, jobs)
        for pgn, sc, res, white in it:
            if pgn is None:
                print("  loi:", res, flush=True)
                continue
            f.write(pgn + "\n\n")
            f.flush()
            n += 1
            tot += sc
            if white:
                nw += 1
                ww += sc
            else:
                nb += 1
                wb += sc
            if n % 10 == 0 or n == len(jobs):
                e, ci = elo(tot, n)
                el = "Elo {:+.0f} [{:+.0f}..{:+.0f}]".format(e, *ci) if e is not None else "Elo n/a"
                sys.stdout.write("\r  {}/{} van | {:.1f} diem = {:.1f}% | {} | Trang {:.1f}/{:.0f} "
                                 "Den {:.1f}/{:.0f} | {:.0f}s".format(
                                     int(n), len(jobs), tot, tot / n * 100, el,
                                     ww, nw, wb, nb, time.time() - t0))
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nDung theo yeu cau.")
    finally:
        if args.workers > 1 and pool:
            pool.terminate()
            pool.join()
        f.close()
    e, ci = elo(tot, n)
    print("\n  KET QUA SF-{}: {:.1f}/{:.0f} = {:.1f}%  | Trang {:.1f}/{:.0f} Den {:.1f}/{:.0f}"
          .format(lvl or "full", tot, n, tot / max(n, 1) * 100, ww, nw, wb, nb), flush=True)
    if e is not None:
        print("  Elo chenh: {:+.0f} (95%: {:+.0f} .. {:+.0f})  PGN: {}".format(e, *ci, out),
              flush=True)
    return (lvl, tot, n, e, ci)


if __name__ == "__main__":
    main()

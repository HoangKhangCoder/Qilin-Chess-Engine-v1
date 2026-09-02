"""Chấm điểm thế cờ trên thang 0..1000 cho bên Trắng.

    python main.py                                  # thế xuất phát -> 505
    python main.py "<FEN>" --depth 10
    python main.py --moves e2e4 e7e5 g1f3
    python main.py "<FEN>" --nnue weights/nnue.npz  # dùng mạng nơ-ron
    python main.py --interactive
"""

import argparse
import json
import os
import sys
import time

from chess_core import Position, Game, START_FEN, move_str, sq_name
from search import Searcher, find_mate_in_one, pv_string
import evaluate
import scoring

CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".calib.json")


def load_evaluator(nnue_path):
    """Trả về (eval_fn_góc_nhìn_bên_đi, eval_fn_góc_nhìn_trắng, tên)."""
    if nnue_path:
        if not os.path.exists(nnue_path):
            sys.exit("Không thấy file trọng số: {}\nChạy datagen.py rồi train.py trước."
                     .format(nnue_path))
        from nnue import NNUEEvaluator
        ev = NNUEEvaluator(nnue_path)
        return ev, ev.white_cp, "NNUE ({})".format(os.path.basename(nnue_path))
    return evaluate.evaluate_stm, evaluate.evaluate, "thủ công (PeSTO)"


def get_calibration(eval_name, depth, eval_fn):
    """Hiệu chỉnh sao cho thế xuất phát ra đúng 505 điểm."""
    key = "{}@d{}".format(eval_name, depth)
    cache = {}
    if os.path.exists(CALIB_FILE):
        try:
            cache = json.load(open(CALIB_FILE))
        except Exception:
            cache = {}
    if key not in cache:
        r = Searcher(eval_fn=eval_fn).search(Position(START_FEN), depth=depth)
        cache[key] = r["score"]
        try:
            json.dump(cache, open(CALIB_FILE, "w"), indent=1)
        except Exception:
            pass
    scoring.calibrate(cache[key])
    return cache[key]


def analyse(game, searcher, depth, time_limit, white_eval_fn, verbose=False):
    """Chấm một thế cờ -> dict kết quả."""
    pos = game.pos

    # 0) ván đã hoà theo luật -> đúng 500 điểm, khỏi cần tìm kiếm
    draw = game.draw_reason()
    if draw:
        return {"score": 500, "cp": 0, "move": None, "pv": [], "depth": 0, "nodes": 0,
                "note": "HOÀ CỜ: " + draw, "static": white_eval_fn(pos)}

    # 1) chiếu hết ngay trong nước này = mốc 1000 (hoặc 0)
    mate1 = find_mate_in_one(pos)
    if mate1 is not None:
        white_to_move = (pos.side == 0)
        who = "Trắng" if white_to_move else "Đen"
        return {"score": 1000 if white_to_move else 0, "cp": None,
                "move": mate1, "pv": [mate1], "depth": 1, "nodes": 0,
                "note": "{} CHIẾU HẾT ngay nước này: {}".format(who, move_str(mate1)),
                "static": white_eval_fn(pos)}

    # 2) đã bị chiếu hết (pat đã bắt ở bước 0)
    if not pos.legal_moves():
        loser = "Trắng" if pos.side == 0 else "Đen"
        return {"score": 0 if pos.side == 0 else 1000, "cp": None, "move": None,
                "pv": [], "depth": 0, "nodes": 0,
                "note": "{} đã bị chiếu hết - ván đã kết thúc".format(loser),
                "static": white_eval_fn(pos)}

    cb = None
    if verbose:
        def cb(d, s, m, n):
            print("  d{:>2} {:>6}cp  {:>4} điểm  {:>9} nút  {}".format(
                d, s if pos.side == 0 else -s, scoring.cp_to_score(
                    s if pos.side == 0 else -s), n, move_str(m)),
                file=sys.stderr)

    t = time.time()
    r = searcher.search(pos, depth=depth, time_limit=time_limit,
                        history_keys=game.keys, on_iteration=cb)
    r["elapsed"] = time.time() - t
    r["cp"] = r["score"]
    r["score"] = scoring.cp_to_score(r["cp"])
    r["static"] = white_eval_fn(pos)
    r["note"] = ""
    return r


def report(game, r, eval_name):
    pos = game.pos
    print(pos)
    print()
    s = r["score"]
    print("  ĐIỂM TRẮNG: {:>4} / 1000   {}".format(s, scoring.bar(s)))
    print("  {}".format(r["note"] or scoring.explain(s, r.get("cp"))))

    # trạng thái các luật cần lịch sử
    flags = []
    if pos.in_check():
        flags.append("đang bị chiếu")
    rep = game.repetitions()
    if rep > 1:
        flags.append("thế cờ đã lặp {} lần".format(rep))
    if pos.halfmove >= 80:
        flags.append("đồng hồ 50 nước: {}/100 nửa nước".format(pos.halfmove))
    cast = "".join(c for c, b in (("K", 1), ("Q", 2), ("k", 4), ("q", 8)) if pos.castling & b)
    if cast:
        flags.append("nhập thành còn: " + cast)
    if pos.ep >= 0:
        flags.append("bắt tốt qua đường tại " + sq_name(pos.ep))
    if flags:
        print("  luật: " + " | ".join(flags))
    print()
    line = "  lượng giá: {} | tĩnh {:+d}cp".format(eval_name, r["static"])
    if r.get("cp") is not None:
        n = scoring.mate_distance(r["cp"])
        line += " | sâu {}".format("#{}".format(n) if n else "{:+d}cp".format(r["cp"]))
    if r.get("depth"):
        line += " | độ sâu {}".format(r["depth"])
        if r.get("seldepth"):
            line += "/{}".format(r["seldepth"])
    if r.get("nodes"):
        nps = r["nodes"] / max(r.get("elapsed", 1e-9), 1e-9)
        line += " | {} nút, {:.1f}s ({:.0f}k nút/s)".format(
            r["nodes"], r["elapsed"], nps / 1000)
    print(line)
    if r["pv"]:
        print("  nước tốt nhất: {}".format(move_str(r["pv"][0])))
        print("  biến chính:    {}".format(pv_string(pos, r["pv"])))
    res, desc = game.outcome()
    if res:
        print("  kết cục: {} ({})".format(res, desc))
    print()


def main():
    ap = argparse.ArgumentParser(description="Chấm điểm thế cờ 0..1000 cho Trắng")
    ap.add_argument("fen", nargs="?", default=None, help="FEN (mặc định: thế xuất phát)")
    ap.add_argument("--moves", nargs="*", default=None, help="các nước UCI từ thế xuất phát")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--time", type=float, default=None, help="giới hạn giây/thế cờ")
    ap.add_argument("--nnue", default=None, help="đường dẫn trọng số .npz")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true", help="in từng vòng lặp sâu dần")
    ap.add_argument("--recalibrate", action="store_true", help="tính lại mốc 505")
    args = ap.parse_args()

    if args.recalibrate and os.path.exists(CALIB_FILE):
        os.remove(CALIB_FILE)

    eval_fn, white_eval_fn, eval_name = load_evaluator(args.nnue)
    cp0 = get_calibration(eval_name, args.depth, eval_fn)
    searcher = Searcher(eval_fn=eval_fn)

    game = Game(args.fen or START_FEN)
    for s in (args.moves or []):
        try:
            game.push_uci(s)
        except ValueError as e:
            sys.exit(str(e))

    if args.interactive:
        print("Nhập nước đi UCI (vd e2e4), 'fen <FEN>', 'undo', hoặc 'q' để thoát.")
        while True:
            r = analyse(game, searcher, args.depth, args.time, white_eval_fn, args.verbose)
            report(game, r, eval_name)
            try:
                cmd = input("> ").strip()
            except EOFError:
                break
            if cmd in ("q", "quit", "exit"):
                break
            if cmd == "undo":
                if game.pop() is None:
                    print("Không còn nước để lùi.")
                continue
            if cmd.startswith("fen "):
                try:
                    game = Game(cmd[4:].strip())
                except Exception as e:
                    print("FEN sai:", e)
                continue
            try:
                game.push_uci(cmd)
            except ValueError:
                print("Nước không hợp lệ.")
        return

    r = analyse(game, searcher, args.depth, args.time, white_eval_fn, args.verbose)
    report(game, r, eval_name)
    print("  (mốc hiệu chỉnh: thế xuất phát chấm {:+d}cp -> quy về đúng 505 điểm)".format(cp0))


if __name__ == "__main__":
    main()

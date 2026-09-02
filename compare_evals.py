"""So ba hàm lượng giá với nhãn Stockfish trên tập giữ lại.

    python -u compare_evals.py data/_holdout_clean.txt
"""
import math, statistics, sys, time
from chess_core import Position
from search import Searcher
import evaluate, scoring
from nnue import NNUEEvaluator

path = sys.argv[1] if len(sys.argv) > 1 else "data/_holdout_clean.txt"
rows = [(l.rsplit('|', 1)[0], int(l.rsplit('|', 1)[1])) for l in open(path)]

conf = [("Thủ công PeSTO", evaluate.evaluate_stm, evaluate.evaluate)]
for lab, p in [("NNUE lambda=0.7", "weights/nnue_kb4.npz"),
               ("NNUE lambda=1.0", "weights/nnue_kb4_lam1.npz")]:
    try:
        ev = NNUEEvaluator(p)
        conf.append((lab, ev, ev.white_cp))
    except Exception as e:
        print("bỏ qua %s: %s" % (lab, e), flush=True)


def corr(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


ref = None
print("%d thế cờ giữ lại, nhãn Stockfish depth 12\n" % len(rows), flush=True)
print("%-18s %12s %11s %14s %11s" % ("", "r (tĩnh)", "sai số", "r (search d6)", "sai số"), flush=True)
for label, ev_stm, ev_white in conf:
    t0 = time.time()
    scoring.calibrate(ev_white(Position()))
    a = [scoring.cp_to_score(ev_white(Position(f))) for f, _ in rows]
    ref = [scoring.cp_to_score(c) for _, c in rows]
    r_s, e_s = corr(a, ref), statistics.mean(abs(x - y) for x, y in zip(a, ref))

    scoring.calibrate(Searcher(eval_fn=ev_stm).search(Position(), depth=8)["score"])
    s = Searcher(eval_fn=ev_stm)
    a = [scoring.cp_to_score(s.search(Position(f), depth=6)["score"]) for f, _ in rows]
    ref = [scoring.cp_to_score(c) for _, c in rows]
    print("%-18s %12.4f %11.1f %14.4f %11.1f   (%.0fs)" % (
        label, r_s, e_s, corr(a, ref),
        statistics.mean(abs(x - y) for x, y in zip(a, ref)), time.time() - t0), flush=True)
print("\nxong", flush=True)

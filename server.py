"""Máy chủ bàn phân tích - chỉ dùng thư viện chuẩn Python.

    python3 server.py --depth 8
    python3 server.py --nnue weights/cap_kb64_h256.npz --depth 8

Rồi mở http://localhost:8000

File này thuộc nhóm ENGINE SẠCH: không python-chess, không Stockfish. Mọi luật
cờ, sinh nước đi và lượng giá đều đến từ chess_core/search/evaluate/nnue.
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chess_core import Game, Position, START_FEN, move_str, parse_uci, sq_name
from search import Searcher, find_mate_in_one
import evaluate
import scoring

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = threading.Lock()          # engine dùng chung, một phân tích tại một thời điểm
ENGINES = {}                     # tên -> (eval_stm, eval_white, cp_thế_xuất_phát)


def register_engine(name, nnue_path, depth):
    if nnue_path:
        from nnue import NNUEEvaluator
        ev = NNUEEvaluator(nnue_path)
        stm, white = ev, ev.white_cp
    else:
        stm, white = evaluate.evaluate_stm, evaluate.evaluate
    cp0 = Searcher(eval_fn=stm).search(Position(START_FEN), depth=depth)["score"]
    ENGINES[name] = (stm, white, cp0)
    print("  {:<8} hiệu chỉnh: thế xuất phát {:+d}cp -> 505 điểm".format(name, cp0))


def draw_en(vi):
    """Dịch lý do hoà sang tiếng Anh (Game.draw_reason trả về tiếng Việt)."""
    if "50 nước" in vi:
        return "fifty-move rule"
    if "lặp lại" in vi:
        return "threefold repetition"
    if "đủ lực" in vi:
        return "insufficient material"
    if "PAT" in vi or "hết nước" in vi:
        return "stalemate"
    return vi


def outcome_en(res, vi):
    if res == "1/2-1/2":
        return "draw - " + draw_en(vi)
    if res in ("1-0", "0-1"):
        return "{} wins by checkmate".format("White" if res == "1-0" else "Black")
    return "game in progress"


def build_game(start_fen, moves):
    """Trả về (Game, các nước đã áp dụng, các nước bị từ chối).

    Nước bất hợp lệ KHÔNG được nuốt im lặng: nếu chỉ `break` thì ván bị cắt
    cụt và giao diện thấy bàn cờ lùi về thế cũ mà không hiểu tại sao. Trả
    danh sách bị từ chối lên để phía giao diện báo cho người dùng.
    """
    g = Game(start_fen or START_FEN)
    applied, rejected = [], []
    for s in moves:
        m = parse_uci(g.pos, s)
        if m is None:
            rejected.append(s)
            break
        g.push(m)
        applied.append(s)
    return g, applied, rejected


def analyse(start_fen, moves, depth, engine, on_depth=None):
    stm_fn, white_fn, cp0 = ENGINES[engine]
    g, applied, rejected = build_game(start_fen, moves)
    pos = g.pos
    out = {
        "fen": pos.fen(),
        "side": "w" if pos.side == 0 else "b",
        "moves": applied,
        "rejected": rejected,
        "legal": sorted(move_str(m) for m in pos.legal_moves()),
        "in_check": pos.in_check(),
        "halfmove": pos.halfmove,
        "repetitions": g.repetitions(),
        "castling": "".join(c for c, b in (("K", 1), ("Q", 2), ("k", 4), ("q", 8))
                            if pos.castling & b),
        "ep": sq_name(pos.ep) if pos.ep >= 0 else None,
        "engine": engine,
    }
    res, desc = g.outcome()
    out["outcome"] = res
    out["outcome_text"] = desc
    out["outcome_text_en"] = outcome_en(res, desc)

    scoring.calibrate(cp0)

    # ván đã kết thúc hoặc đã hoà theo luật -> không cần tìm kiếm
    draw = g.draw_reason()
    if draw:
        out.update(score=500, cp=None, best=None, pv=[],
                   note="HOÀ CỜ: " + draw, note_en="DRAW: " + draw_en(draw),
                   depth=0, nodes=0, elapsed=0.0)
        return out
    if not out["legal"]:
        loser = "Trắng" if pos.side == 0 else "Đen"
        out.update(score=0 if pos.side == 0 else 1000, cp=None, best=None, pv=[],
                   note="{} đã bị chiếu hết".format(loser),
                   note_en="{} has been checkmated".format("White" if pos.side == 0 else "Black"),
                   depth=0, nodes=0, elapsed=0.0)
        return out

    mate1 = find_mate_in_one(pos)
    if mate1 is not None:
        who = "Trắng" if pos.side == 0 else "Đen"
        out.update(score=1000 if pos.side == 0 else 0, cp=None,
                   best=move_str(mate1), pv=[move_str(mate1)],
                   note="{} CHIẾU HẾT ngay nước này".format(who),
                   note_en="{} MATES on this move".format(
                       "White" if pos.side == 0 else "Black"),
                   depth=1, nodes=0, elapsed=0.0)
        return out

    # Phát thông tin thế cờ NGAY, trước khi tốn giây nào cho tìm kiếm: giao diện
    # vẽ được quân cờ và cho bấm nước đi luôn, không phải chờ phân tích xong.
    if on_depth is not None:
        init = {k: out[k] for k in ("fen", "side", "legal", "in_check", "halfmove",
                                    "repetitions", "castling", "ep", "engine",
                                    "outcome", "outcome_text")}
        init.update(partial=True, init=True, static=white_fn(pos))
        on_depth(init)

    t = time.time()
    searcher = Searcher(eval_fn=stm_fn)

    def cb(d, cp_stm, mv, nodes):
        if on_depth is None:
            return
        cp = cp_stm if pos.side == 0 else -cp_stm
        # Phát ngay khi xong mỗi tầng: giao diện có mũi tên sau ~30ms thay vì
        # đợi trọn 1-3 giây. Nếu client đã ngắt, ghi sẽ lỗi -> ngoại lệ bay lên
        # và tự huỷ luôn tìm kiếm, khỏi phí CPU cho kết quả không ai nhận.
        on_depth({"partial": True, "depth": d, "nodes": nodes,
                  "cp": cp, "score": scoring.cp_to_score(cp),
                  "best": move_str(mv) if mv else None,
                  "pv": [move_str(x) for x in searcher.extract_pv(pos, 6)],
                  "elapsed": round(time.time() - t, 2)})

    r = searcher.search(pos, depth=depth, history_keys=g.keys, on_iteration=cb)
    out["partial"] = False
    out.update(
        score=scoring.cp_to_score(r["score"]),
        cp=r["score"],
        mate_in=scoring.mate_distance(r["score"]) or None,
        best=move_str(r["move"]) if r["move"] else None,
        pv=[move_str(m) for m in r["pv"]],
        note="",
        depth=r["depth"], seldepth=r.get("seldepth"),
        nodes=r["nodes"], elapsed=round(time.time() - t, 2),
        static=white_fn(pos),
    )
    out["explain"] = scoring.explain(out["score"], out["cp"])
    out["explain_en"] = scoring.explain(out["score"], out["cp"], lang="en")

    # Điểm SAU KHI đi nước tốt nhất - cho thấy nước đó dẫn tới đâu
    if r["move"]:
        u = pos.make_move(r["move"])
        try:
            after = Searcher(eval_fn=stm_fn).search(pos, depth=max(1, depth - 2))
        finally:
            # try/finally bắt buộc: nếu search ném ngoại lệ mà không hoàn tác
            # nước đi thì `pos` hỏng và mọi thứ tính sau đó đều sai.
            pos.unmake_move(r["move"], u)
        out["score_after"] = scoring.cp_to_score(after["score"])
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "web", "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                return self._send(404, b"web/index.html khong tim thay", "text/plain")
        if path == "/api/engines":
            return self._send(200, json.dumps(sorted(ENGINES)))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path == "/api/stream":
            return self.stream()
        if self.path != "/api/analyse":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            engine = req.get("engine") or sorted(ENGINES)[0]
            if engine not in ENGINES:
                engine = sorted(ENGINES)[0]
            depth = max(1, min(14, int(req.get("depth", 8))))
            with LOCK:
                out = analyse(req.get("start"), req.get("moves") or [], depth, engine)
            self._send(200, json.dumps(out))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, json.dumps({"error": "{}: {}".format(type(e).__name__, e)}))


    def stream(self):
        """Trả kết quả theo dòng NDJSON, mỗi tầng độ sâu một dòng."""
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, json.dumps({"error": "bad request"}))
        engine = req.get("engine") or sorted(ENGINES)[0]
        if engine not in ENGINES:
            engine = sorted(ENGINES)[0]
        depth = max(1, min(14, int(req.get("depth", 8))))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(obj):
            self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
            self.wfile.flush()

        try:
            with LOCK:
                out = analyse(req.get("start"), req.get("moves") or [],
                              depth, engine, on_depth=emit)
            emit(out)
        except (BrokenPipeError, ConnectionResetError):
            pass          # client đã bỏ đi (người dùng đi nước khác) - bình thường
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                emit({"error": "{}: {}".format(type(e).__name__, e)})
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--depth", type=int, default=8, help="độ sâu dùng khi hiệu chỉnh mốc 505")
    ap.add_argument("--nnue", default=None, help="trọng số .npz; thêm engine 'nnue'")
    args = ap.parse_args()

    print("Hiệu chỉnh thang điểm (thế xuất phát phải ra đúng 505)...")
    register_engine("PeSTO", None, args.depth)
    if args.nnue:
        if not os.path.exists(args.nnue):
            sys.exit("Không thấy {}".format(args.nnue))
        register_engine("NNUE", args.nnue, args.depth)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("\n  Bàn phân tích:  http://localhost:{}\n  Ctrl-C để dừng".format(args.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nDừng.")


if __name__ == "__main__":
    main()

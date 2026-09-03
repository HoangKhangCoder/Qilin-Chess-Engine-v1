"""Test hồi quy: luật cờ (perft), thang điểm, và tìm kiếm.

    python test_engine.py
"""

import sys

from chess_core import Position, Game, START_FEN, perft, move_str, WHITE
from search import Searcher, find_mate_in_one
import scoring
import evaluate

PERFT_CASES = [
    (START_FEN, [20, 400, 8902, 197281]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [46, 2079, 89890]),
]

MATE_IN_ONE = [
    ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),
    # chiếu hết học trò: tượng c4 yểm f7 nên Hxf7 là chiếu hết
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1", "f3f7"),
    ("4k3/8/4K3/8/8/8/8/7R w - - 0 1", "h1h8"),
]

# (FEN, độ sâu, số nước tới chiếu hết mong đợi) - đều đã kiểm chứng bằng cách
# chơi hết biến chính và xác nhận thế cờ cuối đúng là chiếu hết
MATE_SEARCH = [
    ("r1bq2r1/b4pk1/p1pp1p2/1p2pP2/1P2P1PB/3P4/1PPQ2P1/R3K2R w KQ - 0 1", 7, 2),
    ("2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", 7, 2),
    ("r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1", 9, 3),
]

fails = []


def check(name, cond, detail=""):
    print(("  OK  " if cond else "  SAI ") + name + ("  " + detail if detail else ""))
    if not cond:
        fails.append(name)


print("== 1. Sinh nước đi (perft)")
for fen, exp in PERFT_CASES:
    p = Position(fen)
    for d, e in enumerate(exp, 1):
        got = perft(p, d)
        check("perft d{} {}".format(d, fen[:28]), got == e, "{} / {}".format(got, e))

print("\n== 2. FEN vòng tròn & make/unmake giữ nguyên trạng thái")
for fen, _ in PERFT_CASES:
    p = Position(fen)
    check("fen roundtrip {}".format(fen[:28]), p.fen() == fen)
    before = (p.pieces[:], p.color_bb[:], p.all_bb, p.board[:], p.side,
              p.castling, p.ep, p.halfmove, p.fullmove, p.key)
    for m in p.legal_moves():
        u = p.make_move(m)
        p.unmake_move(m, u)
    after = (p.pieces[:], p.color_bb[:], p.all_bb, p.board[:], p.side,
             p.castling, p.ep, p.halfmove, p.fullmove, p.key)
    check("unmake khôi phục {}".format(fen[:28]), before == after)

print("\n== 3. Khoá Zobrist cập nhật tăng dần khớp tính lại từ đầu")
bad = 0
p = Position(START_FEN)


def walk(pos, depth):
    global bad
    if depth == 0:
        return
    for m in pos.generate():
        u = pos.make_move(m)
        if not pos.is_attacked(pos.king_sq(pos.side ^ 1), pos.side):
            if pos.key != pos.compute_key():
                bad += 1
            walk(pos, depth - 1)
        pos.unmake_move(m, u)


walk(p, 3)
check("zobrist nhất quán qua 3 tầng", bad == 0, "{} sai".format(bad))

print("\n== 4. Chiếu hết trong 1 nước -> 1000 điểm")
for fen, exp_move in MATE_IN_ONE:
    p = Position(fen)
    m = find_mate_in_one(p)
    check("mate1 {}".format(fen[:28]), m is not None and move_str(m) == exp_move,
          move_str(m) if m else "không tìm thấy")

print("\n== 5. Tìm kiếm phát hiện chiếu hết ép buộc")
for fen, depth, exp_n in MATE_SEARCH:
    p = Position(fen)
    r = Searcher().search(p, depth=depth)
    cp = r["score"] if p.side == WHITE else -r["score"]   # góc nhìn bên đi
    n = abs(scoring.mate_distance(r["score"]))
    check("mate#{} {}".format(exp_n, fen[:28]), n == exp_n and cp > 0,
          "tìm ra #{}".format(n) if n else "cp={}".format(r["score"]))

print("\n== 6. Thang điểm 0..1000")
r = Searcher().search(Position(START_FEN), depth=8)
scoring.calibrate(r["score"])
s0 = scoring.cp_to_score(r["score"])
check("thế xuất phát = 505", s0 == 505, str(s0))
check("cân bằng tuyệt đối < 505", scoring.cp_to_score(0) < 505, str(scoring.cp_to_score(0)))
center = scoring.score_to_cp(500)     # cp cho ra đúng 500 (hơi âm vì đã hiệu chỉnh)
check("phản đối xứng quanh mốc 500",
      all(abs(scoring.cp_to_score(round(center + d)) + scoring.cp_to_score(round(center - d))
              - 1000) <= 1 for d in (50, 150, 300, 600)),
      "±300 -> {} / {}".format(scoring.cp_to_score(round(center + 300)),
                               scoring.cp_to_score(round(center - 300))))
check("đơn điệu tăng theo cp",
     all(scoring.cp_to_score(c) <= scoring.cp_to_score(c + 25) for c in range(-2000, 2000, 25)))
check("chiếu hết 1 nước = 1000", scoring.cp_to_score(scoring.MATE_VALUE - 1) == 1000)
check("Đen chiếu hết 1 nước = 0", scoring.cp_to_score(-(scoring.MATE_VALUE - 1)) == 0)
check("chiếu hết 5 nước = 996", scoring.cp_to_score(scoring.MATE_VALUE - 9) == 996,
      str(scoring.cp_to_score(scoring.MATE_VALUE - 9)))
check("không chiếu hết luôn < 991", scoring.cp_to_score(100000) < scoring.MATE_FLOOR,
      str(scoring.cp_to_score(100000)))
check("nghịch đảo score->cp->score ổn định",
      all(abs(scoring.cp_to_score(round(scoring.score_to_cp(s))) - s) <= 1
          for s in range(50, 951, 50)))

print("\n== 7. Lượng giá đối xứng màu (thế cờ lật màu phải cho điểm đối nhau)")


def mirror_fen(fen):
    parts = fen.split()
    rows = parts[0].split("/")[::-1]
    rows = ["".join(c.lower() if c.isupper() else c.upper() for c in r) for r in rows]
    side = "b" if parts[1] == "w" else "w"
    cast = parts[2]
    if cast != "-":
        cast = "".join(sorted((c.lower() if c.isupper() else c.upper()) for c in cast))
    return " ".join(["/".join(rows), side, cast, "-", "0", "1"])


for fen in ["r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"]:
    a = evaluate.evaluate(Position(fen))
    b = evaluate.evaluate(Position(mirror_fen(fen)))
    check("đối xứng {}".format(fen[:28]), a == -b, "{} vs {}".format(a, -b))

print("\n== 8. Nhập thành")
g = Game()
for mv in "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5".split():
    g.push_uci(mv)
check("nhập thành gần có trong danh sách nước đi",
      "e1g1" in [move_str(m) for m in g.pos.legal_moves()])
g.push_uci("e1g1")
check("vua tới g1, xe tới f1", g.pos.board[6] == 5 and g.pos.board[5] == 3)
check("mất quyền nhập thành của Trắng sau khi nhập", not (g.pos.castling & 3))
g.pop()
check("undo khôi phục quyền nhập thành", g.pos.castling & 3 == 3)

check("không bị cản thì được nhập thành cả hai bên",
      {"e1g1", "e1c1"} <= {move_str(m) for m in
                           Position("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").legal_moves()})
check("có quân chắn thì cấm nhập thành",
      "e1g1" not in [move_str(m) for m in
                     Position("r3k2r/8/8/8/8/8/8/R3KB1R w KQkq - 0 1").legal_moves()])
check("bị chiếu thì cấm nhập thành",
      "e1g1" not in [move_str(m) for m in
                     Position("4r3/8/8/8/8/8/8/R3K2R w KQ - 0 1").legal_moves()])
check("ô vua đi qua bị kiểm soát thì cấm nhập thành",
      "e1g1" not in [move_str(m) for m in
                     Position("5r2/8/8/8/8/8/8/R3K2R w KQ - 0 1").legal_moves()])
check("xe bị ăn thì mất quyền nhập thành bên đó",
      not (Game("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").push_uci("a1a8").pos.castling & 8))

print("\n== 9. Bắt tốt qua đường")
g = Game("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
check("có nước bắt tốt qua đường", "e5d6" in [move_str(m) for m in g.pos.legal_moves()])
g.push_uci("e5d6")
check("tốt đối phương ở d5 bị nhấc khỏi bàn", g.pos.board[35] == -1 and g.pos.board[43] == 0)
g.pop()
check("undo trả lại tốt d5", g.pos.board[35] == 6)
g2 = Game("4k3/8/8/8/3p4/8/2P5/4K3 w - - 0 1").push_uci("c2c4")
check("đi tốt 2 ô đặt ô bắt qua đường", g2.pos.ep == 18, str(g2.pos.ep))
check("cơ hội bắt qua đường chỉ tồn tại 1 nước",
      Game("4k3/8/8/8/3p4/8/2P5/4K3 w - - 0 1").push_uci("c2c4").push_uci("e8d8").pos.ep == -1)
check("cấm bắt qua đường nếu làm lộ vua",
      "b4a3" not in [move_str(m) for m in
                     Position("8/8/8/8/k1pP3R/8/8/4K3 b - d3 0 1").legal_moves()])

print("\n== 10. Luật 50 nước")
g = Game("4k3/8/8/8/8/8/8/R3K3 w - - 98 60")
check("98 nửa nước: chưa hoà", g.draw_reason() is None, str(g.draw_reason()))
g.push_uci("a1a2")
check("99 nửa nước: chưa hoà", g.draw_reason() is None)
g.push_uci("e8e7")
check("100 nửa nước: HOÀ", g.draw_reason() is not None, str(g.draw_reason()))
g2 = Game("4k3/p7/8/8/8/8/8/R3K3 b - - 98 60").push_uci("a7a6")
check("đi tốt reset đồng hồ", g2.pos.halfmove == 0)
g3 = Game("4k3/8/8/8/8/8/8/R2rK3 w - - 98 60").push_uci("e1d1")
check("bắt quân reset đồng hồ", g3.pos.halfmove == 0)

print("\n== 11. Lặp 3 lần (thủ hoà)")
# không còn quyền nhập thành, nếu không nước xe đầu tiên sẽ đổi thế cờ
g = Game("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
check("ban đầu lặp 1 lần", g.repetitions() == 1)
for cycle in range(2):
    for mv in "a1b1 e8d8 b1a1 d8e8".split():
        g.push_uci(mv)
check("sau 2 chu kỳ: lặp 3 lần", g.repetitions() == 3, str(g.repetitions()))
check("mất quyền nhập thành khiến thế cờ KHÁC nhau",
      Game("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
      .push_uci("a1b1").push_uci("e8d8").push_uci("b1a1").push_uci("d8e8")
      .repetitions() == 1)
check("luật hoà kích hoạt", g.draw_reason() is not None, str(g.draw_reason()))
check("kết cục là 1/2-1/2", g.outcome()[0] == "1/2-1/2")

print("\n== 12. Pat và thiếu lực chiếu hết")
check("pat", Position("5k2/5P2/5K2/8/8/8/8/8 b - - 0 1").is_stalemate())
check("pat -> hoà", Game("5k2/5P2/5K2/8/8/8/8/8 b - - 0 1").outcome()[0] == "1/2-1/2")
check("vua+tượng vs vua = hoà", Position("8/8/4k3/8/8/3K1B2/8/8 w - - 0 1").is_insufficient_material())
check("vua+mã vs vua = hoà", Position("8/8/4k3/8/8/3K1N2/8/8 w - - 0 1").is_insufficient_material())
check("còn tốt thì chưa hoà",
      not Position("8/4p3/4k3/8/8/3K1B2/8/8 w - - 0 1").is_insufficient_material())
check("2 tượng thì chưa hoà",
      not Position("8/8/4k3/8/8/2BK1B2/8/8 w - - 0 1").is_insufficient_material())

print("\n== 13. Phong cấp")
check("4 lựa chọn phong cấp",
      sorted(move_str(m) for m in Position("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").legal_moves()
             if m and move_str(m).startswith("a7a8"))
      == ["a7a8b", "a7a8n", "a7a8q", "a7a8r"])
gp = Game("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").push_uci("a7a8q")
check("phong hậu đặt đúng quân", gp.pos.board[56] == 4)
gp.pop()
check("undo phong cấp trả lại tốt", gp.pos.board[48] == 0 and gp.pos.board[56] == -1)

print("\n== 14. Tìm kiếm bị ngắt giữa chừng vẫn trả nước HỢP LỆ")
# Ngoại lệ SearchAbort bay xuyên qua các khung negamax làm unmake_move bị bỏ qua.
# Nếu không dựng lại thế cờ, search trả về nước của thế cờ khác - và match.py
# đánh thẳng nước đó lên bàn, làm hỏng toàn bộ kết quả đấu.
import random as _rnd
_r = _rnd.Random(5)
_bad = _n = 0
for _t in range(40):
    _g = Game(START_FEN)
    for _ in range(_r.randint(0, 30)):
        _ms = _g.pos.legal_moves()
        if not _ms:
            break
        _g.push(_r.choice(_ms))
    if not _g.pos.legal_moves():
        continue
    _legal = set(move_str(m) for m in _g.pos.legal_moves())
    for _lim in ({"max_nodes": _r.randint(50, 2000)}, {"time_limit": 0.01}):
        _res = Searcher().search(_g.pos, depth=64, history_keys=_g.keys, **_lim)
        _n += 1
        if _res["move"] is None or move_str(_res["move"]) not in _legal:
            _bad += 1
        if _res["pv"] and move_str(_res["pv"][0]) not in _legal:
            _bad += 1
check("{} lượt ngắt giữa chừng đều trả nước hợp lệ".format(_n), _bad == 0,
      "{} nước sai".format(_bad))

print("\n== 15. SEE (Static Exchange Evaluation)")
from search import see as _see
from chess_core import parse_uci as _puci
# Giá trị tính tay theo thang SEE_VAL (T=100 M=320 Tg=330 X=500 H=900)
_SEE_CASES = [
    ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 100, "tốt ăn tốt trống"),
    ("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 0, "tốt ăn tốt, tốt ăn lại"),
    ("4k3/8/2p5/3p4/8/8/8/3RK3 w - - 0 1", "d1d5", -400, "xe ăn tốt, tốt ăn lại xe"),
    ("4k3/8/8/3p4/8/8/8/3RK3 w - - 0 1", "d1d5", 100, "xe ăn tốt trống"),
    ("4k3/8/2p5/3p4/2B5/8/8/4K3 w - - 0 1", "c4d5", -230, "tượng ăn tốt, tốt ăn lại"),
    ("4k3/8/5n2/3p4/2B5/8/8/3RK3 w - - 0 1", "c4d5", 90, "chuỗi 3 lượt, xe lộ ra sau tượng"),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6", 100, "bắt tốt qua đường"),
    ("4k3/3r4/8/3q4/8/8/8/3RK3 w - - 0 1", "d1d5", 400, "xe ăn hậu, xe ăn lại"),
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "e1e2", 0, "không phải nước ăn quân"),
]
for _fen, _uci, _want, _note in _SEE_CASES:
    _p = Position(_fen)
    _m = _puci(_p, _uci)
    check("see {} = {} ({})".format(_uci, _want, _note),
          _m is not None and _see(_p, _m) == _want,
          "được {}".format(_see(_p, _m) if _m else "nước không hợp lệ"))

# SEE chỉ được cắt tỉa trong quiescence khi KHÔNG bị chiếu - đòn hy sinh có
# chuỗi chiếu ép buộc phía sau không bao giờ được phép bị loại.
for _fen, _want in [("2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", 2),
                    ("r1bq2r1/b4pk1/p1pp1p2/1p2pP2/1P2P1PB/3P4/1PPQ2P1/R3K2R w KQ - 0 1", 2)]:
    _r = Searcher().search(Position(_fen), depth=9)
    check("hy sinh vẫn tìm ra chiếu hết #{}".format(_want),
          abs(scoring.mate_distance(_r["score"])) == _want,
          "#{}".format(abs(scoring.mate_distance(_r["score"]))))

print()
if fails:
    print("THẤT BẠI ({}): {}".format(len(fails), ", ".join(fails)))
    sys.exit(1)
print("TẤT CẢ TEST ĐỀU ĐẠT")

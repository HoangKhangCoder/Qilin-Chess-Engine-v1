"""Sinh dữ liệu huấn luyện với Stockfish gán nhãn.

CHỈ DÙNG KHI HUẤN LUYỆN. Engine cuối cùng (chess_core / search / nnue / main)
không import python-chess hay gọi Stockfish ở bất cứ đâu - kiểm tra bằng
    python check_purity.py

Mỗi dòng: FEN | cp Stockfish (góc nhìn Trắng) | kết quả ván
Giữ nguyên định dạng cũ nên train.py dùng được không cần sửa.

Hai nguồn nhãn ĐỘC LẬP nhau, đó là chủ ý:
  - cp   : Stockfish chấm thế cờ đó -> tín hiệu dày, gần như không nhiễu
  - result: kết quả ván chơi tiếp -> sự thật cuối cùng, chống việc mạng chỉ
            sao chép lại đúng hàm lượng giá của Stockfish

    python datagen_sf.py --games 5000 --depth 9 --workers 8
    python datagen_sf.py --games 5000 --pgn kho/carlsen.pgn      # mồi từ ván người
"""

import argparse
import gzip
import os
import random
import sys
import time

import chess
import chess.engine
import chess.pgn

SF = os.environ.get("STOCKFISH_PATH", "/opt/homebrew/bin/stockfish")
# Mỗi tiến trình Stockfish tốn ~245MB (mạng NNUE + hash). Trên máy chật RAM,
# ít worker mà chạy đủ CPU nhanh hơn nhiều worker bị swap bỏ đói.
HASH_MB = int(os.environ.get("SF_HASH_MB", 16))

BALANCE_CP = 200      # loại khai cuộc đã lệch hẳn
RESIGN_CP = 1500      # cp vượt ngưỡng này liên tục -> xử thắng luôn
RESIGN_PLIES = 6
MAX_PLIES = 300


def open_engine(threads=1, hash_mb=16):
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": threads, "Hash": hash_mb})
    return eng


# ------------------------------------------------------------- nguồn khai cuộc

def load_openings(pgn_path=None, epd_path=None, plies=(8, 16), limit=200000):
    """Đọc thế cờ mồi từ PGN (ván người) hoặc EPD. Trả về danh sách FEN."""
    out = []
    if epd_path:
        with open(epd_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(line.split(";")[0].strip())
                if len(out) >= limit:
                    return out
    if pgn_path:
        lo, hi = plies
        n_games = n_bad = 0
        with open(pgn_path, errors="replace") as f:
            while len(out) < limit:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                n_games += 1
                # PGN chép tay hay có nước sai; python-chess dừng ở nước lỗi và
                # ghi vào game.errors. Ta vẫn dùng phần đọc được, nhưng phải
                # BÁO ra thay vì nuốt im lặng - nếu không sẽ tưởng đã nạp đủ.
                if game.errors:
                    n_bad += 1
                    print("  ! {} - {}: {}".format(
                        game.headers.get("White", "?"), game.headers.get("Black", "?"),
                        game.errors[0]), file=sys.stderr)
                board = game.board()
                n = 0
                for mv in game.mainline_moves():
                    board.push(mv)
                    n += 1
                    # lấy thế cờ ở đoạn thoát sách khai cuộc
                    if lo <= n <= hi:
                        out.append(board.fen())
                    if n > hi:
                        break
        print("  PGN: {} ván, {} ván có nước sai".format(n_games, n_bad), file=sys.stderr)
    return out


def random_balanced_opening(eng, rng, plies, depth, tries=12):
    """Khai cuộc ngẫu nhiên, loại thế cờ đã lệch hẳn."""
    for _ in range(tries):
        b = chess.Board()
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if b.is_game_over():
            continue
        sc = eng.analyse(b, chess.engine.Limit(depth=max(4, depth - 4)))["score"].white()
        cp = sc.score(mate_score=100000)
        if abs(cp) <= BALANCE_CP:
            return b
    return None


# --------------------------------------------------------------------- ván cờ

def play_game(eng, rng, board, depth, multipv, rand_prob):
    """Stockfish tự chơi. Trả về [(fen, cp_trắng)] và kết quả ván."""
    samples = []
    result = None
    streak = 0

    for _ in range(MAX_PLIES):
        if board.is_checkmate():
            result = 0.0 if board.turn == chess.WHITE else 1.0
            break
        if board.is_stalemate() or board.is_insufficient_material() \
                or board.is_repetition(3) or board.halfmove_clock >= 100:
            result = 0.5
            break

        infos = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        if not isinstance(infos, list):
            infos = [infos]
        top = infos[0]
        sc = top["score"].white()
        best = top["pv"][0]

        # lọc nhiễu: thế đang bị chiếu, hoặc nước tốt nhất là ăn quân/phong cấp
        noisy = (board.is_check() or board.is_capture(best)
                 or best.promotion is not None)
        if not noisy and not sc.is_mate():
            samples.append((board.fen(), sc.score()))

        cp = sc.score(mate_score=100000)
        if abs(cp) > RESIGN_CP:
            streak += 1
            if streak >= RESIGN_PLIES:
                result = 1.0 if cp > 0 else 0.0
                break
        else:
            streak = 0

        # đa dạng hoá: thỉnh thoảng chọn nước tốt thứ 2/3 thay vì nước tốt nhất.
        # Khác hẳn đi bừa - nước top-3 hiếm khi lật ngược kết quả nên nhãn
        # `result` của các thế cờ đứng trước vẫn giữ được ý nghĩa.
        mv = best
        if len(infos) > 1 and rng.random() < rand_prob:
            alt = rng.choice(infos[1:])
            if alt.get("pv"):
                mv = alt["pv"][0]
        board.push(mv)

    if result is None:
        result = 0.5
    return samples, result


def _worker(args):
    (seed, n_games, depth, multipv, rand_prob, op_lo, op_hi,
     openings, out_path) = args
    ppid = os.getppid()
    rng = random.Random(seed)
    eng = open_engine()
    written = 0
    try:
        # gzip cho phép nối nhiều khối nên vẫn ghi tiếp được vào .gz
        op = gzip.open(out_path, "at") if out_path.endswith(".gz") else open(out_path, "a")
        with op as f:
            for _ in range(n_games):
                if os.getppid() != ppid:      # cha đã chết -> đừng thành mồ côi
                    break
                if openings:
                    board = chess.Board(rng.choice(openings))
                    if board.is_game_over():
                        continue
                else:
                    board = random_balanced_opening(
                        eng, rng, rng.randint(op_lo, op_hi), depth)
                    if board is None:
                        continue
                samples, result = play_game(eng, rng, board, depth, multipv, rand_prob)
                for fen, cp in samples:
                    f.write("{}|{}|{}\n".format(fen, cp, result))
                written += len(samples)
                f.flush()
    finally:
        eng.quit()
    return written


def main():
    ap = argparse.ArgumentParser(description="Sinh dữ liệu huấn luyện bằng Stockfish")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=9, help="độ sâu Stockfish mỗi nước")
    ap.add_argument("--out", default="data/sf.txt")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--multipv", type=int, default=3)
    ap.add_argument("--rand-prob", type=float, default=0.15,
                    help="xác suất chọn nước tốt thứ 2/3 để đa dạng hoá")
    ap.add_argument("--opening-min", type=int, default=4)
    ap.add_argument("--opening-max", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=150,
                    help="số ván mỗi worker chơi trước khi mở lại Stockfish. "
                         "Để nhỏ thì Stockfish phải nạp lại mạng NNUE ~50MB "
                         "liên tục - đo được là mất khoảng 30%% thông lượng.")
    ap.add_argument("--pgn", default=None, help="mồi khai cuộc từ file PGN ván người")
    ap.add_argument("--epd", default=None, help="mồi khai cuộc từ file EPD/FEN")
    args = ap.parse_args()

    if not os.path.exists(SF):
        sys.exit("Không thấy Stockfish tại {}. Đặt biến STOCKFISH_PATH.".format(SF))

    openings = load_openings(args.pgn, args.epd)
    if openings:
        print("Nạp {} thế cờ mồi từ {}".format(
            len(openings), args.pgn or args.epd), file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    chunk = max(1, min(args.chunk, args.games // args.workers or 1))
    tasks, remaining, s = [], args.games, args.seed
    while remaining > 0:
        n = min(chunk, remaining)
        tasks.append((s, n, args.depth, args.multipv, args.rand_prob,
                      args.opening_min, args.opening_max, openings, args.out))
        s += 1
        remaining -= n

    total = 0
    if args.workers > 1:
        import multiprocessing as mp
        pool = mp.Pool(args.workers)
        try:
            done = 0
            for w in pool.imap_unordered(_worker, tasks):
                total += w
                done += 1
                el = time.time() - t0
                sys.stderr.write("\r{}/{} lô | {:,} thế cờ | {:.0f}s | {:,.0f} thế cờ/giờ"
                                 .format(done, len(tasks), total, el, total / max(el, 1) * 3600))
        except KeyboardInterrupt:
            sys.stderr.write("\nDừng theo yêu cầu.\n")
        finally:
            pool.terminate()
            pool.join()
    else:
        for t in tasks:
            total += _worker(t)
    sys.stderr.write("\nĐã ghi {:,} thế cờ vào {}\n".format(total, args.out))


if __name__ == "__main__":
    main()

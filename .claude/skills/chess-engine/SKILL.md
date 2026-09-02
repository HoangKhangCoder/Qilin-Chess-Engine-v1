---
name: chess-engine
description: Quy trình làm việc với engine cờ vua Kỳ Lân (chấm điểm 0-1000) trong repo này - sinh dữ liệu bằng Stockfish, huấn luyện NNUE, đo lường xem mạng có thật sự khá hơn, đấu thang Elo với Stockfish thật, và giữ ranh giới engine sạch thư viện ngoài. Dùng khi cần train lại mạng, sinh thêm dữ liệu, đánh giá chất lượng hàm lượng giá, sửa luật cờ trong chess_core, hoặc chạy đấu đối kháng.
---

# Quy trình engine cờ vua Kỳ Lân

**[English below ↓](#english)**

## Trước khi sửa bất cứ thứ gì

```bash
python3 test_engine.py && python3 check_purity.py
```

Cả hai phải đạt. `test_engine.py` (86 test) mất ~3 phút (perft chiếm phần lớn).

## Ranh giới không được phá

Engine chạy thật (`chess_core` `search` `evaluate` `scoring` `nnue` `main`
`server` `test_engine`) **không được import python-chess hay gọi Stockfish**.
Chỉ `datagen_sf.py` `make_book.py` `train.py` `match.py` `play_stockfish.py`
`compare_evals.py` `datagen.py` được phép.

`check_purity.py` thực thi điều này bằng cách quét AST (bỏ qua docstring/chú
thích để tránh dương tính giả) rồi nạp engine với module `chess` bị chặn ở
`sys.meta_path` — phụ thuộc ẩn cũng bị lộ. Chạy nó sau mọi thay đổi đụng file
engine.

## Sửa luật cờ trong chess_core.py

perft **không đủ** để bắt lỗi — nó chỉ đếm nước đi, không kiểm tra đồng hồ 50
nước, ô ep, hay khoá Zobrist. Luôn đối chiếu với python-chess:

```python
import chess
from chess_core import Game, move_str
# chơi vài trăm ván ngẫu nhiên, mỗi thế cờ so:
#   sorted(move_str(m) for m in g.pos.legal_moves()) == sorted(m.uci() for m in b.legal_moves)
#   g.pos.fen() == b.fen()
#   (in_check, is_checkmate, is_stalemate, repetitions>=3, halfmove>=100, insufficient)
```

Cách này đã bắt được 2 bug mà perft bỏ lọt: đồng hồ 50 nước không reset sau
phong cấp, và ô ep ghi vô điều kiện làm hỏng phát hiện lặp 3 lần.

Nếu sửa `search.py`: cẩn thận đường `SearchAbort` — ngoại lệ hết giờ/hết nút
bay xuyên qua các khung `negamax` đệ quy có thể bỏ qua `unmake_move` và làm
`search()` trả về nước bất hợp lệ. Đã vá bằng khôi phục `work.copy()` + lưới
an toàn cuối, có test hồi quy riêng (nhóm 14 trong `test_engine.py`). Bất kỳ
thay đổi nào ở luồng thoát sớm của `negamax`/`quiesce` đều phải chạy lại test
đó.

## Sinh thêm dữ liệu

```bash
# sách khai cuộc (một lần, ~4.5 thế cờ/giây với 7 worker)
python3 make_book.py --count 8000 --workers 7

# sinh dữ liệu - CHUNK LỚN, nếu không Stockfish nạp lại mạng 50MB liên tục
python3 datagen_sf.py --games 400000 --depth 9 --workers 7 --chunk 150 \
  --epd data/book.epd --out data/sf.txt.gz
```

Dữ liệu **cộng dồn** vào file, chạy lại bao nhiêu lần cũng được. Worker flush
theo từng ván nên dừng bằng `pkill -f '[d]atagen_sf.py'` không mất gì (mẫu
ngoặc vuông để không tự giết chính script gọi nó).

`.txt.gz` đọc/ghi được trong suốt và nén còn ~20% dung lượng — luôn dùng nó
thay vì `.txt` thô.

Thông lượng tham chiếu: ~570k thế cờ/giờ với 7 worker depth 9 trên máy 8 nhân,
nhưng **áp lực bộ nhớ hệ thống** (ví dụ Chrome chiếm nhiều RAM) có thể đẩy
swap phình lên và làm tụt xuống còn 1/3 tốc độ đó — kiểm tra `sysctl -n
vm.swapusage` nếu thông lượng bất thường thấp.

Kiểm tra dữ liệu sau khi sinh: dòng hỏng phải bằng 0, trùng lặp dưới 1%, tỉ lệ
mẫu có |cp|<100 nên khoảng 40% (vùng cân bằng là nơi độ chính xác quyết định).

## Huấn luyện

```bash
NNUE_KING_BUCKETS=64 python3 -u train.py --data data/sf.txt.gz \
  --epochs 40 --batch 8192 --device cpu --lambda 1.0 --out weights/x.npz
```

- `-u` bắt buộc khi ghi log ra file, nếu không tưởng treo
- `--device cpu` là đúng: đo được CPU nhanh gấp đôi MPS ở kích thước mạng này
- `NNUE_KING_BUCKETS` phải giống hệt lúc chạy engine sau này (`serve.sh` tự
  đọc từ `.npz` nên không cần nhớ tay)
- `--seed` (mặc định 1234) phải **giữ nguyên** khi so hai kiến trúc khác nhau
  trên cùng dữ liệu, nếu không chênh lệch đo được có thể chỉ do chia tập
  validation khác nhau
- Cache đặc trưng là **thư mục memmap** `<data>.feat_kb<N>/`, không phải một
  file `.npz` — lần train đầu dựng cache (vài phút với hàng triệu thế cờ),
  các lần sau nạp trong vài giây

Đọc train/val gap để biết cần gì:

| Dấu hiệu | Nghĩa | Việc cần làm |
|---|---|---|
| val >> train | overfit | thêm dữ liệu |
| val ≈ train, cả hai cao | underfit | tăng KING_BUCKETS hoặc HIDDEN |
| val ngừng giảm sớm | hết tín hiệu | dữ liệu đa dạng hơn, không phải nhiều hơn |

Đã đo thực tế: cùng 7,91 triệu thế cờ, tăng `KING_BUCKETS` từ 4 lên 64 giảm
val loss 25% — nút thắt từng nằm ở *dung lượng mạng*, không phải *lượng dữ
liệu*. Nhưng mạng KB=64 hiện có khoảng cách train/val +235% (overfit nặng),
nên bây giờ lại cần thêm dữ liệu để lấp đầy dung lượng lớn hơn đó.

## Đo xem mạng có THẬT SỰ khá hơn

Val loss thấp không có nghĩa cờ đánh hay hơn. Thứ tự tin cậy:

```bash
# 1. Đấu thật với Stockfish bị giới hạn Elo, khai cuộc bắt cặp - đáng tin nhất
python3 play_stockfish.py --games 100 --ladder 1600,1800,2000,2200 \
  --time 2.0 --nnue weights/x.npz --book data/book.epd

# 2. Đấu đối kháng hai hàm lượng giá ở CÙNG số nút hoặc CÙNG thời gian
python3 match.py --nnue weights/x.npz --games 100 --time 2.0

# 3. Tương quan với Stockfish trên tập giữ lại (phải lọc trùng với tập train!)
python3 datagen_sf.py --games 40 --depth 12 --seed 999999 --out data/_holdout.txt
# rồi loại mọi FEN đã có trong data/sf.txt.gz trước khi đo
```

**Bẫy đã dính (hai lần):**

1. Train mạng dự đoán hỗn hợp `λ·sigmoid(cp) + (1−λ)·kết_quả` rồi chấm nó
   bằng cp thuần. Khi so với nhãn cp, phải train `--lambda 1.0`, nếu không
   đang phạt mạng vì đúng thứ nó được dạy. Không so val loss giữa hai giá trị
   `--lambda` khác nhau — mục tiêu học khác nhau thì loss không cùng thang.
2. Mạng khớp cp Stockfish tốt hơn hàm thủ công 16% mà đấu thật ở cùng số nút
   thì **không hơn** — độ chính xác tuyệt đối không tự động thành sức cờ.
   `match.py`/`play_stockfish.py` là trọng tài cuối cùng, không phải loss.

## Bất biến của thang điểm

Sau mọi thay đổi đụng `scoring.py` hay hàm lượng giá, kiểm lại:

```bash
python3 main.py --recalibrate --depth 8                    # phải ra 505
python3 main.py "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"        # phải ra 1000
python3 main.py "5k2/5P2/5K2/8/8/8/8/8 b - - 0 1"          # phải ra 500 (pat)
```

Mốc 505 đến từ `scoring.calibrate()`, không phải cộng thêm 5. Đổi hàm lượng giá
thì phải `--recalibrate` hoặc xoá `.calib.json`.

## Đấu Stockfish thật / thang Elo

```bash
python3 play_stockfish.py --games 100 --ladder 1600,1800,2000,2200 \
  --time 2.0 --nnue weights/x.npz --book data/book.epd --out games/ladder.pgn
```

- Khai cuộc **bắt cặp**: mỗi thế cờ trong sách đấu đúng 2 lần, đổi màu - số
  ván mỗi màu bằng nhau tuyệt đối và giảm nhiễu do khai cuộc thuận lợi
- `--time` áp dụng cho **cả hai bên** - so sánh công bằng về tính toán
- Đếm ván đã ghi trong PGN để **chạy tiếp đúng chỗ dừng** nếu tiến trình bị
  giết giữa chừng - luôn bọc trong vòng lặp giám sát tự khởi động lại khi
  chạy hàng chục giờ
- Kết quả hiện tại (`weights/cap_kb64_h256.npz`): ~2000-2070 Elo, xem
  `CLAUDE.md` mục "Kết quả hiện tại"

## Giới hạn cần nói thẳng

Engine này chạy ~16-80k nút/giây (Python thuần). Stockfish chạy 10-50
**triệu**. Chênh ~600 lần đó không thể bù bằng hàm lượng giá tốt hơn. Thêm
nữa, dữ liệu huấn luyện do chính Stockfish gán nhãn, nên trần lý thuyết là
"xấp xỉ Stockfish, kém chính xác hơn" - trừ khi chuyển sang vòng lặp tự chơi
bằng chính mạng đã huấn luyện (`datagen.py --nnue` đã sẵn sàng).

Mục tiêu đo được và có ý nghĩa: đấu với Stockfish **bị giới hạn Elo**
(`play_stockfish.py --ladder`) để biết engine mạnh cỡ nào, rồi đẩy con số đó
lên. Đừng đặt mục tiêu thắng Stockfish đầy đủ - đã đo được 0/10 ở sức đầy đủ.

Hai hướng còn chưa làm, có thể cho nhiều Elo mà không cần dữ liệu mới: SEE
(Static Exchange Evaluation, chỉ cắt tỉa trong quiescence khi không bị chiếu
- để không loại nhầm đòn hy sinh có chuỗi chiếu ép buộc phía sau) và
accumulator NNUE tăng dần (hiện tính lại toàn bộ mỗi lần thay vì cập nhật
O(1) khi quân di chuyển).

---

<a id="english"></a>

# Working with the Kỳ Lân chess engine

which Claude Code reads for its name/description frontmatter. This file is
plain documentation for English-speaking readers browsing the repo.)*

## Before touching anything

```bash
python3 test_engine.py && python3 check_purity.py
```

Both must pass. `test_engine.py` (86 tests) takes ~3 minutes (perft is most
of it).

## The boundary that must not break

The real engine (`chess_core` `search` `evaluate` `scoring` `nnue` `main`
`server` `test_engine`) **must never import python-chess or call
Stockfish**. Only `datagen_sf.py` `make_book.py` `train.py` `match.py`
`play_stockfish.py` `compare_evals.py` `datagen.py` are allowed to.

`check_purity.py` enforces this by walking the AST (skipping
docstrings/comments to avoid false positives) then loading the engine with
`chess` blocked at `sys.meta_path` — hidden dependencies get exposed too.
Run it after any change touching an engine file.

## Fixing chess rules in chess_core.py

Perft is **not enough** to catch bugs — it only counts moves, it doesn't
check the 50-move clock, the ep square, or the Zobrist key. Always
cross-check against python-chess:

```python
import chess
from chess_core import Game, move_str
# play a few hundred random games, at each position compare:
#   sorted(move_str(m) for m in g.pos.legal_moves()) == sorted(m.uci() for m in b.legal_moves)
#   g.pos.fen() == b.fen()
#   (in_check, is_checkmate, is_stalemate, repetitions>=3, halfmove>=100, insufficient)
```

This caught 2 bugs perft missed: the 50-move clock not resetting after
promotion, and an unconditionally-recorded ep square breaking
threefold-repetition detection.

If you touch `search.py`: watch the `SearchAbort` path — a timeout/node-limit
exception propagating through recursive `negamax` frames can skip
`unmake_move` and make `search()` return an illegal move. Fixed via a
`work.copy()` restore plus a final safety net, with its own regression test
(group 14 in `test_engine.py`). Any change to `negamax`/`quiesce`'s
early-exit path must rerun that test.

## Generating more data

```bash
# opening book (one-time, ~4.5 positions/sec with 7 workers)
python3 make_book.py --count 8000 --workers 7

# data generation — LARGE chunk, otherwise Stockfish keeps reloading its 50MB net
python3 datagen_sf.py --games 400000 --depth 9 --workers 7 --chunk 150 \
  --epd data/book.epd --out data/sf.txt.gz
```

Data **accumulates** into the file — rerun as many times as you like.
Workers flush per-game, so stopping with `pkill -f '[d]atagen_sf.py'` loses
nothing (bracket pattern so the command doesn't kill itself).

`.txt.gz` is read/written transparently and compresses to ~20% the size —
always prefer it over raw `.txt`.

Reference throughput: ~570k positions/hour with 7 workers at depth 9 on an
8-core machine, but **system memory pressure** (e.g. Chrome hogging RAM) can
inflate swap and drop that to a third — check `sysctl -n vm.swapusage` if
throughput looks abnormally low.

Sanity-check data after generating: 0 corrupt lines, under 1% duplicates,
and roughly 40% of samples with |cp|<100 (the balanced zone is where
accuracy actually decides games).

## Training

```bash
NNUE_KING_BUCKETS=64 python3 -u train.py --data data/sf.txt.gz \
  --epochs 40 --batch 8192 --device cpu --lambda 1.0 --out weights/x.npz
```

- `-u` is mandatory when logging to a file, otherwise it looks hung
- `--device cpu` is correct: measured CPU running twice as fast as MPS at
  this network size
- `NNUE_KING_BUCKETS` must exactly match whatever runs the engine afterward
  (`serve.sh` reads it straight from the `.npz`, so you never set it by
  hand)
- `--seed` (default 1234) must **stay fixed** when comparing two
  architectures on the same data, otherwise a measured gap could just be a
  different validation split
- The feature cache is a **memmap directory** `<data>.feat_kb<N>/`, not a
  single `.npz` file — the first run builds it (a few minutes for millions
  of positions), later runs load in seconds

Read the train/val gap to know what's needed:

| Signal | Meaning | What to do |
|---|---|---|
| val >> train | overfitting | add more data |
| val ≈ train, both high | underfitting | raise KING_BUCKETS or HIDDEN |
| val plateaus early | signal exhausted | more diverse data, not more of it |

Measured in practice: on the same 7.91M positions, raising `KING_BUCKETS`
from 4 to 64 cut val loss by 25% — the bottleneck used to be *network
capacity*, not *data volume*. But the KB=64 network now has a train/val gap
of +235% (heavy overfitting), so more data is needed again to fill that
larger capacity.

## Measuring whether the network is ACTUALLY better

Low val loss doesn't mean stronger play. In order of trust:

```bash
# 1. Real games vs Elo-limited Stockfish, paired openings — most trustworthy
python3 play_stockfish.py --games 100 --ladder 1600,1800,2000,2200 \
  --time 2.0 --nnue weights/x.npz --book data/book.epd

# 2. Head-to-head between two evaluators at the SAME node or SAME time budget
python3 match.py --nnue weights/x.npz --games 100 --time 2.0

# 3. Correlation with Stockfish on a held-out set (must dedupe vs training set!)
python3 datagen_sf.py --games 40 --depth 12 --seed 999999 --out data/_holdout.txt
# then remove every FEN already in data/sf.txt.gz before measuring
```

**Traps hit twice:**

1. Trained a network on a mixed target `λ·sigmoid(cp) + (1−λ)·result`, then
   scored it against pure cp. When comparing to cp labels, train with
   `--lambda 1.0`, otherwise you're penalizing the network for exactly what
   it was taught. Never compare val loss across two different `--lambda`
   values — different training objectives aren't on the same scale.
2. A network that matches Stockfish's cp 16% better than the handcrafted
   evaluator can still be **no better** in a real same-node match — raw
   accuracy doesn't automatically become playing strength. `match.py` /
   `play_stockfish.py` are the final judges, not loss.

## Scoring-scale invariants

After any change touching `scoring.py` or the evaluation function, verify:

```bash
python3 main.py --recalibrate --depth 8                    # must give 505
python3 main.py "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"        # must give 1000
python3 main.py "5k2/5P2/5K2/8/8/8/8/8 b - - 0 1"          # must give 500 (stalemate)
```

The 505 anchor comes from `scoring.calibrate()`, not an added constant.
Changing the evaluator requires `--recalibrate` or deleting `.calib.json`.

## Playing real Stockfish / the Elo ladder

```bash
python3 play_stockfish.py --games 100 --ladder 1600,1800,2000,2200 \
  --time 2.0 --nnue weights/x.npz --book data/book.epd --out games/ladder.pgn
```

- Openings are **paired**: each book position is played exactly twice with
  colors swapped — perfectly equal games per color, and much lower variance
  from lucky/unlucky opening draws
- `--time` applies to **both sides** — fair on compute
- Counts games already written to the PGN so it can **resume exactly where
  it stopped** if the process gets killed mid-run — always wrap long
  (many-hour) runs in a self-restarting supervisor loop
- Current result (`weights/cap_kb64_h256.npz`): roughly 2000–2070 Elo, see
  the "Current results" section in `CLAUDE.md`

## Limitations worth stating plainly

This engine runs ~16–80k nodes/sec (pure Python). Stockfish runs 10–50
**million**. That ~600× gap can't be closed by a better evaluation function
alone. On top of that, the training data is labeled by Stockfish itself, so
the theoretical ceiling is "approximates Stockfish, less accurately" —
unless you switch to a self-play loop using the trained network itself
(`datagen.py --nnue` is already wired up for this).

A measurable, meaningful goal: play against Elo-limited Stockfish
(`play_stockfish.py --ladder`) to find out how strong the engine actually
is, then push that number up. Don't aim for beating full-strength
Stockfish — measured 0/10 there already.

Two directions not yet done that could add Elo without any new data: SEE
(Static Exchange Evaluation, pruning only inside quiescence search and only
when not in check — so a sacrifice followed by a forcing check sequence
never gets wrongly discarded) and an incremental NNUE accumulator (currently
recomputed from scratch every call instead of updated in O(1) as pieces
move).

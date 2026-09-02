# Kỳ Lân v1 — chấm điểm thế cờ vua trên thang 0–1000

Cho một thế cờ bất kỳ, trả lời: **Trắng đang được bao nhiêu điểm trên 1000?**

```bash
./serve.sh                                                   # bàn phân tích web
python3 main.py "<FEN>" --nnue weights/cap_kb64_h256.npz --depth 8
```

**[English below ↓](#english)**

## Ràng buộc bất di bất dịch

**Engine chạy thật KHÔNG được dùng thư viện cờ ngoài.** Không python-chess, không
Stockfish, không thư viện cờ nào. Chỉ pipeline huấn luyện mới được dùng.

```
SẠCH (engine)    chess_core.py  search.py  evaluate.py  scoring.py  nnue.py
                 main.py  server.py  test_engine.py
ĐƯỢC DÙNG (train) datagen_sf.py  make_book.py  train.py  match.py
                 play_stockfish.py  compare_evals.py  datagen.py
```

Ranh giới này được **thực thi bằng code**, không phải quy ước:

```bash
python3 check_purity.py   # quét import + nạp engine trong môi trường đã chặn module `chess`
```

Chạy nó sau mọi thay đổi đụng tới các file engine. Nó dùng AST để quét chuỗi
**thực sự chạy trong code**, bỏ qua docstring/chú thích (tránh dương tính giả
khi file chỉ *nhắc tên* Stockfish trong tài liệu) — rồi thực sự khởi động
engine với `chess` bị chặn ở `sys.meta_path`, nên phụ thuộc ẩn cũng bị lộ.

## Thang điểm

Ý nghĩa: **S = 1000 × kỳ vọng kết quả ván cờ của Trắng.**

| Điểm | Nghĩa |
|---|---|
| 1000 | Trắng chiếu hết ngay nước này |
| 991–999 | Trắng có chiếu hết ép buộc (991 = xa nhất) |
| 505 | **Thế xuất phát** — 5 điểm là lợi thế tiên thủ |
| 500 | Cân bằng tuyệt đối, hoặc đã hoà theo luật |
| 0 | Đen chiếu hết ngay nước này |

Mốc 505 **không phải hằng số cộng thêm**. `scoring.calibrate(cp_thế_xuất_phát)`
đặt một bù trừ bên trong hàm sigmoid sao cho hàm lượng giá đang dùng, khi chấm
thế xuất phát, ra đúng 505. Nhờ vậy đổi từ hàm thủ công sang NNUE thì mốc vẫn
giữ. Kết quả hiệu chỉnh cache ở `.calib.json`; xoá file đó hoặc dùng
`--recalibrate` khi đổi hàm lượng giá.

Dải không-chiếu-hết bị kẹp trong [10, 990] nên không bao giờ đụng dải chiếu hết —
chiếu hết ép buộc luôn được xếp trên mọi ưu thế vật chất.

## Kiểm thử

```bash
python3 test_engine.py     # 86 test: perft, FEN, Zobrist, luật hoà, thang điểm, search
python3 check_purity.py    # ranh giới thư viện ngoài
```

`test_engine.py` phải đạt 100% trước mọi commit. Nó phủ 14 nhóm: perft trên 6 vị
trí chuẩn, make/unmake khôi phục trạng thái, Zobrist tăng dần khớp tính lại,
chiếu hết 1 nước, chiếu hết ép buộc, thang điểm, đối xứng màu, nhập thành, bắt
tốt qua đường, luật 50 nước, lặp 3 lần, pat/thiếu lực, phong cấp, và **tìm kiếm
bị ngắt giữa chừng vẫn trả về nước hợp lệ**.

Đối chiếu độc lập với python-chess (55k+ thế cờ, khớp hoàn toàn) — chạy lại
mỗi khi sửa `chess_core.py`.

## Bẫy đã gặp — đừng lặp lại

**SearchAbort bỏ qua unmake_move.** Khi hết giờ/hết nút giữa lượt tìm kiếm,
`SearchAbort` bay xuyên qua các khung `negamax` đệ quy, khiến `unmake_move`
bị bỏ qua — bàn cờ nội bộ (`work`) kẹt ở một thế cờ đang thử dở, thậm chí sai
lượt. `extract_pv` khi đó đọc nhầm thế cờ và `search()` có thể trả về nước
**bất hợp lệ**. Vá bằng cách khôi phục `work = pos.copy()` khi bắt được
`SearchAbort`, cộng lưới an toàn cuối: nếu nước trả về không nằm trong danh
sách hợp lệ thì thay bằng `legal[0]`. Bug này từng làm hỏng âm thầm mọi kết
quả `match.py`/`play_stockfish.py` vì chúng đánh thẳng nước trả về lên bàn cờ
mà không kiểm tra lại — giờ cả hai đều tự kiểm tra hợp lệ trước khi `push()`.

**Đồng hồ 50 nước và phong cấp.** Trong `make_move`, biến `piece` bị gán lại
thành quân mới khi phong cấp. Phải dùng `moved_type` đã lưu trước đó, nếu không
nước phong cấp không ăn quân sẽ bị tính là nước im lặng. perft không bắt được
lỗi này vì perft chỉ đếm nước đi.

**Ô bắt tốt qua đường.** Chỉ ghi khi nước bắt đó *thực sự hợp lệ*
(`_has_legal_ep`). Ghi vô điều kiện thì ô ep "ma" lọt vào khoá Zobrist, khiến hai
thế cờ giống hệt nhau có khoá khác nhau và hỏng phát hiện lặp 3 lần. `set_fen`
cũng phải lọc, vì nhiều FEN bên ngoài ghi vô điều kiện.

**`NNUE_KING_BUCKETS` phải khớp giữa train và chạy.** `nnue.py` kiểm tra và báo
lỗi rõ nếu lệch — đừng gỡ kiểm tra đó, sai lệch âm thầm sẽ cho ra số vô nghĩa.
`serve.sh` tự đọc giá trị này từ file `.npz` nên không cần đặt tay.

**Đệm stdout.** Khi ghi log ra file, `print` không tự flush. Chạy python bằng
`-u`, nếu không sẽ tưởng tiến trình treo trong khi nó chạy bình thường.

**CPU nhanh hơn MPS cho mạng này.** Đo được 28ms/batch trên CPU so với 58ms trên
MPS — mạng quá nhỏ nên chi phí khởi chạy kernel GPU lấn át tính toán.
`train.py --device` mặc định `cpu` là có chủ ý.

**Kích thước lô sinh dữ liệu.** `datagen_sf.py --chunk` nhỏ khiến Stockfish nạp
lại mạng NNUE ~50MB liên tục. Tăng từ 8 lên 150 cho thông lượng gấp 2,3 lần.
Dữ liệu vẫn ghi theo từng ván nên chunk lớn không mất gì ngoài độ mượt của dòng
tiến độ.

**`pkill -f <tên_script>` tự giết chính script đang gọi nó.** Dòng lệnh của
shell chứa chuỗi tên script, nên `pkill -f run_x.sh` bên trong `run_x.sh` giết
luôn tiến trình cha trước khi nó kịp ghi log. Dùng mẫu ngoặc vuông
`pkill -f '[r]un_x.sh'` để loại trừ chính nó. macOS cũng không có lệnh
`setsid`; tách tiến trình nền bằng `( nohup ... & )` trong subshell.

## Pipeline huấn luyện

```bash
python3 make_book.py --count 8000 --workers 7               # sách khai cuộc MultiPV
python3 datagen_sf.py --games 400000 --epd data/book.epd \
    --out data/sf.txt.gz                                     # Stockfish gán nhãn
NNUE_KING_BUCKETS=64 python3 -u train.py --data data/sf.txt.gz --lambda 1.0
python3 match.py --nnue weights/x.npz --games 100            # đấu đối kháng
python3 play_stockfish.py --games 100 --ladder 1600,1800,2000,2200 \
    --time 2.0 --nnue weights/x.npz                          # thang Elo với Stockfish thật
```

Mỗi dòng dữ liệu: `FEN | cp Stockfish (góc nhìn Trắng) | kết quả ván`.

`train.py` và `datagen_sf.py` đọc/ghi được cả `.txt` lẫn `.txt.gz` trong suốt.
Dữ liệu nén còn ~20% (456 MB → 95 MB) và giải nén nhanh hơn nhiều so với thời
gian phân tích FEN, nên luôn dùng `.gz`. gzip cho phép nối khối nên vẫn ghi
tiếp được vào file đã nén.

Hai nguồn nhãn **cố ý độc lập**: `cp` cho tín hiệu dày ít nhiễu, `result` cho sự
thật cuối cùng. Nếu lấy cả hai từ Stockfish thì mạng chỉ sao chép Stockfish và
không còn gì để vượt. `--lambda` điều khiển tỉ lệ trộn (1.0 = chỉ bắt chước cp).

`train.py` cache đặc trưng ra **thư mục memmap** `<data>.feat_kb<N>/` (không
phải một file `.npz` duy nhất) — chỉ số lưu `int16` khi đủ chỗ, ghi thẳng ra
đĩa bằng hai lượt (đếm rồi ghi) để không giữ hết trong RAM. Cache gắn với
`KING_BUCKETS`; lần train sau nạp trong vài giây thay vì vài phút.

`--seed` (mặc định 1234) cố định thứ tự chia tập train/val — **bắt buộc phải
giữ cùng seed** khi so hai kiến trúc khác nhau, nếu không chênh lệch đo được
có thể chỉ do chia tập khác nhau chứ không phải do kiến trúc.

## Đo lường: cẩn thận với chính phép đo

Loss giảm **không** đảm bảo cờ đánh hay hơn. Đã dính hai lần:

1. Train mạng dự đoán hỗn hợp `0,7×sigmoid(cp) + 0,3×kết_quả` rồi lại chấm nó
   bằng cp thuần — 30% trọng số kéo mạng khỏi đúng cái đang đo.
2. Mạng khớp nhãn Stockfish **tốt hơn hàm thủ công 16%** (đo bằng tương quan)
   nhưng đấu ở cùng số nút thì **không hơn** — độ chính xác tuyệt đối không
   tự động thành sức cờ, vì alpha-beta chỉ cần đúng thứ tự tương đối giữa
   các nước. Đây là bằng chứng cho thấy nên đổi hàm mất mát sang học xếp
   hạng thay vì hồi quy centipawn thuần.

Thước đo đáng tin, theo thứ tự:

1. `play_stockfish.py --ladder` — đấu thật với Stockfish bị giới hạn Elo,
   khai cuộc bắt cặp (mỗi thế cờ đấu 2 lần đổi màu) — đáng tin nhất, cho ra
   một con số Elo tuyệt đối
2. `match.py` — đấu đối kháng hai hàm lượng giá ở **cùng số nút hoặc cùng
   thời gian** (công bằng về tính toán)
3. Tương quan với nhãn Stockfish trên **tập giữ lại đã lọc trùng** với tập train
4. Val loss — chỉ dùng để phát hiện overfit, không dùng để so hai mục tiêu khác nhau

## Kết quả hiện tại (tham khảo, xem README.md để cập nhật)

Mạng `weights/cap_kb64_h256.npz` (KING_BUCKETS=64, HIDDEN=256, 10,5 triệu tham
số, train trên 7,91 triệu thế cờ) đạt khoảng **2000–2070 Elo** khi đấu thật
với Stockfish 18 bị giới hạn sức. Khoảng cách train/val của mạng này là
**+235%** — đang overfit, cần thêm dữ liệu (dự kiến 15–20 triệu thế cờ) để
tận dụng hết 10,5 triệu tham số.

## Bản đồ file

| File | Vai trò |
|---|---|
| `chess_core.py` | Bitboard, sinh nước đi, make/unmake, FEN, Zobrist, `Game` (luật hoà) |
| `search.py` | Negamax + alpha-beta, TT, quiescence, null-move, LMR |
| `evaluate.py` | Hàm lượng giá thủ công PeSTO (tapered), mốc so sánh |
| `scoring.py` | Ánh xạ cp → 0..1000, hiệu chỉnh mốc 505, dải chiếu hết |
| `nnue.py` | Bộ đặc trưng, suy luận numpy, mô hình PyTorch |
| `main.py` | CLI chấm điểm, chế độ tương tác |
| `server.py` + `web/` | Bàn phân tích web, mũi tên nước tốt nhất, streaming |
| `serve.sh` | Khởi động bàn phân tích, tự đọc KING_BUCKETS/HIDDEN từ .npz |
| `test_engine.py` | 86 test hồi quy |
| `check_purity.py` | Thực thi ranh giới thư viện ngoài |
| `datagen_sf.py` | Sinh dữ liệu, Stockfish gán nhãn |
| `make_book.py` | Sách khai cuộc bằng MultiPV |
| `train.py` | Huấn luyện NNUE |
| `match.py` | Đấu đối kháng hai hàm lượng giá |
| `play_stockfish.py` | Đấu với Stockfish thật, xuất PGN, thang Elo |

---

<a id="english"></a>

# Kỳ Lân v1 — chess position scorer on a 0–1000 scale

Given any position, answer one question: **how many points out of 1000 is
White getting?**

```bash
./serve.sh                                                   # web analysis board
python3 main.py "<FEN>" --nnue weights/cap_kb64_h256.npz --depth 8
```

**[⇡ Bản tiếng Việt ở trên](#kỳ-lân-v1--chấm-điểm-thế-cờ-vua-trên-thang-0–1000)**

## Invariant that must never break

**The real engine must NEVER depend on an external chess library.** No
python-chess, no Stockfish, no chess library of any kind. Only the training
pipeline is allowed to use them.

```
CLEAN (engine)    chess_core.py  search.py  evaluate.py  scoring.py  nnue.py
                  main.py  server.py  test_engine.py
ALLOWED (train)   datagen_sf.py  make_book.py  train.py  match.py
                  play_stockfish.py  compare_evals.py  datagen.py
```

This boundary is **enforced by code**, not convention:

```bash
python3 check_purity.py   # scans imports + loads the engine with `chess` blocked
```

Run it after every change touching an engine file. It walks the AST to scan
only strings that actually **execute** in the code, skipping docstrings and
comments (to avoid false positives when a file merely *mentions* Stockfish in
its documentation) — then it actually boots the engine with `chess` blocked
at `sys.meta_path`, so hidden dependencies get exposed too.

## The scoring scale

Meaning: **S = 1000 × White's expected game outcome.**

| Score | Meaning |
|---|---|
| 1000 | White mates on this exact move |
| 991–999 | White has a forced mate (991 = furthest out) |
| 505 | **Starting position** — White's first-move advantage is worth 5 points |
| 500 | Perfectly balanced, or already drawn by rule |
| 0 | Black mates on this exact move |

The 505 mark is **not an arbitrary additive constant**.
`scoring.calibrate(cp_starting_position)` sets an internal offset inside the
sigmoid so that whichever evaluation function is in use, scoring the
starting position, yields exactly 505. That's why swapping from the
handcrafted evaluator to NNUE keeps the anchor fixed. The calibration result
is cached in `.calib.json`; delete that file or pass `--recalibrate` when
you change the evaluation function.

The non-mate range is clamped to [10, 990] so it never touches the mate
range — a forced mate always outranks any material advantage, however large.

## Testing

```bash
python3 test_engine.py     # 86 tests: perft, FEN, Zobrist, draw rules, scoring, search
python3 check_purity.py    # external-library boundary
```

`test_engine.py` must pass 100% before every commit. It covers 14 groups:
perft on 6 standard positions, make/unmake state restoration, incremental
Zobrist matching a from-scratch recompute, mate-in-one, forced mate,
scoring scale, color symmetry, castling, en passant, the 50-move rule,
threefold repetition, stalemate/insufficient material, promotion, and
**a search aborted mid-way must still return a legal move**.

Cross-validated independently against python-chess (55k+ positions, exact
match) — rerun whenever `chess_core.py` changes.

## Traps hit before — don't repeat them

**SearchAbort skipping unmake_move.** When time/nodes run out mid-search,
`SearchAbort` propagates through the recursive `negamax` frames, causing
`unmake_move` to be skipped — the internal board (`work`) gets stuck mid-way
through a trial move, sometimes with the wrong side to move. `extract_pv`
then reads the wrong position and `search()` could return an **illegal**
move. Fixed by restoring `work = pos.copy()` when catching `SearchAbort`,
plus a final safety net: if the returned move isn't in the legal list, fall
back to `legal[0]`. This bug had silently corrupted every `match.py` /
`play_stockfish.py` result, since they played the returned move straight
onto the board with no re-validation — both now check legality before
`push()`.

**50-move clock and promotion.** In `make_move`, the `piece` variable gets
reassigned to the new piece on promotion. Must use the `moved_type` saved
beforehand, otherwise a non-capturing promotion gets counted as a quiet
move. Perft can't catch this since it only counts moves.

**En passant square.** Only record it when the capture is *actually legal*
(`_has_legal_ep`). Recording it unconditionally lets a "ghost" ep square
leak into the Zobrist key, making two identical positions hash differently
and breaking threefold-repetition detection. `set_fen` must filter it too,
since many external FENs write it unconditionally.

**`NNUE_KING_BUCKETS` must match between training and inference.**
`nnue.py` checks this and raises a clear error on mismatch — don't remove
that check; a silent mismatch produces meaningless numbers. `serve.sh` reads
this value straight from the `.npz` file so you never set it by hand.

**stdout buffering.** When logging to a file, `print` doesn't auto-flush.
Run python with `-u`, otherwise you'll think the process hung while it's
running fine.

**CPU is faster than MPS for this network.** Measured 28ms/batch on CPU vs
58ms on MPS — the network is small enough that GPU kernel-launch overhead
dominates the actual compute. `train.py --device` defaulting to `cpu` is
deliberate.

**Data-generation chunk size.** A small `datagen_sf.py --chunk` makes
Stockfish reload its ~50MB NNUE network constantly. Raising it from 8 to 150
gives 2.3× the throughput. Data is still flushed per-game, so a large chunk
costs nothing but progress-line smoothness.

**`pkill -f <script_name>` kills the very script calling it.** The shell's
own command line contains the script name string, so `pkill -f run_x.sh`
inside `run_x.sh` kills the parent process before it can even log anything.
Use the bracket trick `pkill -f '[r]un_x.sh'` to exclude itself. macOS also
has no `setsid`; detach background processes with `( nohup ... & )` inside a
subshell instead.

## Training pipeline

```bash
python3 make_book.py --count 8000 --workers 7               # MultiPV opening book
python3 datagen_sf.py --games 400000 --epd data/book.epd \
    --out data/sf.txt.gz                                     # Stockfish-labeled data
NNUE_KING_BUCKETS=64 python3 -u train.py --data data/sf.txt.gz --lambda 1.0
python3 match.py --nnue weights/x.npz --games 100            # head-to-head
python3 play_stockfish.py --games 100 --ladder 1600,1800,2000,2200 \
    --time 2.0 --nnue weights/x.npz                          # real-Stockfish Elo ladder
```

Each data line: `FEN | Stockfish cp (White's perspective) | game result`.

`train.py` and `datagen_sf.py` transparently read/write both `.txt` and
`.txt.gz`. Compressed data is ~20% the size (456 MB → 95 MB) and decompresses
far faster than FEN parsing takes, so always use `.gz`. gzip supports
appending blocks, so you can still keep writing into an already-compressed
file.

The two label sources are **deliberately independent**: `cp` gives a dense,
low-noise signal; `result` gives ground truth. If both came from Stockfish,
the network would just copy Stockfish with nothing left to surpass.
`--lambda` controls the mixing ratio (1.0 = pure cp imitation).

`train.py` caches features into a **memmap directory** `<data>.feat_kb<N>/`
(not a single `.npz` file) — indices stored as `int16` when they fit,
written straight to disk in two passes (count, then write) so nothing needs
to stay fully in RAM. The cache is keyed by `KING_BUCKETS`; subsequent
training runs load in seconds instead of minutes.

`--seed` (default 1234) fixes the train/val split order — **must stay the
same** when comparing two different architectures, otherwise a measured gap
could just be due to a different split rather than the architecture itself.

## Measurement: be careful with the measurement itself

Falling loss does **not** guarantee stronger play. Bitten by this twice:

1. Trained a network on a mixed target `0.7×sigmoid(cp) + 0.3×result`, then
   scored it against pure cp — the 30% weight pulls the network away from
   exactly what's being measured.
2. The network matches Stockfish's labels **16% better** than the
   handcrafted evaluator (measured by correlation) but performs **no better**
   in a same-node match — raw accuracy doesn't automatically translate to
   playing strength, because alpha-beta only needs the correct *relative
   ordering* between moves. This is evidence for switching the loss function
   to rank-based learning instead of pure centipawn regression.

Reliable measures, in order of trust:

1. `play_stockfish.py --ladder` — real games against Elo-limited Stockfish,
   with paired openings (each opening played twice, colors swapped) — most
   trustworthy, yields an absolute Elo number
2. `match.py` — head-to-head between two evaluators at the **same node
   budget or same time budget** (fair on compute)
3. Correlation with Stockfish's labels on a **held-out set deduplicated
   against the training set**
4. Val loss — only useful for detecting overfit, never for comparing two
   different training objectives

## Current results (reference — see README.md for the latest)

The network `weights/cap_kb64_h256.npz` (KING_BUCKETS=64, HIDDEN=256, 10.5M
parameters, trained on 7.91M positions) reaches roughly **2000–2070 Elo**
in real games against Stockfish 18 at limited strength. This network's
train/val gap is **+235%** — it's overfitting, and needs more data
(projected 15–20M positions) to make full use of its 10.5M parameters.

## File map

| File | Role |
|---|---|
| `chess_core.py` | Bitboards, move generation, make/unmake, FEN, Zobrist, `Game` (draw rules) |
| `search.py` | Negamax + alpha-beta, TT, quiescence, null-move, LMR |
| `evaluate.py` | Handcrafted PeSTO evaluation (tapered), comparison baseline |
| `scoring.py` | cp → 0..1000 mapping, 505-anchor calibration, mate range |
| `nnue.py` | Feature extraction, numpy inference, PyTorch model |
| `main.py` | CLI scorer, interactive mode |
| `server.py` + `web/` | Live web analysis board, best-move arrows, streaming |
| `serve.sh` | Launches the analysis board, auto-reads KING_BUCKETS/HIDDEN from the .npz |
| `test_engine.py` | 86 regression tests |
| `check_purity.py` | Enforces the external-library boundary |
| `datagen_sf.py` | Data generation, Stockfish labeling |
| `make_book.py` | Opening book via MultiPV |
| `train.py` | NNUE training |
| `match.py` | Head-to-head between two evaluators |
| `play_stockfish.py` | Real games against Stockfish, PGN export, Elo ladder |

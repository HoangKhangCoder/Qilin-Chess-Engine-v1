# Kỳ Lân v1

Engine cờ vua tự viết từ số 0 (bitboard, alpha-beta, NNUE) — chấm điểm mọi
thế cờ trên thang **0–1000** và đạt khoảng **2000–2070 Elo**, đo bằng cách
đấu thật với Stockfish 18 bị giới hạn sức.

**[English below ↓](#english)**

```bash
./serve.sh                                    # mở bàn phân tích tại localhost:8000
python3 main.py "<FEN>" --nnue weights/cap_kb64_h256.npz --depth 8
```

---

## Mục lục

1. [Vì sao lại có dự án này](#1-vì-sao-lại-có-dự-án-này)
2. [Thang điểm 0–1000 nghĩa là gì](#2-thang-điểm-0–1000-nghĩa-là-gì)
3. [Bốn tầng của hệ thống](#3-bốn-tầng-của-hệ-thống)
4. [Ranh giới sạch — thực thi bằng code](#4-ranh-giới-sạch--thực-thi-bằng-code)
5. [Cách chạy nhanh](#5-cách-chạy-nhanh)
6. [Huấn luyện NNUE](#6-huấn-luyện-nnue)
7. [Kết quả đo được](#7-kết-quả-đo-được)
8. [Bàn phân tích trên web](#8-bàn-phân-tích-trên-web)
9. [Kiểm thử](#9-kiểm-thử)
10. [Giới hạn và hướng đi tiếp](#10-giới-hạn-và-hướng-đi-tiếp)
11. [Bản đồ file](#11-bản-đồ-file)

---

## 1. Vì sao lại có dự án này

Cho một thế cờ bất kỳ, trả lời một câu duy nhất: **Trắng đang được bao nhiêu
điểm trên 1000?** Toàn bộ luật cờ, sinh nước đi, tìm kiếm và mạng lượng giá
đều tự viết bằng Python thuần — không `python-chess`, không gọi Stockfish lúc
chạy. Stockfish chỉ xuất hiện ở pipeline **huấn luyện**, đóng vai trò thầy
giáo gán nhãn dữ liệu, và ranh giới đó được **thực thi bằng code**
(`check_purity.py`), không phải quy ước suông.

## 2. Thang điểm 0–1000 nghĩa là gì

> **S = 1000 × kỳ vọng kết quả ván cờ của Trắng**
> (thắng = 1, hoà = 0.5, thua = 0)

| Điểm | Ý nghĩa |
|---|---|
| **1000** | Trắng chiếu hết **ngay trong nước đi này** |
| 991–999 | Trắng có chiếu hết ép buộc |
| **505** | **Thế xuất phát** — lợi thế đi trước của Trắng |
| 500 | Cân bằng tuyệt đối, hoặc đã hoà theo luật |
| 1–9 | Đen có chiếu hết ép buộc |
| **0** | Đen chiếu hết ngay trong nước đi này |

Mốc 505 **không phải hằng số cộng thêm tuỳ tiện**. `scoring.calibrate()` đặt
một bù trừ bên trong hàm sigmoid sao cho hàm lượng giá đang dùng, khi chấm
thế xuất phát, ra đúng 505 điểm. Nhờ vậy đổi từ hàm thủ công sang NNUE thì
mốc vẫn giữ nguyên, và điểm số của hai cấu hình khác nhau vẫn so sánh được.

Dải không-chiếu-hết bị kẹp trong `[10, 990]` nên không bao giờ đụng dải chiếu
hết — chiếu hết ép buộc luôn được xếp trên mọi ưu thế vật chất, dù ưu thế đó
lớn đến đâu.

## 3. Bốn tầng của hệ thống

```
main.py / server.py     giao diện: FEN -> điểm 0..1000
  └─ scoring.py         cp -> 0..1000 (sigmoid + hiệu chỉnh + vùng chiếu hết)
  └─ search.py          alpha-beta: "đánh giá sâu" thật sự nằm ở đây
       └─ nnue.py            mạng nơ-ron (mặc định khi có trọng số)
       └─ evaluate.py        lượng giá thủ công PeSTO (mốc so sánh)
            └─ chess_core.py     bitboard, luật cờ, Zobrist, sinh nước đi
```

- **`chess_core.py`** — bàn cờ biểu diễn bằng bitboard 64-bit. Cài đầy đủ
  nhập thành, bắt tốt qua đường, phong cấp, luật 50 nước, lặp 3 lần, pat,
  thiếu lực chiếu hết. Đối chiếu độc lập với `python-chess` trên hàng chục
  nghìn thế cờ ngẫu nhiên — khớp hoàn toàn nước đi hợp lệ, FEN, và mọi
  trạng thái luật.
- **`search.py`** — negamax + alpha-beta, bảng chuyển vị Zobrist, sắp xếp
  nước đi (MVV-LVA, killer, history), null-move pruning, LMR, cửa sổ kỳ
  vọng, quiescence search.
- **`evaluate.py`** — bảng PeSTO (giá trị quân + vị trí ô, nội suy trung
  cuộc/tàn cuộc) cộng cơ động, cặp tượng, cấu trúc tốt, an toàn vua.
- **`nnue.py`** — mạng NNUE: đặc trưng thưa `(nhóm ô vua) × (loại quân
  tương đối) × (ô)` → accumulator dùng chung hai góc nhìn → 2 lớp dense nhỏ.
  Huấn luyện bằng PyTorch, suy luận bằng NumPy thuần (không cần torch lúc
  chạy engine).

## 4. Ranh giới sạch — thực thi bằng code

```
SẠCH (engine)     chess_core.py  search.py  evaluate.py  scoring.py
                  nnue.py  main.py  server.py  test_engine.py
ĐƯỢC DÙNG (train) datagen_sf.py  make_book.py  train.py  match.py
                  play_stockfish.py
```

```bash
python3 check_purity.py
```

Không chỉ đọc `import` bằng AST (bỏ qua docstring/chú thích để tránh dương
tính giả) — nó còn thực sự **khởi động engine với module `chess` bị chặn ở
`sys.meta_path`**, nên phụ thuộc ẩn cũng bị lộ ngay.

## 5. Cách chạy nhanh

```bash
# bàn phân tích web (khuyến nghị) — tự chọn trọng số mới nhất, tự mở trình duyệt
./serve.sh
./serve.sh --hand                              # chỉ dùng hàm lượng giá thủ công
./serve.sh --nnue weights/cap_kb64_h256.npz --port 8080

# dòng lệnh
python3 main.py "<FEN>" --depth 10
python3 main.py --moves e2e4 e7e5 g1f3
python3 main.py --interactive
python3 main.py "<FEN>" --nnue weights/cap_kb64_h256.npz -v   # in từng tầng độ sâu
```

## 6. Huấn luyện NNUE

```bash
python3 make_book.py --count 8000 --workers 7               # sách khai cuộc MultiPV
python3 datagen_sf.py --games 400000 --epd data/book.epd \
    --out data/sf.txt.gz                                    # Stockfish gán nhãn
NNUE_KING_BUCKETS=64 python3 -u train.py --data data/sf.txt.gz --lambda 1.0
python3 match.py --nnue weights/x.npz --games 100            # đấu đối kháng NNUE vs thủ công
python3 play_stockfish.py --games 100 --ladder 1600,1800,2000,2200 \
    --time 2.0 --nnue weights/x.npz                          # thang Elo với Stockfish thật
```

Mỗi dòng dữ liệu: `FEN | cp Stockfish (góc nhìn Trắng) | kết quả ván`. Hai
nguồn nhãn **cố ý độc lập** — `cp` cho tín hiệu dày ít nhiễu, `result` cho sự
thật cuối cùng — để mạng không chỉ sao chép lại đúng Stockfish.

`train.py` đọc/ghi được cả `.txt` lẫn `.txt.gz` trong suốt, và cache đặc
trưng ra memmap trên đĩa (không giữ hết trong RAM) nên nạp hàng chục triệu
thế cờ vẫn an toàn trên máy ít bộ nhớ.

## 7. Kết quả đo được

**Thang Elo**, đấu với Stockfish 18 bị giới hạn sức (`UCI_LimitStrength`),
2,0 giây/nước cho cả hai bên, khai cuộc bắt cặp (mỗi thế cờ đấu 2 lần đổi
màu để công bằng tuyệt đối):

| Đối thủ | Điểm | Tỉ lệ | Elo engine ước lượng |
|---|---|---|---|
| Stockfish (Elo 1600) | 85,0/100 | 85,0% | ≈ 1901 |
| Stockfish (Elo 1800) | 77,0/100 | 77,0% | ≈ 2010 |
| Stockfish (Elo 1900) | 69,0/100 | 69,0% | ≈ 2039 |
| Stockfish (Elo 2000) | 52,0/87 | 59,8% | ≈ 2069 |
| Stockfish (sức đầy đủ) | 0,0/10 | 0% | thua sạch |

**→ Sức cờ ước lượng: khoảng 2000–2070 Elo**, dù engine chạy bằng Python
thuần và chỉ đạt ~16–80 nghìn nút/giây (chậm hơn Stockfish khoảng 600 lần).

**Dữ liệu huấn luyện:** 7.914.424 thế cờ do Stockfish 18 gán nhãn, cộng 8.338
thế cờ khai cuộc từ MultiPV.

**Thí nghiệm kiến trúc** (cùng dữ liệu, cùng tập validation nhờ seed cố
định) cho thấy nút thắt từng nằm ở *dung lượng mạng* chứ không phải *lượng
dữ liệu* — tăng gấp đôi dữ liệu ở mạng nhỏ không cải thiện gì, nhưng tăng
dung lượng mạng ở cùng lượng dữ liệu cải thiện rõ rệt:

| Kiến trúc | Tham số | Val loss | So với mốc |
|---|---|---|---|
| KB=4, H=256 (mốc) | 0,7 triệu | 0,00306 | — |
| KB=4, H=512 | 1,3 triệu | 0,00271 | tốt hơn 11% |
| **KB=64, H=256** (đang dùng) | **10,5 triệu** | **0,00228** | **tốt hơn 25%** |

Một phát hiện quan trọng khác: mạng chấm điểm **sát Stockfish hơn hàm thủ
công 16%** (đo bằng tương quan trên tập giữ lại) mà **không thắng hơn khi
đấu thật** ở cùng số nút tìm kiếm — bằng chứng cho thấy độ chính xác tuyệt
đối không tự động chuyển thành sức cờ, vì alpha-beta chỉ cần đúng *thứ tự
tương đối* giữa các nước.

## 8. Bàn phân tích trên web

`server.py` + `web/index.html`: bàn cờ tương tác, mũi tên chỉ nước tốt nhất,
thanh điểm 0–1000, kết quả **phát trực tiếp theo từng tầng độ sâu** (mũi tên
đầu tiên hiện sau ~40 mili-giây thay vì chờ trọn vài giây). `serve.sh` tự
đọc `KING_BUCKETS`/`HIDDEN` từ file `.npz` nên không cần nhớ biến môi trường.

## 9. Kiểm thử

```bash
python3 test_engine.py     # 86 test: perft, FEN, Zobrist, luật hoà, thang điểm
python3 check_purity.py    # ranh giới thư viện ngoài
```

Phủ 14 nhóm: perft trên 6 vị trí chuẩn, make/unmake khôi phục trạng thái,
Zobrist tăng dần khớp tính lại, chiếu hết 1 nước và chiếu hết ép buộc, thang
điểm, đối xứng màu, nhập thành, bắt tốt qua đường, luật 50 nước, lặp 3 lần,
pat/thiếu lực, phong cấp, và **tìm kiếm bị ngắt giữa chừng vẫn phải trả về
nước hợp lệ** (một bug thật đã từng làm hỏng kết quả đấu, giờ có lưới an
toàn + test hồi quy).

## 10. Giới hạn và hướng đi tiếp

- **Tốc độ.** ~16–80 nghìn nút/giây trong Python thuần, chậm hơn Stockfish
  khoảng 600 lần. Kế hoạch: accumulator NNUE **tăng dần** (đúng chữ
  *Updatable* trong NNUE — hiện đang tính lại toàn bộ mỗi lần).
- **Chưa có SEE** (Static Exchange Evaluation). Sẽ dùng để sắp xếp nước đi
  ở mọi nơi, nhưng chỉ được cắt tỉa hẳn nước đi trong quiescence search và
  chỉ khi **không đang bị chiếu** — để không loại nhầm các đòn hy sinh có
  chuỗi chiếu ép buộc phía sau.
- **Mạng KB=64 đang overfit** (khoảng cách train/val +235%) — cần thêm dữ
  liệu (từ 7,9 lên khoảng 15–20 triệu thế cờ) để tận dụng hết dung lượng
  10,5 triệu tham số.
- **Hàm mất mát học centipawn, chưa học thứ tự nước đi.** Đây là hướng có
  thể giải thích nghịch lý "khớp Stockfish tốt hơn mà đấu không hơn".
- **Vòng lặp tự chơi chưa chạy.** `datagen.py --nnue` đã sẵn sàng để sinh
  dữ liệu bằng chính mạng đã huấn luyện, thay vì mãi học từ Stockfish.

## 11. Bản đồ file

| File | Vai trò |
|---|---|
| `chess_core.py` | Bitboard, sinh nước đi, make/unmake, FEN, Zobrist, `Game` |
| `search.py` | Negamax + alpha-beta, TT, quiescence, null-move, LMR |
| `evaluate.py` | Hàm lượng giá thủ công PeSTO |
| `scoring.py` | Ánh xạ cp → 0..1000, hiệu chỉnh mốc 505 |
| `nnue.py` | Bộ đặc trưng, suy luận numpy, mô hình PyTorch |
| `main.py` | CLI chấm điểm |
| `server.py` + `web/` | Bàn phân tích trực tiếp, mũi tên gợi ý |
| `serve.sh` | Khởi động bàn phân tích |
| `test_engine.py` | 86 test hồi quy |
| `check_purity.py` | Thực thi ranh giới thư viện ngoài |
| `datagen_sf.py` | Sinh dữ liệu, Stockfish gán nhãn |
| `make_book.py` | Sách khai cuộc bằng MultiPV |
| `train.py` | Huấn luyện NNUE |
| `match.py` | Đấu đối kháng hai hàm lượng giá |
| `play_stockfish.py` | Đấu với Stockfish thật, xuất PGN, thang Elo |

---

*Giấy phép: MIT (xem `LICENSE`).*

---

<a id="english"></a>

# Kỳ Lân v1

A chess engine written from scratch (bitboards, alpha-beta, NNUE) — scores
any position on a **0–1000** scale and reaches roughly **2000–2070 Elo**,
measured by playing real games against a strength-limited Stockfish 18.

**[⇡ Bản tiếng Việt ở trên](#kỳ-lân-v1)**

```bash
./serve.sh                                    # opens the analysis board at localhost:8000
python3 main.py "<FEN>" --nnue weights/cap_kb64_h256.npz --depth 8
```

---

## Table of Contents

1. [Why this project exists](#1-why-this-project-exists)
2. [What the 0–1000 scale means](#2-what-the-0–1000-scale-means)
3. [The four layers of the system](#3-the-four-layers-of-the-system)
4. [A clean boundary — enforced by code](#4-a-clean-boundary--enforced-by-code)
5. [Quick start](#5-quick-start)
6. [Training the NNUE](#6-training-the-nnue)
7. [Measured results](#7-measured-results)
8. [The web analysis board](#8-the-web-analysis-board)
9. [Testing](#9-testing)
10. [Limitations and what's next](#10-limitations-and-whats-next)
11. [File map](#11-file-map)

---

## 1. Why this project exists

Given any position, answer exactly one question: **how many points out of
1000 is White getting?** Every bit of chess rules, move generation, search,
and the evaluation network is written from scratch in pure Python — no
`python-chess`, no calling Stockfish at runtime. Stockfish only appears in
the **training** pipeline, playing the role of a teacher that labels data,
and that boundary is **enforced by code** (`check_purity.py`), not just
convention.

## 2. What the 0–1000 scale means

> **S = 1000 × White's expected game outcome**
> (win = 1, draw = 0.5, loss = 0)

| Score | Meaning |
|---|---|
| **1000** | White mates **on this exact move** |
| 991–999 | White has a forced mate |
| **505** | **Starting position** — White's first-move advantage |
| 500 | Perfectly balanced, or already drawn by rule |
| 1–9 | Black has a forced mate |
| **0** | Black mates on this exact move |

The 505 anchor is **not an arbitrary added constant**. `scoring.calibrate()`
sets an offset inside the sigmoid so that whichever evaluation function is
in use, scoring the starting position, yields exactly 505 points. That's
why swapping the handcrafted evaluator for NNUE keeps the anchor fixed, and
why scores from two different configurations remain comparable.

The non-mate range is clamped to `[10, 990]` so it never touches the mate
range — a forced mate always outranks any material advantage, however large.

## 3. The four layers of the system

```
main.py / server.py     interface: FEN -> 0..1000 score
  └─ scoring.py         cp -> 0..1000 (sigmoid + calibration + mate range)
  └─ search.py          alpha-beta: the real "deep evaluation" lives here
       └─ nnue.py            neural evaluator (default once weights exist)
       └─ evaluate.py        handcrafted PeSTO evaluator (comparison baseline)
            └─ chess_core.py     bitboards, chess rules, Zobrist, move generation
```

- **`chess_core.py`** — the board is represented as 64-bit bitboards.
  Fully implements castling, en passant, promotion, the 50-move rule,
  threefold repetition, stalemate, insufficient material. Cross-validated
  independently against `python-chess` on tens of thousands of random
  positions — exact match on legal moves, FEN, and every rule state.
- **`search.py`** — negamax + alpha-beta, Zobrist transposition table, move
  ordering (MVV-LVA, killers, history), null-move pruning, LMR, aspiration
  windows, quiescence search.
- **`evaluate.py`** — a PeSTO table (piece values + piece-square tables,
  tapered between middlegame/endgame) plus mobility, bishop pair, pawn
  structure, king safety.
- **`nnue.py`** — the NNUE network: sparse features
  `(king square bucket) × (relative piece type) × (square)` → an
  accumulator shared across both perspectives → two small dense layers.
  Trained with PyTorch, inference runs on pure NumPy (torch isn't needed to
  run the engine).

## 4. A clean boundary — enforced by code

```
CLEAN (engine)    chess_core.py  search.py  evaluate.py  scoring.py
                  nnue.py  main.py  server.py  test_engine.py
ALLOWED (train)   datagen_sf.py  make_book.py  train.py  match.py
                  play_stockfish.py
```

```bash
python3 check_purity.py
```

It doesn't just read `import` statements via AST (skipping
docstrings/comments to avoid false positives) — it also actually **boots
the engine with the `chess` module blocked at `sys.meta_path`**, so hidden
dependencies get exposed immediately.

## 5. Quick start

```bash
# web analysis board (recommended) — auto-picks the latest weights, opens the browser
./serve.sh
./serve.sh --hand                              # handcrafted evaluator only
./serve.sh --nnue weights/cap_kb64_h256.npz --port 8080

# command line
python3 main.py "<FEN>" --depth 10
python3 main.py --moves e2e4 e7e5 g1f3
python3 main.py --interactive
python3 main.py "<FEN>" --nnue weights/cap_kb64_h256.npz -v   # print each depth iteration
```

## 6. Training the NNUE

```bash
python3 make_book.py --count 8000 --workers 7               # MultiPV opening book
python3 datagen_sf.py --games 400000 --epd data/book.epd \
    --out data/sf.txt.gz                                    # Stockfish-labeled data
NNUE_KING_BUCKETS=64 python3 -u train.py --data data/sf.txt.gz --lambda 1.0
python3 match.py --nnue weights/x.npz --games 100            # NNUE vs handcrafted head-to-head
python3 play_stockfish.py --games 100 --ladder 1600,1800,2000,2200 \
    --time 2.0 --nnue weights/x.npz                          # Elo ladder vs real Stockfish
```

Each data line: `FEN | Stockfish cp (White's perspective) | game result`. The
two label sources are **deliberately independent** — `cp` gives a dense,
low-noise signal, `result` gives ground truth — so the network never simply
copies Stockfish.

`train.py` transparently reads/writes both `.txt` and `.txt.gz`, and caches
features to a disk memmap (not held fully in RAM), so loading tens of
millions of positions stays safe even on memory-constrained machines.

## 7. Measured results

**Elo ladder**, played against a strength-limited Stockfish 18
(`UCI_LimitStrength`), 2.0 seconds per move for both sides, paired openings
(each position played twice with colors swapped for perfect fairness):

| Opponent | Score | Rate | Estimated engine Elo |
|---|---|---|---|
| Stockfish (Elo 1600) | 85.0/100 | 85.0% | ≈ 1901 |
| Stockfish (Elo 1800) | 77.0/100 | 77.0% | ≈ 2010 |
| Stockfish (Elo 1900) | 69.0/100 | 69.0% | ≈ 2039 |
| Stockfish (Elo 2000) | 52.0/87 | 59.8% | ≈ 2069 |
| Stockfish (full strength) | 0.0/10 | 0% | swept |

**→ Estimated playing strength: roughly 2000–2070 Elo**, despite running in
pure Python at only ~16–80 thousand nodes/second (about 600× slower than
Stockfish).

**Training data:** 7,914,424 positions labeled by Stockfish 18, plus 8,338
opening positions from MultiPV.

**Architecture experiment** (same data, same validation split via a fixed
seed) showed the bottleneck used to be *network capacity*, not *data
volume* — doubling data on a small network improved nothing, but growing
network capacity on the same data improved clearly:

| Architecture | Parameters | Val loss | vs. baseline |
|---|---|---|---|
| KB=4, H=256 (baseline) | 0.7M | 0.00306 | — |
| KB=4, H=512 | 1.3M | 0.00271 | 11% better |
| **KB=64, H=256** (current) | **10.5M** | **0.00228** | **25% better** |

Another important finding: the network matches Stockfish's evaluations
**16% more accurately** than the handcrafted evaluator (measured by
correlation on held-out positions) yet **doesn't play any better** in a
same-node match — evidence that raw accuracy doesn't automatically become
playing strength, since alpha-beta only needs the correct *relative
ordering* between moves.

## 8. The web analysis board

`server.py` + `web/index.html`: an interactive board, arrows pointing to the
best move, a 0–1000 eval bar, results **streamed per depth iteration** (the
first arrow appears after ~40 milliseconds instead of waiting several
seconds). `serve.sh` auto-reads `KING_BUCKETS`/`HIDDEN` from the `.npz`
file, so you never need to remember the environment variables.

## 9. Testing

```bash
python3 test_engine.py     # 86 tests: perft, FEN, Zobrist, draw rules, scoring
python3 check_purity.py    # external-library boundary
```

Covers 14 groups: perft on 6 standard positions, make/unmake state
restoration, incremental Zobrist matching a from-scratch recompute,
mate-in-one and forced mate, scoring scale, color symmetry, castling, en
passant, the 50-move rule, threefold repetition, stalemate/insufficient
material, promotion, and **a search aborted mid-way must still return a
legal move** (a real bug that once silently corrupted match results, now
guarded by a safety net plus a regression test).

## 10. Limitations and what's next

- **Speed.** ~16–80 thousand nodes/second in pure Python, about 600× slower
  than Stockfish. Planned: an **incremental** NNUE accumulator (the "U" in
  NNUE stands for *Updatable* — currently everything is recomputed from
  scratch on every call).
- **No SEE yet** (Static Exchange Evaluation). Will be used for move
  ordering everywhere, but only allowed to fully prune a move inside
  quiescence search, and only **when not in check** — so a sacrifice
  followed by a forcing check sequence never gets wrongly discarded.
- **The KB=64 network is overfitting** (train/val gap +235%) — needs more
  data (from 7.9M to roughly 15–20M positions) to make full use of its
  10.5M parameters.
- **The loss function learns centipawns, not move ranking.** This may
  explain the paradox of "matches Stockfish better but doesn't play
  better."
- **No self-play loop yet.** `datagen.py --nnue` is already wired up to
  generate data using the trained network itself, instead of forever
  learning from Stockfish.

## 11. File map

| File | Role |
|---|---|
| `chess_core.py` | Bitboards, move generation, make/unmake, FEN, Zobrist, `Game` |
| `search.py` | Negamax + alpha-beta, TT, quiescence, null-move, LMR |
| `evaluate.py` | Handcrafted PeSTO evaluation |
| `scoring.py` | cp → 0..1000 mapping, 505-anchor calibration |
| `nnue.py` | Feature extraction, numpy inference, PyTorch model |
| `main.py` | CLI scorer |
| `server.py` + `web/` | Live analysis board, best-move arrows |
| `serve.sh` | Launches the analysis board |
| `test_engine.py` | 86 regression tests |
| `check_purity.py` | Enforces the external-library boundary |
| `datagen_sf.py` | Data generation, Stockfish labeling |
| `make_book.py` | Opening book via MultiPV |
| `train.py` | NNUE training |
| `match.py` | Head-to-head between two evaluators |
| `play_stockfish.py` | Real games against Stockfish, PGN export, Elo ladder |

---

*License: MIT (see `LICENSE`).*

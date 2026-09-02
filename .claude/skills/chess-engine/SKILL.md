---
name: chess-engine
description: Quy trình làm việc với engine cờ vua chấm điểm 0-1000 trong repo này - sinh dữ liệu bằng Stockfish, huấn luyện NNUE, đo lường xem mạng có thật sự khá hơn, và giữ ranh giới engine sạch thư viện ngoài. Dùng khi cần train lại mạng, sinh thêm dữ liệu, đánh giá chất lượng hàm lượng giá, sửa luật cờ trong chess_core, hoặc chạy đấu đối kháng.
---

# Quy trình engine cờ vua

## Trước khi sửa bất cứ thứ gì

```bash
python3 test_engine.py && python3 check_purity.py
```

Cả hai phải đạt. `test_engine.py` mất ~3 phút (perft chiếm phần lớn).

## Ranh giới không được phá

Engine chạy thật (`chess_core` `search` `evaluate` `scoring` `nnue` `main`
`test_engine`) **không được import python-chess hay gọi Stockfish**. Chỉ
`datagen_sf.py` `make_book.py` `train.py` `match.py` được phép.

`check_purity.py` thực thi điều này bằng cách nạp engine với module `chess` bị
chặn ở `sys.meta_path` — phụ thuộc ẩn cũng bị lộ. Chạy nó sau mọi thay đổi
đụng file engine.

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

## Sinh thêm dữ liệu

```bash
# sách khai cuộc (một lần, ~4.5 thế cờ/giây với 7 worker)
python3 make_book.py --count 8000 --workers 7

# sinh dữ liệu - CHUNK LỚN, nếu không Stockfish nạp lại mạng 50MB liên tục
python3 datagen_sf.py --games 400000 --depth 9 --workers 7 --chunk 150 \
  --epd data/book.epd --out data/sf.txt
```

Dữ liệu **cộng dồn** vào file, chạy lại bao nhiêu lần cũng được. Worker flush
theo từng ván nên dừng bằng `pkill -f datagen_sf.py` không mất gì.

Thông lượng tham chiếu: ~570k thế cờ/giờ với 7 worker depth 9 trên máy 8 nhân.

Kiểm tra dữ liệu sau khi sinh: dòng hỏng phải bằng 0, trùng lặp dưới 1%, tỉ lệ
mẫu có |cp|<100 nên khoảng 40% (vùng cân bằng là nơi độ chính xác quyết định).

## Huấn luyện

```bash
NNUE_KING_BUCKETS=4 python3 -u train.py --data data/sf.txt \
  --epochs 40 --batch 8192 --device cpu --out weights/nnue_kb4.npz
```

- `-u` bắt buộc khi ghi log ra file, nếu không tưởng treo
- `--device cpu` là đúng: đo được CPU nhanh gấp đôi MPS ở kích thước mạng này
- `NNUE_KING_BUCKETS` phải giống hệt lúc chạy engine sau này
- Lần train đầu tạo cache `<data>.feat_kb<N>.npz`, các lần sau nạp trong vài giây

Đọc train/val gap để biết cần gì:

| Dấu hiệu | Nghĩa | Việc cần làm |
|---|---|---|
| val >> train | overfit | thêm dữ liệu |
| val ≈ train, cả hai cao | underfit | tăng KING_BUCKETS hoặc HIDDEN |
| val ngừng giảm sớm | hết tín hiệu | dữ liệu đa dạng hơn, không phải nhiều hơn |

## Đo xem mạng có THẬT SỰ khá hơn

Val loss thấp không có nghĩa cờ đánh hay hơn. Thứ tự tin cậy:

```bash
# 1. Đấu đối kháng ở CÙNG số nút - đáng tin nhất
python3 match.py --nnue weights/nnue_kb4.npz --games 40 --nodes 4000

# 2. Tương quan với Stockfish trên tập giữ lại (phải lọc trùng với tập train!)
python3 datagen_sf.py --games 40 --depth 12 --seed 999999 --out data/_holdout.txt
# rồi loại mọi FEN đã có trong data/sf.txt trước khi đo
```

**Bẫy đã dính:** train mạng dự đoán hỗn hợp `λ·sigmoid(cp) + (1−λ)·kết_quả` rồi
chấm nó bằng cp thuần. Khi so với nhãn cp, phải train `--lambda 1.0`, nếu không
đang phạt mạng vì đúng thứ nó được dạy.

Không so val loss giữa hai giá trị `--lambda` khác nhau — mục tiêu học khác nhau
thì loss không cùng thang.

## Bất biến của thang điểm

Sau mọi thay đổi đụng `scoring.py` hay hàm lượng giá, kiểm lại:

```bash
python3 main.py --recalibrate --depth 8                    # phải ra 505
python3 main.py "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"        # phải ra 1000
python3 main.py "5k2/5P2/5K2/8/8/8/8/8 b - - 0 1"          # phải ra 500 (pat)
```

Mốc 505 đến từ `scoring.calibrate()`, không phải cộng thêm 5. Đổi hàm lượng giá
thì phải `--recalibrate` hoặc xoá `.calib.json`.

## Giới hạn cần nói thẳng

Engine này chạy ~50–70k nút/giây (Python thuần). Stockfish chạy 10–50 **triệu**.
Chênh 100–500 lần đó không thể bù bằng hàm lượng giá tốt hơn. Thêm nữa, dữ liệu
huấn luyện do chính Stockfish gán nhãn, nên trần lý thuyết là "xấp xỉ Stockfish,
kém chính xác hơn".

Mục tiêu đo được và có ý nghĩa: đấu với Stockfish **bị giới hạn Elo**
(`UCI_LimitStrength` + `UCI_Elo`) để biết engine mạnh cỡ nào, rồi đẩy con số đó
lên. Đừng đặt mục tiêu thắng Stockfish đầy đủ.

# chess-analysis — chấm điểm thế cờ vua trên thang 0–1000

Cho một thế cờ bất kỳ, trả lời: **Trắng đang được bao nhiêu điểm trên 1000?**

```bash
python3 main.py "<FEN>" --depth 8
python3 main.py --nnue weights/nnue_kb4.npz --depth 8    # dùng mạng nơ-ron
```

## Ràng buộc bất di bất dịch

**Engine chạy thật KHÔNG được dùng thư viện cờ ngoài.** Không python-chess, không
Stockfish, không thư viện cờ nào. Chỉ pipeline huấn luyện mới được dùng.

```
SẠCH (engine)    chess_core.py  search.py  evaluate.py  scoring.py  nnue.py  main.py  test_engine.py
ĐƯỢC DÙNG (train) datagen_sf.py  make_book.py  train.py  match.py
```

Ranh giới này được **thực thi bằng code**, không phải quy ước:

```bash
python3 check_purity.py   # quét import + nạp engine trong môi trường đã chặn module `chess`
```

Chạy nó sau mọi thay đổi đụng tới các file engine. Nó không chỉ đọc import mà
còn thực sự khởi động engine với `chess` bị chặn ở `sys.meta_path`, nên phụ
thuộc ẩn cũng bị lộ.

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
python3 test_engine.py     # 85 test: perft, FEN, Zobrist, luật hoà, thang điểm
python3 check_purity.py    # ranh giới thư viện ngoài
```

`test_engine.py` phải đạt 100% trước mọi commit. Nó phủ 13 nhóm: perft trên 6 vị
trí chuẩn, make/unmake khôi phục trạng thái, Zobrist tăng dần khớp tính lại,
chiếu hết 1 nước, chiếu hết ép buộc, thang điểm, đối xứng màu, nhập thành, bắt
tốt qua đường, luật 50 nước, lặp 3 lần, pat/thiếu lực, phong cấp.

Đối chiếu độc lập với python-chess (55k thế cờ, khớp hoàn toàn) — script ở
scratchpad, chạy lại khi sửa `chess_core.py`.

## Bẫy đã gặp — đừng lặp lại

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

**Đệm stdout.** Khi ghi log ra file, `print` không tự flush. Chạy python bằng
`-u`, nếu không sẽ tưởng tiến trình treo trong khi nó chạy bình thường.

**CPU nhanh hơn MPS cho mạng này.** Đo được 28ms/batch trên CPU so với 58ms trên
MPS — mạng quá nhỏ nên chi phí khởi chạy kernel GPU lấn át tính toán.
`train.py --device` mặc định `cpu` là có chủ ý.

**Kích thước lô sinh dữ liệu.** `datagen_sf.py --chunk` nhỏ khiến Stockfish nạp
lại mạng NNUE ~50MB liên tục. Tăng từ 8 lên 150 cho thông lượng gấp 2,3 lần.
Dữ liệu vẫn ghi theo từng ván nên chunk lớn không mất gì ngoài độ mượt của dòng
tiến độ.

## Pipeline huấn luyện

```bash
python3 make_book.py --count 8000                          # sách khai cuộc MultiPV
python3 datagen_sf.py --games 400000 --epd data/book.epd --out data/sf.txt.gz
NNUE_KING_BUCKETS=4 python3 -u train.py --epochs 40        # train
python3 match.py --nnue weights/nnue_kb4.npz --games 40    # đấu đối kháng
```

Mỗi dòng dữ liệu: `FEN | cp Stockfish (góc nhìn Trắng) | kết quả ván`.

`train.py` và `datagen_sf.py` đọc/ghi được cả `.txt` lẫn `.txt.gz` trong suốt.
Dữ liệu nén còn ~20% (456 MB → 95 MB) và giải nén nhanh hơn nhiều so với thời
gian phân tích FEN, nên luôn dùng `.gz`. gzip cho phép nối khối nên vẫn ghi
tiếp được vào file đã nén.

Hai nguồn nhãn **cố ý độc lập**: `cp` cho tín hiệu dày ít nhiễu, `result` cho sự
thật cuối cùng. Nếu lấy cả hai từ Stockfish thì mạng chỉ sao chép Stockfish và
không còn gì để vượt. `--lambda` điều khiển tỉ lệ trộn (1.0 = chỉ bắt chước cp).

`train.py` cache đặc trưng ra `<data>.feat_kb<N>.npz`, nên lần train sau nạp
trong vài giây thay vì vài phút. Cache gắn với `KING_BUCKETS`.

## Đo lường: cẩn thận với chính phép đo

Loss giảm **không** đảm bảo cờ đánh hay hơn. Đã dính một lần: train mạng dự đoán
hỗn hợp `0,7×sigmoid(cp) + 0,3×kết_quả` rồi lại chấm nó bằng cp thuần — 30%
trọng số kéo mạng khỏi đúng cái đang đo.

Thước đo đáng tin, theo thứ tự:

1. `match.py` — đấu đối kháng ở **cùng số nút** (công bằng về tính toán)
2. Tương quan với nhãn Stockfish trên **tập giữ lại đã lọc trùng** với tập train
3. Val loss — chỉ dùng để phát hiện overfit, không dùng để so hai mục tiêu khác nhau

## Bản đồ file

| File | Vai trò |
|---|---|
| `chess_core.py` | Bitboard, sinh nước đi, make/unmake, FEN, Zobrist, `Game` (luật hoà) |
| `search.py` | Negamax + alpha-beta, TT, quiescence, null-move, LMR |
| `evaluate.py` | Hàm lượng giá thủ công PeSTO (tapered), mốc so sánh |
| `scoring.py` | Ánh xạ cp → 0..1000, hiệu chỉnh mốc 505, dải chiếu hết |
| `nnue.py` | Bộ đặc trưng, suy luận numpy, mô hình PyTorch |
| `main.py` | CLI chấm điểm, chế độ tương tác |
| `test_engine.py` | 85 test hồi quy |
| `check_purity.py` | Thực thi ranh giới thư viện ngoài |
| `datagen_sf.py` | Sinh dữ liệu, Stockfish gán nhãn |
| `play_stockfish.py` | Đấu với Stockfish, xuất PGN, thang Elo |
| `server.py` + `web/` | Bàn phân tích web, mũi tên nước tốt nhất |
| `serve.sh` | Khởi động bàn phân tích, tự đọc KING_BUCKETS từ .npz |
| `make_book.py` | Sách khai cuộc bằng MultiPV |
| `train.py` | Huấn luyện NNUE |
| `match.py` | Đấu đối kháng hai hàm lượng giá |

# Chấm điểm thế cờ vua trên thang 0–1000

Cho một thế cờ bất kỳ, trả lời: **Trắng đang được bao nhiêu điểm?**

```
python main.py "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
```

Không dùng bất kỳ thư viện cờ vua nào (không `python-chess`, không Stockfish).
Toàn bộ luật cờ, sinh nước đi và tìm kiếm đều tự viết. PyTorch chỉ dùng để
**huấn luyện** mạng lượng giá; khi chạy engine chỉ cần NumPy.

---

## 1. Thang điểm nghĩa là gì

> **S = 1000 × kỳ vọng kết quả ván cờ của Trắng**
> (thắng = 1, hoà = 0.5, thua = 0)

| điểm | ý nghĩa |
|---|---|
| **1000** | Trắng chiếu hết **ngay trong nước đi này** |
| 991–999 | Trắng có chiếu hết ép buộc sau 2–10 nước |
| 505 | **thế xuất phát** — lợi thế đi trước của Trắng |
| 500 | hoàn toàn cân bằng (kể cả các thế đã hoà theo luật) |
| 1–9 | Đen có chiếu hết ép buộc |
| **0** | Đen chiếu hết ngay trong nước đi này |

Định nghĩa này khiến "505 ở thế xuất phát" **không phải** một con số cộng thêm
tuỳ tiện. Nó nói: *cầm quân Trắng ở thế xuất phát thì kỳ vọng ghi được 50.5%
số điểm* — đúng thống kê thực tế của các ván đấu trình độ cao.

### Vì sao không dùng thẳng centipawn

Centipawn không có trần: hơn 3 quân hậu là +2700cp, nhưng cũng chỉ là "thắng",
không hơn gì +1500cp. Thang xác suất có trần tự nhiên tại 0 và 1000, nên mốc
"chiếu hết = 1000" gắn được vào đúng chỗ. Chuyển đổi bằng hàm sigmoid:

```
S = 1000 × sigmoid((cp + CALIB) / SCALE)      SCALE = 310cp
```

`CALIB` được **hiệu chỉnh tự động** sao cho thế xuất phát ra đúng 505 điểm
(xem `scoring.calibrate`). Nhờ vậy khi đổi hàm lượng giá — thủ công hay mạng
nơ-ron, nông hay sâu — mốc 505 vẫn giữ nguyên, và điểm số của hai cấu hình
khác nhau vẫn so sánh được với nhau.

Vùng chiếu hết được tách riêng ở `[991, 1000]` và `[0, 9]`, còn thế cờ thường
bị kẹp trong `[10, 990]`. Do đó **chiếu hết ép buộc sau 10 nước vẫn luôn được
chấm cao hơn mọi ưu thế vật chất**, dù ưu thế đó lớn đến đâu — đúng như thực
tế chơi cờ.

---

## 2. Bốn tầng của hệ thống

```
main.py            giao diện: FEN -> điểm 0..1000
  └─ scoring.py    cp  ->  0..1000  (sigmoid + hiệu chỉnh + vùng chiếu hết)
  └─ search.py     alpha-beta: "đánh giá sâu" thật sự nằm ở đây
       └─ nnue.py       mạng nơ-ron (mặc định khi có trọng số)
       └─ evaluate.py   lượng giá thủ công PeSTO (mặc định khi chưa huấn luyện)
            └─ chess_core.py   luật cờ, bitboard, sinh nước đi
```

### `chess_core.py` — luật cờ
Bàn cờ biểu diễn bằng **bitboard** (số nguyên 64 bit, 1 bit = 1 ô). Quân trượt
(tượng/xe/hậu) dùng bảng tia dựng sẵn cộng thủ thuật "chặn đầu tiên" qua
`lsb`/`msb`.

Đã cài đặt đầy đủ: **nhập thành** (cả điều kiện ô đi qua bị kiểm soát và mất
quyền khi xe bị ăn), **bắt tốt qua đường**, **phong cấp** (4 lựa chọn),
**luật 50 nước**, **lặp 3 lần / thủ hoà**, **pat**, **thiếu lực chiếu hết**.
Ba luật cuối cần lịch sử ván nên nằm ở lớp `Game`.

Độ đúng được kiểm chứng bằng **perft** trên 6 thế cờ chuẩn tới độ sâu 4 —
đây là phép thử duy nhất bắt được lỗi sinh nước đi ở các trường hợp biên.

### `evaluate.py` — lượng giá thủ công
Bảng PeSTO (giá trị quân + bảng vị trí ô, nội suy giữa trung cuộc và tàn cuộc)
cộng cơ động, cặp tượng, tốt chồng/cô lập/thông, xe cột mở, áp lực quanh vua.
Dùng để sinh dữ liệu huấn luyện ban đầu và làm mốc so sánh.

### `search.py` — đánh giá sâu
Negamax + alpha-beta, bảng chuyển vị (Zobrist), sắp xếp nước đi (MVV-LVA,
killer, history), cắt tỉa null-move, LMR, cửa sổ kỳ vọng, và **quiescence** —
chỉ dừng ở thế "yên tĩnh" nên không bị đánh lừa bởi chuỗi ăn quân dở dang.

> Hàm lượng giá nhìn thế cờ **tĩnh**. Tìm kiếm mới cho biết điều gì thực sự
> xảy ra. Đây là lý do một engine với eval đơn giản + tìm kiếm sâu vẫn mạnh
> hơn nhiều so với eval tinh vi + tìm kiếm nông.

### `nnue.py` — mạng nơ-ron
Bộ đặc trưng kiểu **HalfKP rút gọn**: mỗi bên mã hoá thế cờ theo góc nhìn
của mình, chỉ số = (nhóm ô vua) × (loại quân tương đối) × (ô của quân).

```
640 đặc trưng thưa ─┬─> 256 (góc nhìn Trắng) ─┐
                    └─> 256 (góc nhìn Đen)  ──┴─> 512 -> 32 -> 32 -> 1
```

Số đặc trưng do `KING_BUCKETS` quyết định (640 / 2560 / 40960). Mặc định là
640 — xem mục 5 để biết vì sao mạng nhỏ lại thắng ở quy mô dữ liệu hiện tại.

Toạ độ được lật dọc cho Đen nên mạng học **một** biểu diễn dùng chung cho cả
hai màu — giảm nửa số mẫu cần thiết và ép tính đối xứng đúng theo luật cờ.

Huấn luyện bằng PyTorch, suy luận bằng NumPy. Đo trên máy rảnh:

| thao tác | thời gian |
|---|---|
| sinh nước đi | 25 µs |
| lượng giá thủ công | 95 µs |
| **lượng giá NNUE** | **97 µs** |

Tức mạng nơ-ron **gần như miễn phí** so với eval thủ công, vì trích đặc trưng
đã tối ưu (16 µs) và các lớp dense đủ nhỏ để chi phí chủ yếu là overhead gọi
NumPy chứ không phải phép nhân.

---

## 3. Học máy: học cái gì?

Điểm mấu chốt là **nhãn**. Mạng không học "centipawn", nó học **xác suất thắng**:

```
target = 0.7 × sigmoid(điểm_tìm_kiếm / 310)  +  0.3 × kết_quả_ván_thật
```

- **Điểm tìm kiếm** cho tín hiệu dày và ít nhiễu ở mọi thế cờ.
- **Kết quả ván** là sự thật cuối cùng, chống việc mạng chỉ sao chép lại đúng
  hàm lượng giá thủ công đã dùng để sinh dữ liệu — nếu chỉ học từ eval thủ
  công thì trần chất lượng chính là eval thủ công.

Học trong không gian `[0,1]` còn có tác dụng quan trọng: sai 200cp ở thế đã
thắng chắc gần như không bị phạt, còn sai 50cp ở thế cân bằng thì bị phạt
nặng — đúng chỗ mà độ chính xác thực sự quyết định ván cờ.

---

## 4. Quy trình chạy

```bash
# 0) môi trường (chỉ cần torch để huấn luyện)
python3 -m venv .venv && .venv/bin/pip install torch numpy

# 1) chấm điểm ngay bằng lượng giá thủ công — không cần huấn luyện gì
python main.py "<FEN>" --depth 10
python main.py --moves e2e4 e7e5 g1f3        # đi từ thế xuất phát
python main.py --interactive                  # nhập nước đi liên tục
python main.py "<FEN>" -v                     # in từng vòng lặp sâu dần

# 2) sinh dữ liệu tự chơi (chạy được bao nhiêu tuỳ thời gian)
python datagen.py --games 8000 --nodes 3500 --workers 6

# 3) huấn luyện
.venv/bin/python train.py --data data/selfplay.txt --epochs 40

# 4) chấm điểm bằng mạng
.venv/bin/python main.py "<FEN>" --nnue weights/nnue.npz --depth 8

# 5) kiểm chứng mạng có THẬT SỰ mạnh hơn không
.venv/bin/python match.py --nnue weights/nnue.npz --games 40 --nodes 4000

# hoặc gộp bước 3+5 vào một lệnh: huấn luyện cả KB=1 và KB=4 rồi đấu cả hai
./retrain.sh

# kiểm thử toàn bộ (perft, luật cờ, thang điểm)
python test_engine.py
```

Bước 2 và 3 lặp lại được: dùng mạng vừa huấn luyện để sinh dữ liệu tốt hơn,
rồi huấn luyện lại. Đó chính là vòng lặp tự nâng cấp mà các engine NNUE dùng.

---

## 5. Kết quả đo thực tế

Đây là phần quan trọng nhất, và cũng là phần dễ tự lừa mình nhất.

**Loss huấn luyện giảm đẹp không có nghĩa là cờ đánh hay hơn.** Mạng đầu tiên
(20 nghìn thế cờ, 12 epoch) có val loss giảm đều từ 0.091 xuống 0.0069, trông
rất thuyết phục. Đem đấu thật với chính hàm lượng giá thủ công ở cùng số nút:

```
NNUE ghi 4.5/30 = 15.0%
Elo chênh: -301  (khoảng tin cậy 95%: -657 .. -166)
```

Thua 301 Elo. Chẩn đoán bằng tương quan với nhãn tìm kiếm trên tập giữ lại:

| hàm lượng giá | tương quan r | RMSE |
|---|---|---|
| PeSTO thủ công | **0.967** | 158cp |
| NNUE (34k thế cờ) | 0.856 | 319cp |

Không phải lỗi dấu hay lỗi góc nhìn — mọi phép thử vật chất đều đúng hướng.
Đơn giản là **655 nghìn tham số học từ 34 nghìn mẫu**: thiếu dữ liệu khoảng
hai bậc độ lớn.

Thu nhỏ mạng xuống 164 nghìn tham số (bảng ngay dưới) rồi đấu lại:

```
NNUE ghi 9.5/40 = 23.8%
Elo chênh: -203  (khoảng tin cậy 95%: -371 .. -93)
```

Khá hơn 98 Elo — đúng hướng mà chỉ số tương quan đã dự đoán, nên r trên tập
giữ lại là chỉ số thay thế đáng tin để dò kiến trúc mà không phải đấu hàng
trăm ván.

### Chọn kiến trúc bằng thực nghiệm

Cùng 34k thế cờ, kiểm trên tập giữ lại hoàn toàn ngoài mẫu:

| KING_BUCKETS | HIDDEN | tham số | r |
|---|---|---|---|
| **1** | **256** | 164k | **0.856** |
| 1 | 128 | 82k | 0.849 |
| 4 | 256 | 655k | 0.804 |
| 4 | 128 | 328k | 0.802 |

Ít dữ liệu thì mạng nhỏ **tổng quát tốt hơn**, dù khả năng biểu đạt kém hơn.
Vì vậy mặc định là `KING_BUCKETS=1`; đổi bằng biến môi trường
`NNUE_KING_BUCKETS` khi dữ liệu nhiều lên.

### Vì sao mạng CÓ THỂ vượt được eval thủ công

Nhãn đến từ **tìm kiếm** chứ không phải từ eval tĩnh. Nghĩa là mạng đang
chưng cất kết quả của một cuộc tìm kiếm vài nghìn nút vào một hàm chỉ tốn
một lần chạy. Một lần gọi NNUE khi đó xấp xỉ được thứ mà eval thủ công phải
tìm kiếm sâu mới thấy — đó chính là chỗ mà học máy thắng, và nó chỉ xuất hiện
khi có đủ dữ liệu.

---

## 6. Giới hạn cần biết

- **Tốc độ.** Python thuần chạy ~30–46 nghìn nút/giây, chậm hơn engine C
  khoảng 1000 lần. Độ sâu 8–10 là thực tế; Stockfish đạt 25+.
- **Lượng dữ liệu là nút thắt.** Xem mục 5: ở 34k thế cờ mạng còn kém eval
  thủ công 301 Elo. Sinh dữ liệu chạy ~18 thế cờ/giây trên 5 nhân, tức khoảng
  66 nghìn thế cờ mỗi giờ. Cần để chạy vài giờ đến qua đêm trước khi mạng có
  cơ hội thắng. **Luôn xác nhận bằng `match.py`, đừng tin loss huấn luyện.**
- **Điểm số phụ thuộc độ sâu.** Cùng thế cờ, độ sâu khác nhau cho điểm khác
  nhau. Việc hiệu chỉnh giữ mốc 505 cố định nhưng không xoá được sự thật là
  đánh giá sâu hơn thì chính xác hơn.

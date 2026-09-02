#!/bin/bash
# Khởi động bàn phân tích cờ vua.
#
#   ./serve.sh                          # tự chọn trọng số mới nhất
#   ./serve.sh --nnue weights/x.npz     # chỉ định trọng số
#   ./serve.sh --hand                   # chỉ dùng hàm lượng giá thủ công
#   ./serve.sh --port 8080 --depth 10
#   ./serve.sh --stop
set -u
cd "$(dirname "$0")"

PORT=8000; DEPTH=8; NNUE=""; HAND=0; STOP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port)  PORT="$2"; shift 2 ;;
    --depth) DEPTH="$2"; shift 2 ;;
    --nnue)  NNUE="$2"; shift 2 ;;
    --hand)  HAND=1; shift ;;
    --stop)  STOP=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "Tham số lạ: $1"; exit 1 ;;
  esac
done

# Mẫu ngoặc vuông để pkill KHÔNG khớp chính dòng lệnh đang chạy nó -
# nếu không, script tự giết mình trước khi kịp làm gì.
pkill -f '[s]erver.py' 2>/dev/null && echo "Đã dừng máy chủ cũ."
[ "$STOP" = 1 ] && exit 0

PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"

if [ "$HAND" = 0 ] && [ -z "$NNUE" ]; then
  NNUE=$(ls -t weights/*.npz 2>/dev/null | head -1)
fi

ARGS=(--port "$PORT" --depth "$DEPTH")
if [ "$HAND" = 0 ] && [ -n "$NNUE" ]; then
  [ -f "$NNUE" ] || { echo "Không thấy trọng số: $NNUE"; exit 1; }
  # Số nhóm vua phải khớp giữa lúc train và lúc chạy, nếu lệch mạng sẽ cho ra
  # số vô nghĩa. Đọc thẳng từ file thay vì bắt người dùng nhớ.
  KB=$("$PY" -c "import numpy,sys; print(int(numpy.load(sys.argv[1])['king_buckets']))" "$NNUE" 2>/dev/null)
  HID=$("$PY" -c "import numpy,sys; print(int(numpy.load(sys.argv[1])['hidden']))" "$NNUE" 2>/dev/null)
  if [ -n "$KB" ]; then
    export NNUE_KING_BUCKETS="$KB"
    [ -n "$HID" ] && export NNUE_HIDDEN="$HID"
    echo "Mạng: $NNUE  (KING_BUCKETS=$KB, HIDDEN=${HID:-256})"
  fi
  ARGS+=(--nnue "$NNUE")
else
  echo "Chỉ dùng hàm lượng giá thủ công (PeSTO)."
fi

mkdir -p logs
echo "Đang hiệu chỉnh mốc 505 ở độ sâu $DEPTH..."
( nohup "$PY" -u server.py "${ARGS[@]}" > logs/server.log 2>&1 & )

for i in $(seq 1 60); do
  sleep 1
  if curl -s -o /dev/null -m 1 "http://localhost:$PORT/" 2>/dev/null; then
    echo
    sed -n 's/^  \(.*hiệu chỉnh.*\)$/  \1/p' logs/server.log
    echo "  Bàn phân tích: http://localhost:$PORT"
    command -v open >/dev/null && open "http://localhost:$PORT"
    echo "  Dừng: ./serve.sh --stop     Nhật ký: logs/server.log"
    exit 0
  fi
  pgrep -f '[s]erver.py' > /dev/null || { echo "Máy chủ chết khi khởi động:"; tail -15 logs/server.log; exit 1; }
done
echo "Quá hạn chờ. Xem logs/server.log"; exit 1

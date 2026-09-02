#!/bin/bash
# Huấn luyện lại và kiểm chứng bằng đấu thật.
# Dùng: ./retrain.sh [king_buckets]   (mặc định thử cả 1 và 4)
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
N=$(wc -l < data/selfplay.txt)
echo "=== Dữ liệu: $N thế cờ"
echo

BUCKETS="${1:-1 4}"
for KB in $BUCKETS; do
  echo "--- Huấn luyện KING_BUCKETS=$KB"
  NNUE_KING_BUCKETS=$KB $PY train.py --data data/selfplay.txt \
      --epochs 40 --out weights/nnue_kb$KB.npz | tail -4
  echo
done

echo "=== Đấu với hàm lượng giá thủ công (cùng số nút)"
for KB in $BUCKETS; do
  echo "--- KING_BUCKETS=$KB"
  NNUE_KING_BUCKETS=$KB $PY match.py --nnue weights/nnue_kb$KB.npz \
      --games 30 --nodes 4000 2>/dev/null | tail -2
  echo
done

echo "Elo DƯƠNG nghĩa là mạng đã thắng eval thủ công."
echo "Chọn bản tốt nhất rồi:  cp weights/nnue_kbN.npz weights/nnue.npz"

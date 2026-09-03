#!/bin/bash
# Sinh thêm dữ liệu tới 15 triệu thế cờ (hiện có 7,91 triệu). Mục tiêu này là
# mức cần để mạng KB=64 (đang overfit +235%) có đủ dữ liệu lấp đầy dung lượng.
# Tự khởi động lại nếu bị giết, tự dừng nếu đĩa cạn hoặc đạt mục tiêu.
cd "$(dirname "$0")"
TARGET=15000000
MIN_DISK_MB=3000
OUT=data/sf.txt.gz
LOG=logs/moredata.log

count() { .venv/bin/python -c "
import gzip
print(sum(1 for _ in gzip.open('$OUT','rt')))" 2>/dev/null || echo 0; }

for attempt in $(seq 1 500); do
  n=$(count)
  echo "$(date '+%m-%d %H:%M') lan $attempt | dang co ${n} the co | dia $(df -m . | tail -1 | awk '{print $4}')MB" >> "$LOG"
  [ "$n" -ge "$TARGET" ] && { echo "$(date '+%H:%M') DAT MUC TIEU ${n}" >> "$LOG"; break; }
  free=$(df -m . | tail -1 | awk '{print $4}')
  if [ "$free" -lt "$MIN_DISK_MB" ]; then
    echo "$(date '+%H:%M') DUNG: dia con ${free}MB" >> "$LOG"
    break
  fi
  remaining=$(( TARGET - n ))
  # ước lượng số ván cần (mỗi ván cho ~30-60 mẫu sau lọc nhiễu)
  games=$(( remaining / 20 + 5000 ))
  .venv/bin/python -u datagen_sf.py --games "$games" --depth 9 --workers 5 \
      --chunk 150 --seed $(( RANDOM * RANDOM + attempt )) \
      --epd data/book.epd --pgn data/pgn/kinh_dien.pgn \
      --out "$OUT" >> "$LOG" 2>&1
  pkill -9 stockfish 2>/dev/null
  sleep 15
done

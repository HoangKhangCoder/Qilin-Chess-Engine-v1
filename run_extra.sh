#!/bin/bash
# Thêm 100 ván Elo 1900 (mức mới) và 100 ván Elo 2000 (cộng dồn vào 100 ván
# của thang chính -> 200 ván ở mức quan trọng nhất).
#
# --seed 777 khác với thang chính (seed 1) nên bốc bộ khai cuộc KHÁC, tránh
# lặp lại cùng những thế cờ đã đấu.
cd "$(dirname "$0")"
export NNUE_KING_BUCKETS=64
MIN_DISK_MB=700
count() { grep -c "^\[Event " "$1" 2>/dev/null || echo 0; }

for elo in 1900 2000; do
  out="games/extra_elo${elo}.pgn"
  for attempt in $(seq 1 200); do
    free=$(df -m . | tail -1 | awk '{print $4}')
    [ "$free" -lt "$MIN_DISK_MB" ] && { echo "$(date '+%H:%M') extra DUNG: dia ${free}MB" >> logs/heartbeat.log; exit 1; }
    n=$(count "$out")
    echo "$(date '+%m-%d %H:%M') extra-$elo lan $attempt | $n/100 van | dia ${free}MB" >> logs/heartbeat.log
    [ "$n" -ge 100 ] && break
    .venv/bin/python -u play_stockfish.py \
        --games 100 --sf-elo $elo --time 2.0 --workers 2 --seed 777 \
        --nnue weights/cap_kb64_h256.npz --book data/book.epd \
        --out games/extra.pgn >> logs/extra.log 2>&1
    sleep 10
  done
  echo "$(date '+%H:%M') extra-$elo XONG: $(count "$out")/100" >> logs/heartbeat.log
done

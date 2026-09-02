#!/bin/bash
# 400 ván thang Elo (100 ván mỗi mức 1600/1800/2000/2200), 2 giây/nước cả hai bên.
#
# Ba lớp bảo vệ, vì tiến trình dài đã bị giết hai lần trên máy này:
#  1. vòng lặp tự chạy lại; play_stockfish.py đếm ván trong PGN để tiếp đúng chỗ
#  2. chốt chặn đĩa - swap macOS phình theo áp lực bộ nhớ, đầy đĩa giữa lúc ghi
#     sẽ hỏng PGN
#  3. dọn Stockfish mồ côi sau mỗi lần thoát, nếu không chúng ăn CPU vô ích
cd "$(dirname "$0")"
export NNUE_KING_BUCKETS=64
MIN_DISK_MB=700
TARGET=400
count() { cat games/ladder_elo*.pgn 2>/dev/null | grep -c "^\[Event "; }

for attempt in $(seq 1 500); do
  free=$(df -m . | tail -1 | awk '{print $4}')
  if [ "$free" -lt "$MIN_DISK_MB" ]; then
    echo "$(date '+%m-%d %H:%M') DUNG: dia con ${free}MB" >> logs/ladder.log
    break
  fi
  echo "$(date '+%m-%d %H:%M') lan $attempt | $(count)/$TARGET van | dia ${free}MB" >> logs/heartbeat.log
  .venv/bin/python -u play_stockfish.py \
      --games $TARGET --ladder 1600,1800,2000,2200 --time 2.0 --workers 3 \
      --nnue weights/cap_kb64_h256.npz --book data/book.epd \
      --out games/ladder.pgn >> logs/ladder.log 2>&1
  pkill -9 stockfish 2>/dev/null
  [ "$(count)" -ge "$TARGET" ] && { echo "HOAN TAT $(date '+%m-%d %H:%M')" >> logs/heartbeat.log; break; }
  sleep 15
done

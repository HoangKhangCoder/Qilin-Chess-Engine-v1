#!/bin/bash
# Chạy/chạy tiếp sinh dữ liệu. Gọi lại bao nhiêu lần cũng được - dữ liệu cộng dồn.
cd "$(dirname "$0")"
mkdir -p logs data
TARGET=1430000
N=$(wc -l < data/selfplay.txt 2>/dev/null || echo 0)

if pgrep -f "datagen.py --games" > /dev/null; then
  echo "Đang chạy sẵn rồi (pid $(pgrep -f 'datagen.py --games'))."
else
  nohup python3 datagen.py --games 200000 --nodes 3500 --workers 7 \
        --seed $RANDOM --out data/selfplay.txt >> logs/datagen.log 2>&1 &
  disown
  echo "Đã khởi động datagen."
fi

pgrep -x caffeinate > /dev/null || { nohup caffeinate -i -m -s >/dev/null 2>&1 & disown; }

echo "Hiện có : $N / $TARGET thế cờ  ($(( N * 100 / TARGET ))%)"
echo "Còn cần : $(( (TARGET - N) / 27 / 3600 )) giờ máy THỨC (ở ~27 thế cờ/giây)"
echo
echo "Xem tiến độ : wc -l data/selfplay.txt"
echo "Dừng lại    : pkill -f 'datagen.py --games'"
echo "Khi đủ      : ./retrain.sh"

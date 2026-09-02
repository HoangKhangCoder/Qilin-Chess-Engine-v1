"""Huấn luyện mạng NNUE bằng PyTorch.

Nhãn học là XÁC SUẤT THẮNG chứ không phải centipawn:

    target = LAMBDA * sigmoid(cp_tìm_kiếm / SCALE) + (1 - LAMBDA) * kết_quả_ván
    pred   = sigmoid(mạng(x) * CP_SCALE / SCALE)
    loss   = MSE(pred, target)

Học trong không gian [0,1] khiến sai số ở thế cờ đã ngã ngũ (±2000cp) không lấn
át thế cờ cân bằng - đúng chỗ mà độ chính xác thực sự quan trọng. Đây cũng là
lý do thang 0..1000 ở scoring.py ánh xạ trực tiếp được sang đầu ra của mạng.

    python train.py --data data/sf.txt --epochs 30 --out weights/nnue.npz
"""

import argparse
import math
import os
import random
import time

import gzip
import json

import numpy as np
import torch
import torch.nn as nn

from chess_core import Position
from nnue import build_torch_model, export_npz, features, CP_SCALE, NUM_FEATURES, KING_BUCKETS
from scoring import SCALE

LAMBDA = 0.7        # mặc định; đổi bằng --lambda


def _open(path):
    """Đọc được cả file thường lẫn .gz. Dữ liệu huấn luyện nén còn ~20%,
    và việc giải nén nhanh hơn nhiều so với thời gian phân tích FEN."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _cache_dir(path):
    return "{}.feat_kb{}".format(path, KING_BUCKETS)


def _build_cache(path, cache):
    """Hai lượt: lượt 1 đếm kích thước, lượt 2 ghi thẳng ra đĩa.

    Không giữ toàn bộ chỉ số trong RAM. Ở mức 15 triệu thế cờ, mảng chỉ số là
    ~3,6 GB và đỉnh lúc copy gấp đôi - quá sức máy 16 GB đang chạy song song
    7 worker sinh dữ liệu. Ghi ra đĩa rồi memmap thì RAM chỉ còn là page cache,
    hệ điều hành tự thu hồi khi thiếu.
    """
    os.makedirs(cache, exist_ok=True)
    n = tot_w = tot_b = 0
    t0 = time.time()
    with _open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if len(p) != 3:
                continue
            wi, bi = features(Position(p[0]))
            tot_w += len(wi)
            tot_b += len(bi)
            n += 1
            if n % 1000000 == 0:
                print("  đếm {:,} ({:.0f}s)".format(n, time.time() - t0), flush=True)
    print("  {:,} thế cờ, {:,} chỉ số ({:.0f}s)".format(n, tot_w + tot_b, time.time() - t0),
          flush=True)

    idt = np.int16 if NUM_FEATURES <= 32767 else np.int32
    mm = {
        "idx_w": np.lib.format.open_memmap(cache + "/idx_w.npy", "w+", idt, (tot_w,)),
        "idx_b": np.lib.format.open_memmap(cache + "/idx_b.npy", "w+", idt, (tot_b,)),
        "off_w": np.lib.format.open_memmap(cache + "/off_w.npy", "w+", np.int64, (n + 1,)),
        "off_b": np.lib.format.open_memmap(cache + "/off_b.npy", "w+", np.int64, (n + 1,)),
        "stm": np.lib.format.open_memmap(cache + "/stm.npy", "w+", np.float32, (n,)),
        "cp": np.lib.format.open_memmap(cache + "/cp.npy", "w+", np.float32, (n,)),
        "res": np.lib.format.open_memmap(cache + "/res.npy", "w+", np.float32, (n,)),
    }
    i = pw = pb = 0
    mm["off_w"][0] = 0
    mm["off_b"][0] = 0
    with _open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if len(p) != 3:
                continue
            pos = Position(p[0])
            wi, bi = features(pos)
            mm["idx_w"][pw:pw + len(wi)] = wi
            mm["idx_b"][pb:pb + len(bi)] = bi
            pw += len(wi)
            pb += len(bi)
            mm["off_w"][i + 1] = pw
            mm["off_b"][i + 1] = pb
            white = pos.side == 0
            mm["stm"][i] = 1.0 if white else 0.0
            cp = int(p[1])
            mm["cp"][i] = float(cp if white else -cp)
            r = float(p[2])
            mm["res"][i] = r if white else 1.0 - r
            i += 1
            if i % 1000000 == 0:
                print("  ghi {:,} ({:.0f}s)".format(i, time.time() - t0), flush=True)
    for a in mm.values():
        a.flush()
    with open(cache + "/meta.json", "w") as f:
        json.dump({"n": n, "src_mtime": os.path.getmtime(path),
                   "king_buckets": KING_BUCKETS}, f)
    return n


def load_dataset(path, limit=None, use_cache=True):
    """Nạp đặc trưng qua cache memmap. Chỉ số lưu int16 khi đủ chỗ (2560 < 32767)."""
    cache = _cache_dir(path)
    meta_p = cache + "/meta.json"
    fresh = False
    if use_cache and os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        fresh = (meta.get("src_mtime") == os.path.getmtime(path)
                 and meta.get("king_buckets") == KING_BUCKETS)
    if not fresh:
        print("Dựng cache đặc trưng ->", cache, flush=True)
        _build_cache(path, cache)
    n = json.load(open(meta_p))["n"]

    def mm(name):
        return np.load("{}/{}.npy".format(cache, name), mmap_mode="r")

    ds = {k: mm(k) for k in ("idx_w", "idx_b", "off_w", "off_b", "stm", "cp", "res")}
    ds["n"] = n if limit is None else min(n, limit)
    print("Cache: {:,} thế cờ, chỉ số {} ({:.1f} GB trên đĩa)".format(
        n, ds["idx_w"].dtype,
        sum(os.path.getsize("{}/{}.npy".format(cache, k))
            for k in ("idx_w", "idx_b", "off_w", "off_b")) / 1e9), flush=True)
    return ds


def slice_batch(ds, rows):
    """Cắt các hàng `rows` (numpy int64) từ cache memmap thành batch CSR."""
    r = rows.numpy() if hasattr(rows, "numpy") else rows
    ow, ob = ds["off_w"], ds["off_b"]
    sw, lw = ow[r], ow[r + 1] - ow[r]
    sb, lb = ob[r], ob[r + 1] - ob[r]

    def gather(idx, starts, lens):
        total = int(lens.sum())
        offs = np.empty(len(lens) + 1, dtype=np.int64)
        offs[0] = 0
        np.cumsum(lens, out=offs[1:])
        if total == 0:
            return torch.zeros(0, dtype=torch.int32), torch.from_numpy(offs)
        pos = np.arange(total, dtype=np.int64) - np.repeat(offs[:-1], lens)
        flat = np.asarray(idx[np.repeat(starts, lens) + pos], dtype=np.int32)
        return torch.from_numpy(flat), torch.from_numpy(offs)

    fw, nw = gather(ds["idx_w"], sw, lw)
    fb, nb = gather(ds["idx_b"], sb, lb)
    t = lambda k: torch.from_numpy(np.asarray(ds[k][r])).unsqueeze(1)
    return fw, nw, fb, nb, t("stm"), t("cp"), t("res")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sf.txt")
    ap.add_argument("--out", default="weights/nnue.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234,
                    help="Cố định để hai lần train khác kiến trúc dùng CHUNG "
                         "tập val - nếu không, chênh lệch đo được có thể chỉ "
                         "là do chia tập khác nhau.")
    ap.add_argument("--lambda", dest="lam", type=float, default=LAMBDA,
                    help="1.0 = chỉ bắt chước cp của Stockfish; "
                         "0.0 = chỉ học kết quả ván")
    ap.add_argument("--resume", default=None, help="tiếp tục từ checkpoint .pt")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"],
                    help="Mặc định CPU: mạng NNUE quá nhỏ nên chi phí khởi chạy "
                         "kernel GPU lấn át tính toán - đo được CPU nhanh gấp đôi MPS.")
    args = ap.parse_args()

    print("Bộ đặc trưng: {} chiều ({} nhóm vua)".format(NUM_FEATURES, KING_BUCKETS))
    t0 = time.time()
    ds = load_dataset(args.data, args.limit)
    n = ds["n"]
    print("Đã nạp {:,} thế cờ trong {:.1f}s".format(n, time.time() - t0), flush=True)
    if n < 200:
        raise SystemExit("Quá ít dữ liệu. Chạy datagen_sf.py trước.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    perm = torch.randperm(n).numpy()
    n_val = max(64, int(n * args.val_frac))
    val_rows, train_rows = perm[:n_val], perm[n_val:]

    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        dev = args.device
    print("Thiết bị:", dev, flush=True)
    model = build_torch_model()
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print("Đã nạp lại", args.resume)
    model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    k = CP_SCALE / SCALE

    def run(rows, train):
        model.train(train)
        total, count = 0.0, 0
        order = rows[torch.randperm(len(rows)).numpy()] if train else rows
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            fw, nw, fb, nb, stm, cp, res = slice_batch(ds, b)
            fw, nw, fb, nb = fw.to(dev), nw.to(dev), fb.to(dev), nb.to(dev)
            stm, cp, res = stm.to(dev), cp.to(dev), res.to(dev)
            target = args.lam * torch.sigmoid(cp / SCALE) + (1 - args.lam) * res
            with torch.set_grad_enabled(train):
                out = model(fw, nw, fb, nb, stm)
                pred = torch.sigmoid(out * k)
                loss = ((pred - target) ** 2).mean()
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * len(b)
            count += len(b)
        return total / max(count, 1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        ep_t0 = time.time()
        tr = run(train_rows, True)
        va = run(val_rows, False)
        sched.step()
        # sai số quy đổi ra "điểm trên thang 1000" cho dễ hình dung
        pts = math.sqrt(va) * 1000
        flag = ""
        if va < best:
            best = va
            model_cpu = model.to("cpu")
            export_npz(model_cpu, args.out)
            torch.save(model_cpu.state_dict(), args.out.replace(".npz", ".pt"))
            model.to(dev)
            flag = "  <- đã lưu"
        print("epoch {:>3}  train {:.5f}  val {:.5f}  (~±{:.0f} điểm/1000)  {:.0f}s{}".format(
            ep, tr, va, pts, time.time() - ep_t0, flag), flush=True)
    print("Trọng số:", args.out)


if __name__ == "__main__":
    main()

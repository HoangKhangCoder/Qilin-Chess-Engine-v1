# Qilin v1 — chess position scorer on a 0–1000 scale

Given any position, answer one question: **how many points out of 1000 is
White getting?**

```bash
./serve.sh                                                   # web analysis board
python3 main.py "<FEN>" --nnue weights/cap_kb64_h256.npz --depth 8
```

## Invariant that must never break

**The real engine must NEVER depend on an external chess library.** No
python-chess, no Stockfish, no chess library of any kind. Only the training
pipeline is allowed to use them.

```
CLEAN (engine)    chess_core.py  search.py  evaluate.py  scoring.py  nnue.py
                  main.py  server.py  test_engine.py
ALLOWED (train)   datagen_sf.py  make_book.py  train.py  match.py
                  play_stockfish.py  compare_evals.py  datagen.py
```

This boundary is **enforced by code**, not convention:

```bash
python3 check_purity.py   # scans imports + loads the engine with `chess` blocked
```

Run it after every change touching an engine file. It walks the AST to scan
only strings that actually **execute** in the code, skipping docstrings and
comments (to avoid false positives when a file merely *mentions* Stockfish in
its documentation) — then it actually boots the engine with `chess` blocked
at `sys.meta_path`, so hidden dependencies get exposed too.

## The scoring scale

Meaning: **S = 1000 × White's expected game outcome.**

| Score | Meaning |
|---|---|
| 1000 | White mates on this exact move |
| 991–999 | White has a forced mate (991 = furthest out) |
| 505 | **Starting position** — White's first-move advantage is worth 5 points |
| 500 | Perfectly balanced, or already drawn by rule |
| 0 | Black mates on this exact move |

The 505 mark is **not an arbitrary additive constant**.
`scoring.calibrate(cp_starting_position)` sets an internal offset inside the
sigmoid so that whichever evaluation function is in use, scoring the
starting position, yields exactly 505. That's why swapping from the
handcrafted evaluator to NNUE keeps the anchor fixed. The calibration result
is cached in `.calib.json`; delete that file or pass `--recalibrate` when
you change the evaluation function.

The non-mate range is clamped to [10, 990] so it never touches the mate
range — a forced mate always outranks any material advantage, however large.

## Testing

```bash
python3 test_engine.py     # 86 tests: perft, FEN, Zobrist, draw rules, scoring, search
python3 check_purity.py    # external-library boundary
```

`test_engine.py` must pass 100% before every commit. It covers 14 groups:
perft on 6 standard positions, make/unmake state restoration, incremental
Zobrist matching a from-scratch recompute, mate-in-one, forced mate,
scoring scale, color symmetry, castling, en passant, the 50-move rule,
threefold repetition, stalemate/insufficient material, promotion, and
**a search aborted mid-way must still return a legal move**.

Cross-validated independently against python-chess (55k+ positions, exact
match) — rerun whenever `chess_core.py` changes.

## Traps hit before — don't repeat them

**SearchAbort skipping unmake_move.** When time/nodes run out mid-search,
`SearchAbort` propagates through the recursive `negamax` frames, causing
`unmake_move` to be skipped — the internal board (`work`) gets stuck mid-way
through a trial move, sometimes with the wrong side to move. `extract_pv`
then reads the wrong position and `search()` could return an **illegal**
move. Fixed by restoring `work = pos.copy()` when catching `SearchAbort`,
plus a final safety net: if the returned move isn't in the legal list, fall
back to `legal[0]`. This bug had silently corrupted every `match.py` /
`play_stockfish.py` result, since they played the returned move straight
onto the board with no re-validation — both now check legality before
`push()`.

**50-move clock and promotion.** In `make_move`, the `piece` variable gets
reassigned to the new piece on promotion. Must use the `moved_type` saved
beforehand, otherwise a non-capturing promotion gets counted as a quiet
move. Perft can't catch this since it only counts moves.

**En passant square.** Only record it when the capture is *actually legal*
(`_has_legal_ep`). Recording it unconditionally lets a "ghost" ep square
leak into the Zobrist key, making two identical positions hash differently
and breaking threefold-repetition detection. `set_fen` must filter it too,
since many external FENs write it unconditionally.

**`NNUE_KING_BUCKETS` must match between training and inference.**
`nnue.py` checks this and raises a clear error on mismatch — don't remove
that check; a silent mismatch produces meaningless numbers. `serve.sh` reads
this value straight from the `.npz` file so you never set it by hand.

**stdout buffering.** When logging to a file, `print` doesn't auto-flush.
Run python with `-u`, otherwise you'll think the process hung while it's
running fine.

**CPU is faster than MPS for this network.** Measured 28ms/batch on CPU vs
58ms on MPS — the network is small enough that GPU kernel-launch overhead
dominates the actual compute. `train.py --device` defaulting to `cpu` is
deliberate.

**Data-generation chunk size.** A small `datagen_sf.py --chunk` makes
Stockfish reload its ~50MB NNUE network constantly. Raising it from 8 to 150
gives 2.3× the throughput. Data is still flushed per-game, so a large chunk
costs nothing but progress-line smoothness.

**`pkill -f <script_name>` kills the very script calling it.** The shell's
own command line contains the script name string, so `pkill -f run_x.sh`
inside `run_x.sh` kills the parent process before it can even log anything.
Use the bracket trick `pkill -f '[r]un_x.sh'` to exclude itself. macOS also
has no `setsid`; detach background processes with `( nohup ... & )` inside a
subshell instead.

## Training pipeline

```bash
python3 make_book.py --count 8000 --workers 7               # MultiPV opening book
python3 datagen_sf.py --games 400000 --epd data/book.epd \
    --out data/sf.txt.gz                                     # Stockfish-labeled data
NNUE_KING_BUCKETS=64 python3 -u train.py --data data/sf.txt.gz --lambda 1.0
python3 match.py --nnue weights/x.npz --games 100            # head-to-head
python3 play_stockfish.py --games 100 --ladder 1600,1800,2000,2200 \
    --time 2.0 --nnue weights/x.npz                          # real-Stockfish Elo ladder
```

Each data line: `FEN | Stockfish cp (White's perspective) | game result`.

`train.py` and `datagen_sf.py` transparently read/write both `.txt` and
`.txt.gz`. Compressed data is ~20% the size (456 MB → 95 MB) and decompresses
far faster than FEN parsing takes, so always use `.gz`. gzip supports
appending blocks, so you can still keep writing into an already-compressed
file.

The two label sources are **deliberately independent**: `cp` gives a dense,
low-noise signal; `result` gives ground truth. If both came from Stockfish,
the network would just copy Stockfish with nothing left to surpass.
`--lambda` controls the mixing ratio (1.0 = pure cp imitation).

`train.py` caches features into a **memmap directory** `<data>.feat_kb<N>/`
(not a single `.npz` file) — indices stored as `int16` when they fit,
written straight to disk in two passes (count, then write) so nothing needs
to stay fully in RAM. The cache is keyed by `KING_BUCKETS`; subsequent
training runs load in seconds instead of minutes.

`--seed` (default 1234) fixes the train/val split order — **must stay the
same** when comparing two different architectures, otherwise a measured gap
could just be due to a different split rather than the architecture itself.

## Measurement: be careful with the measurement itself

Falling loss does **not** guarantee stronger play. Bitten by this twice:

1. Trained a network on a mixed target `0.7×sigmoid(cp) + 0.3×result`, then
   scored it against pure cp — the 30% weight pulls the network away from
   exactly what's being measured.
2. The network matches Stockfish's labels **16% better** than the
   handcrafted evaluator (measured by correlation) but performs **no better**
   in a same-node match — raw accuracy doesn't automatically translate to
   playing strength, because alpha-beta only needs the correct *relative
   ordering* between moves. This is evidence for switching the loss function
   to rank-based learning instead of pure centipawn regression.

Reliable measures, in order of trust:

1. `play_stockfish.py --ladder` — real games against Elo-limited Stockfish,
   with paired openings (each opening played twice, colors swapped) — most
   trustworthy, yields an absolute Elo number
2. `match.py` — head-to-head between two evaluators at the **same node
   budget or same time budget** (fair on compute)
3. Correlation with Stockfish's labels on a **held-out set deduplicated
   against the training set**
4. Val loss — only useful for detecting overfit, never for comparing two
   different training objectives

## Current results (reference — see README.md for the latest)

The network `weights/cap_kb64_h256.npz` (KING_BUCKETS=64, HIDDEN=256, 10.5M
parameters, trained on 7.91M positions) reaches roughly **2000–2070 Elo**
in real games against Stockfish 18 at limited strength. This network's
train/val gap is **+235%** — it's overfitting, and needs more data
(projected 15–20M positions) to make full use of its 10.5M parameters.

## File map

| File | Role |
|---|---|
| `chess_core.py` | Bitboards, move generation, make/unmake, FEN, Zobrist, `Game` (draw rules) |
| `search.py` | Negamax + alpha-beta, TT, quiescence, null-move, LMR |
| `evaluate.py` | Handcrafted PeSTO evaluation (tapered), comparison baseline |
| `scoring.py` | cp → 0..1000 mapping, 505-anchor calibration, mate range |
| `nnue.py` | Feature extraction, numpy inference, PyTorch model |
| `main.py` | CLI scorer, interactive mode |
| `server.py` + `web/` | Live web analysis board, best-move arrows, streaming |
| `serve.sh` | Launches the analysis board, auto-reads KING_BUCKETS/HIDDEN from the .npz |
| `test_engine.py` | 86 regression tests |
| `check_purity.py` | Enforces the external-library boundary |
| `datagen_sf.py` | Data generation, Stockfish labeling |
| `make_book.py` | Opening book via MultiPV |
| `train.py` | NNUE training |
| `match.py` | Head-to-head between two evaluators |
| `play_stockfish.py` | Real games against Stockfish, PGN export, Elo ladder |

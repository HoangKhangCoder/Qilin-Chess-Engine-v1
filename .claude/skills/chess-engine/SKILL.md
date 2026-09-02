---
name: chess-engine
description: Workflow for the Qilin chess engine (0-1000 position scorer) in this repo - generating data with Stockfish, training the NNUE, measuring whether the network is actually better, running the Elo ladder against real Stockfish, and keeping the engine's external-library boundary clean. Use when retraining the network, generating more data, evaluating evaluator quality, fixing chess rules in chess_core, or running head-to-head matches.
---

# Working with the Qilin chess engine

## Before touching anything

```bash
python3 test_engine.py && python3 check_purity.py
```

Both must pass. `test_engine.py` (86 tests) takes ~3 minutes (perft is most
of it).

## The boundary that must not break

The real engine (`chess_core` `search` `evaluate` `scoring` `nnue` `main`
`server` `test_engine`) **must never import python-chess or call
Stockfish**. Only `datagen_sf.py` `make_book.py` `train.py` `match.py`
`play_stockfish.py` `compare_evals.py` `datagen.py` are allowed to.

`check_purity.py` enforces this by walking the AST (skipping
docstrings/comments to avoid false positives) then loading the engine with
`chess` blocked at `sys.meta_path` — hidden dependencies get exposed too.
Run it after any change touching an engine file.

## Fixing chess rules in chess_core.py

Perft is **not enough** to catch bugs — it only counts moves, it doesn't
check the 50-move clock, the ep square, or the Zobrist key. Always
cross-check against python-chess:

```python
import chess
from chess_core import Game, move_str
# play a few hundred random games, at each position compare:
#   sorted(move_str(m) for m in g.pos.legal_moves()) == sorted(m.uci() for m in b.legal_moves)
#   g.pos.fen() == b.fen()
#   (in_check, is_checkmate, is_stalemate, repetitions>=3, halfmove>=100, insufficient)
```

This caught 2 bugs perft missed: the 50-move clock not resetting after
promotion, and an unconditionally-recorded ep square breaking
threefold-repetition detection.

If you touch `search.py`: watch the `SearchAbort` path — a timeout/node-limit
exception propagating through recursive `negamax` frames can skip
`unmake_move` and make `search()` return an illegal move. Fixed via a
`work.copy()` restore plus a final safety net, with its own regression test
(group 14 in `test_engine.py`). Any change to `negamax`/`quiesce`'s
early-exit path must rerun that test.

## Generating more data

```bash
# opening book (one-time, ~4.5 positions/sec with 7 workers)
python3 make_book.py --count 8000 --workers 7

# data generation — LARGE chunk, otherwise Stockfish keeps reloading its 50MB net
python3 datagen_sf.py --games 400000 --depth 9 --workers 7 --chunk 150 \
  --epd data/book.epd --out data/sf.txt.gz
```

Data **accumulates** into the file — rerun as many times as you like.
Workers flush per-game, so stopping with `pkill -f '[d]atagen_sf.py'` loses
nothing (bracket pattern so the command doesn't kill itself).

`.txt.gz` is read/written transparently and compresses to ~20% the size —
always prefer it over raw `.txt`.

Reference throughput: ~570k positions/hour with 7 workers at depth 9 on an
8-core machine, but **system memory pressure** (e.g. Chrome hogging RAM) can
inflate swap and drop that to a third — check `sysctl -n vm.swapusage` if
throughput looks abnormally low.

Sanity-check data after generating: 0 corrupt lines, under 1% duplicates,
and roughly 40% of samples with |cp|<100 (the balanced zone is where
accuracy actually decides games).

## Training

```bash
NNUE_KING_BUCKETS=64 python3 -u train.py --data data/sf.txt.gz \
  --epochs 40 --batch 8192 --device cpu --lambda 1.0 --out weights/x.npz
```

- `-u` is mandatory when logging to a file, otherwise it looks hung
- `--device cpu` is correct: measured CPU running twice as fast as MPS at
  this network size
- `NNUE_KING_BUCKETS` must exactly match whatever runs the engine afterward
  (`serve.sh` reads it straight from the `.npz`, so you never set it by
  hand)
- `--seed` (default 1234) must **stay fixed** when comparing two
  architectures on the same data, otherwise a measured gap could just be a
  different validation split
- The feature cache is a **memmap directory** `<data>.feat_kb<N>/`, not a
  single `.npz` file — the first run builds it (a few minutes for millions
  of positions), later runs load in seconds

Read the train/val gap to know what's needed:

| Signal | Meaning | What to do |
|---|---|---|
| val >> train | overfitting | add more data |
| val ≈ train, both high | underfitting | raise KING_BUCKETS or HIDDEN |
| val plateaus early | signal exhausted | more diverse data, not more of it |

Measured in practice: on the same 7.91M positions, raising `KING_BUCKETS`
from 4 to 64 cut val loss by 25% — the bottleneck used to be *network
capacity*, not *data volume*. But the KB=64 network now has a train/val gap
of +235% (heavy overfitting), so more data is needed again to fill that
larger capacity.

## Measuring whether the network is ACTUALLY better

Low val loss doesn't mean stronger play. In order of trust:

```bash
# 1. Real games vs Elo-limited Stockfish, paired openings — most trustworthy
python3 play_stockfish.py --games 100 --ladder 1600,1800,2000,2200 \
  --time 2.0 --nnue weights/x.npz --book data/book.epd

# 2. Head-to-head between two evaluators at the SAME node or SAME time budget
python3 match.py --nnue weights/x.npz --games 100 --time 2.0

# 3. Correlation with Stockfish on a held-out set (must dedupe vs training set!)
python3 datagen_sf.py --games 40 --depth 12 --seed 999999 --out data/_holdout.txt
# then remove every FEN already in data/sf.txt.gz before measuring
```

**Traps hit twice:**

1. Trained a network on a mixed target `λ·sigmoid(cp) + (1−λ)·result`, then
   scored it against pure cp. When comparing to cp labels, train with
   `--lambda 1.0`, otherwise you're penalizing the network for exactly what
   it was taught. Never compare val loss across two different `--lambda`
   values — different training objectives aren't on the same scale.
2. A network that matches Stockfish's cp 16% better than the handcrafted
   evaluator can still be **no better** in a real same-node match — raw
   accuracy doesn't automatically become playing strength. `match.py` /
   `play_stockfish.py` are the final judges, not loss.

## Scoring-scale invariants

After any change touching `scoring.py` or the evaluation function, verify:

```bash
python3 main.py --recalibrate --depth 8                    # must give 505
python3 main.py "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"        # must give 1000
python3 main.py "5k2/5P2/5K2/8/8/8/8/8 b - - 0 1"          # must give 500 (stalemate)
```

The 505 anchor comes from `scoring.calibrate()`, not an added constant.
Changing the evaluator requires `--recalibrate` or deleting `.calib.json`.

## Playing real Stockfish / the Elo ladder

```bash
python3 play_stockfish.py --games 100 --ladder 1600,1800,2000,2200 \
  --time 2.0 --nnue weights/x.npz --book data/book.epd --out games/ladder.pgn
```

- Openings are **paired**: each book position is played exactly twice with
  colors swapped — perfectly equal games per color, and much lower variance
  from lucky/unlucky opening draws
- `--time` applies to **both sides** — fair on compute
- Counts games already written to the PGN so it can **resume exactly where
  it stopped** if the process gets killed mid-run — always wrap long
  (many-hour) runs in a self-restarting supervisor loop
- Current result (`weights/cap_kb64_h256.npz`): roughly 2000–2070 Elo, see
  the "Current results" section in `CLAUDE.md`

## Limitations worth stating plainly

This engine runs ~16–80k nodes/sec (pure Python). Stockfish runs 10–50
**million**. That ~600× gap can't be closed by a better evaluation function
alone. On top of that, the training data is labeled by Stockfish itself, so
the theoretical ceiling is "approximates Stockfish, less accurately" —
unless you switch to a self-play loop using the trained network itself
(`datagen.py --nnue` is already wired up for this).

A measurable, meaningful goal: play against Elo-limited Stockfish
(`play_stockfish.py --ladder`) to find out how strong the engine actually
is, then push that number up. Don't aim for beating full-strength
Stockfish — measured 0/10 there already.

Two directions not yet done that could add Elo without any new data: SEE
(Static Exchange Evaluation, pruning only inside quiescence search and only
when not in check — so a sacrifice followed by a forcing check sequence
never gets wrongly discarded) and an incremental NNUE accumulator (currently
recomputed from scratch every call instead of updated in O(1) as pieces
move).

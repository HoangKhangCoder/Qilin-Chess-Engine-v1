# Qilin v1

A chess engine written from scratch (bitboards, alpha-beta, NNUE) — scores
any position on a **0–1000** scale and reaches roughly **2000–2070 Elo**,
measured by playing real games against a strength-limited Stockfish 18.

```bash
./serve.sh                                    # opens the analysis board at localhost:8000
python3 main.py "<FEN>" --nnue weights/cap_kb64_h256.npz --depth 8
```

---

## Table of Contents

1. [Why this project exists](#1-why-this-project-exists)
2. [What the 0–1000 scale means](#2-what-the-0–1000-scale-means)
3. [The four layers of the system](#3-the-four-layers-of-the-system)
4. [A clean boundary — enforced by code](#4-a-clean-boundary--enforced-by-code)
5. [Quick start](#5-quick-start)
6. [Training the NNUE](#6-training-the-nnue)
7. [Measured results](#7-measured-results)
8. [The web analysis board](#8-the-web-analysis-board)
9. [Testing](#9-testing)
10. [Limitations and what's next](#10-limitations-and-whats-next)
11. [File map](#11-file-map)

---

## 1. Why this project exists

Given any position, answer exactly one question: **how many points out of
1000 is White getting?** Every bit of chess rules, move generation, search,
and the evaluation network is written from scratch in pure Python — no
`python-chess`, no calling Stockfish at runtime. Stockfish only appears in
the **training** pipeline, playing the role of a teacher that labels data,
and that boundary is **enforced by code** (`check_purity.py`), not just
convention.

## 2. What the 0–1000 scale means

> **S = 1000 × White's expected game outcome**
> (win = 1, draw = 0.5, loss = 0)

| Score | Meaning |
|---|---|
| **1000** | White mates **on this exact move** |
| 991–999 | White has a forced mate |
| **505** | **Starting position** — White's first-move advantage |
| 500 | Perfectly balanced, or already drawn by rule |
| 1–9 | Black has a forced mate |
| **0** | Black mates on this exact move |

The 505 anchor is **not an arbitrary added constant**. `scoring.calibrate()`
sets an offset inside the sigmoid so that whichever evaluation function is
in use, scoring the starting position, yields exactly 505 points. That's
why swapping the handcrafted evaluator for NNUE keeps the anchor fixed, and
why scores from two different configurations remain comparable.

The non-mate range is clamped to `[10, 990]` so it never touches the mate
range — a forced mate always outranks any material advantage, however large.

## 3. The four layers of the system

```
main.py / server.py     interface: FEN -> 0..1000 score
  └─ scoring.py         cp -> 0..1000 (sigmoid + calibration + mate range)
  └─ search.py          alpha-beta: the real "deep evaluation" lives here
       └─ nnue.py            neural evaluator (default once weights exist)
       └─ evaluate.py        handcrafted PeSTO evaluator (comparison baseline)
            └─ chess_core.py     bitboards, chess rules, Zobrist, move generation
```

- **`chess_core.py`** — the board is represented as 64-bit bitboards.
  Fully implements castling, en passant, promotion, the 50-move rule,
  threefold repetition, stalemate, insufficient material. Cross-validated
  independently against `python-chess` on tens of thousands of random
  positions — exact match on legal moves, FEN, and every rule state.
- **`search.py`** — negamax + alpha-beta, Zobrist transposition table, move
  ordering (MVV-LVA, killers, history), null-move pruning, LMR, aspiration
  windows, quiescence search.
- **`evaluate.py`** — a PeSTO table (piece values + piece-square tables,
  tapered between middlegame/endgame) plus mobility, bishop pair, pawn
  structure, king safety.
- **`nnue.py`** — the NNUE network: sparse features
  `(king square bucket) × (relative piece type) × (square)` → an
  accumulator shared across both perspectives → two small dense layers.
  Trained with PyTorch, inference runs on pure NumPy (torch isn't needed to
  run the engine).

## 4. A clean boundary — enforced by code

```
CLEAN (engine)    chess_core.py  search.py  evaluate.py  scoring.py
                  nnue.py  main.py  server.py  test_engine.py
ALLOWED (train)   datagen_sf.py  make_book.py  train.py  match.py
                  play_stockfish.py
```

```bash
python3 check_purity.py
```

It doesn't just read `import` statements via AST (skipping
docstrings/comments to avoid false positives) — it also actually **boots
the engine with the `chess` module blocked at `sys.meta_path`**, so hidden
dependencies get exposed immediately.

## 5. Quick start

```bash
# web analysis board (recommended) — auto-picks the latest weights, opens the browser
./serve.sh
./serve.sh --hand                              # handcrafted evaluator only
./serve.sh --nnue weights/cap_kb64_h256.npz --port 8080

# command line
python3 main.py "<FEN>" --depth 10
python3 main.py --moves e2e4 e7e5 g1f3
python3 main.py --interactive
python3 main.py "<FEN>" --nnue weights/cap_kb64_h256.npz -v   # print each depth iteration
```

## 6. Training the NNUE

```bash
python3 make_book.py --count 8000 --workers 7               # MultiPV opening book
python3 datagen_sf.py --games 400000 --epd data/book.epd \
    --out data/sf.txt.gz                                    # Stockfish-labeled data
NNUE_KING_BUCKETS=64 python3 -u train.py --data data/sf.txt.gz --lambda 1.0
python3 match.py --nnue weights/x.npz --games 100            # NNUE vs handcrafted head-to-head
python3 play_stockfish.py --games 100 --ladder 1600,1800,2000,2200 \
    --time 2.0 --nnue weights/x.npz                          # Elo ladder vs real Stockfish
```

Each data line: `FEN | Stockfish cp (White's perspective) | game result`. The
two label sources are **deliberately independent** — `cp` gives a dense,
low-noise signal, `result` gives ground truth — so the network never simply
copies Stockfish.

`train.py` transparently reads/writes both `.txt` and `.txt.gz`, and caches
features to a disk memmap (not held fully in RAM), so loading tens of
millions of positions stays safe even on memory-constrained machines.

## 7. Measured results

**Elo ladder**, played against a strength-limited Stockfish 18
(`UCI_LimitStrength`), 2.0 seconds per move for both sides, paired openings
(each position played twice with colors swapped for perfect fairness):

| Opponent | Score | Rate | Estimated engine Elo |
|---|---|---|---|
| Stockfish (Elo 1600) | 85.0/100 | 85.0% | ≈ 1901 |
| Stockfish (Elo 1800) | 77.0/100 | 77.0% | ≈ 2010 |
| Stockfish (Elo 1900) | 69.0/100 | 69.0% | ≈ 2039 |
| Stockfish (Elo 2000) | 52.0/87 | 59.8% | ≈ 2069 |
| Stockfish (full strength) | 0.0/10 | 0% | swept |

**→ Estimated playing strength: roughly 2000–2070 Elo**, despite running in
pure Python at only ~16–80 thousand nodes/second (about 600× slower than
Stockfish).

**Training data:** 7,914,424 positions labeled by Stockfish 18, plus 8,338
opening positions from MultiPV.

**Architecture experiment** (same data, same validation split via a fixed
seed) showed the bottleneck used to be *network capacity*, not *data
volume* — doubling data on a small network improved nothing, but growing
network capacity on the same data improved clearly:

| Architecture | Parameters | Val loss | vs. baseline |
|---|---|---|---|
| KB=4, H=256 (baseline) | 0.7M | 0.00306 | — |
| KB=4, H=512 | 1.3M | 0.00271 | 11% better |
| **KB=64, H=256** (current) | **10.5M** | **0.00228** | **25% better** |

Another important finding: the network matches Stockfish's evaluations
**16% more accurately** than the handcrafted evaluator (measured by
correlation on held-out positions) yet **doesn't play any better** in a
same-node match — evidence that raw accuracy doesn't automatically become
playing strength, since alpha-beta only needs the correct *relative
ordering* between moves.

## 8. The web analysis board

`server.py` + `web/index.html`: an interactive board, arrows pointing to the
best move, a 0–1000 eval bar, results **streamed per depth iteration** (the
first arrow appears after ~40 milliseconds instead of waiting several
seconds). `serve.sh` auto-reads `KING_BUCKETS`/`HIDDEN` from the `.npz`
file, so you never need to remember the environment variables.

## 9. Testing

```bash
python3 test_engine.py     # 86 tests: perft, FEN, Zobrist, draw rules, scoring
python3 check_purity.py    # external-library boundary
```

Covers 14 groups: perft on 6 standard positions, make/unmake state
restoration, incremental Zobrist matching a from-scratch recompute,
mate-in-one and forced mate, scoring scale, color symmetry, castling, en
passant, the 50-move rule, threefold repetition, stalemate/insufficient
material, promotion, and **a search aborted mid-way must still return a
legal move** (a real bug that once silently corrupted match results, now
guarded by a safety net plus a regression test).

## 10. Limitations and what's next

- **Speed.** ~16–80 thousand nodes/second in pure Python, about 600× slower
  than Stockfish. Planned: an **incremental** NNUE accumulator (the "U" in
  NNUE stands for *Updatable* — currently everything is recomputed from
  scratch on every call).
- **No SEE yet** (Static Exchange Evaluation). Will be used for move
  ordering everywhere, but only allowed to fully prune a move inside
  quiescence search, and only **when not in check** — so a sacrifice
  followed by a forcing check sequence never gets wrongly discarded.
- **The KB=64 network is overfitting** (train/val gap +235%) — needs more
  data (from 7.9M to roughly 15–20M positions) to make full use of its
  10.5M parameters.
- **The loss function learns centipawns, not move ranking.** This may
  explain the paradox of "matches Stockfish better but doesn't play
  better."
- **No self-play loop yet.** `datagen.py --nnue` is already wired up to
  generate data using the trained network itself, instead of forever
  learning from Stockfish.

## 11. File map

| File | Role |
|---|---|
| `chess_core.py` | Bitboards, move generation, make/unmake, FEN, Zobrist, `Game` |
| `search.py` | Negamax + alpha-beta, TT, quiescence, null-move, LMR |
| `evaluate.py` | Handcrafted PeSTO evaluation |
| `scoring.py` | cp → 0..1000 mapping, 505-anchor calibration |
| `nnue.py` | Feature extraction, numpy inference, PyTorch model |
| `main.py` | CLI scorer |
| `server.py` + `web/` | Live analysis board, best-move arrows |
| `serve.sh` | Launches the analysis board |
| `test_engine.py` | 86 regression tests |
| `check_purity.py` | Enforces the external-library boundary |
| `datagen_sf.py` | Data generation, Stockfish labeling |
| `make_book.py` | Opening book via MultiPV |
| `train.py` | NNUE training |
| `match.py` | Head-to-head between two evaluators |
| `play_stockfish.py` | Real games against Stockfish, PGN export, Elo ladder |

---

*License: MIT (see `LICENSE`).*

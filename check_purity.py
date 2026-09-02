"""Kiểm tra ranh giới: engine cuối cùng phải SẠCH thư viện cờ ngoài.

Stockfish và python-chess chỉ được phép xuất hiện trong pipeline huấn luyện.
Chạy: python check_purity.py
"""

import ast
import os
import sys

# Các file tạo nên engine chạy thật - phải sạch tuyệt đối
RUNTIME = ["chess_core.py", "search.py", "evaluate.py", "scoring.py",
           "nnue.py", "main.py", "test_engine.py", "server.py"]
# Các file chỉ dùng khi huấn luyện - được phép dùng công cụ ngoài
TRAINING = ["datagen_sf.py", "train.py", "datagen.py", "make_book.py",
            "match.py", "compare_evals.py", "play_stockfish.py"]

BANNED = {"chess", "stockfish", "chess.engine", "chess.pgn", "chess.polyglot"}
# Mẫu chuỗi phải đủ hẹp: "uci(" sẽ bắt nhầm parse_uci/push_uci của chính ta
BANNED_STR = ["stockfish", "popen_uci", "simpleengine", "chess.engine",
              "chess.pgn", "import chess"]

fails = []


def code_strings(path):
    """Gộp mọi chuỗi ký tự THỰC SỰ CHẠY trong file, viết thường.

    Bỏ qua docstring và chú thích: một file nói "không dùng Stockfish" trong
    tài liệu của nó không phải là vi phạm. Thứ cần bắt là chuỗi dùng trong code
    như subprocess.Popen("stockfish") hay import động - những chuỗi đó là
    ast.Constant nằm ngoài vị trí docstring.
    """
    tree = ast.parse(open(path).read(), path)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    parts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            parts.append(node.value)
        elif isinstance(node, ast.Attribute):
            parts.append(node.attr)
        elif isinstance(node, ast.Name):
            parts.append(node.id)
    return " ".join(parts).lower()


def imports_of(path):
    tree = ast.parse(open(path).read(), path)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


print("== Engine chạy thật: phải KHÔNG có thư viện cờ ngoài ==")
for path in RUNTIME:
    if not os.path.exists(path):
        continue
    bad = imports_of(path) & BANNED
    hits = [s for s in BANNED_STR if s.lower() in code_strings(path)]
    if bad or hits:
        fails.append(path)
        print("  SAI  {}  import={} chuỗi={}".format(path, sorted(bad), hits))
    else:
        print("  OK   {}".format(path))

print("\n== Pipeline huấn luyện: ĐƯỢC PHÉP dùng ==")
for path in TRAINING:
    if os.path.exists(path):
        used = sorted(imports_of(path) & BANNED) or ["(không dùng)"]
        print("  --   {}  {}".format(path, used))

print("\n== Nạp engine trong môi trường KHÔNG có python-chess ==")
code = (
    "import sys\n"
    "class Block:\n"
    "    def find_module(self, name, path=None):\n"
    "        return self if name.split('.')[0] == 'chess' else None\n"
    "    def load_module(self, name):\n"
    "        raise ImportError('chess bị chặn có chủ ý')\n"
    "sys.meta_path.insert(0, Block())\n"
    "import chess_core, search, evaluate, scoring, main\n"
    "from chess_core import Position, START_FEN\n"
    "from search import Searcher\n"
    "r = Searcher().search(Position(START_FEN), depth=6)\n"
    "print('   engine chạy bình thường, cp =', r['score'])\n"
)
import subprocess
p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
if p.returncode == 0:
    print(p.stdout.rstrip())
    print("  OK   engine không cần python-chess")
else:
    fails.append("import test")
    print("  SAI\n" + p.stderr[-800:])

print()
if fails:
    print("RANH GIỚI BỊ PHÁ: {}".format(", ".join(fails)))
    sys.exit(1)
print("RANH GIỚI SẠCH: engine không phụ thuộc Stockfish/python-chess")

# server_tools.py
import json
from typing import Dict, Any

from pathlib import Path
import ast
import operator as op
from fastmcp import FastMCP

# Создаём MCP-сервер (в 2.11.3 без description)
# Можно именованно:
server = FastMCP(name="local-tools", version="0.1.0")
# ...или позиционно (эквивалентно):
# server = FastMCP("local-tools", "0.1.0")

# ---- Безопасная математика (только арифметика) ----
_ALLOWED = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

def _eval(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("Only arithmetic is allowed")

@server.tool()
def math_eval(expr: str) -> float:
    """
    Evaluate a simple arithmetic expression offline, e.g. "2 + 2 * 3".
    Allowed ops: + - * / ** and unary -.
    """
    node = ast.parse(expr, mode="eval").body
    return _eval(node)

# ---- Чтение файла в песочнице ----
DATA_DIR = Path(__file__).parent / "data_dir"
DATA_DIR.mkdir(exist_ok=True)

@server.tool()
def read_text(relative_path: str) -> str:
    """
    Read a UTF-8 text file from ./data_dir (sandbox).
    """
    p = (DATA_DIR / relative_path).resolve()
    if DATA_DIR not in p.parents and p != DATA_DIR:
        raise PermissionError("Access outside data_dir is blocked")
    return p.read_text(encoding="utf-8")

@server.tool()
def update_json_simple(
    updates: Dict[str, Any],
    input_filename: str = "input.txt",
    output_filename: str = "Output.txt",
    create_missing: bool = True,
) -> dict:
    """
    Обновляет поля ВЕРХНЕГО уровня в JSON из data_dir/input_filename и
    сохраняет результат в data_dir/output_filename.

    updates: {"name":"Иван", "age": 30}
    create_missing: если ключа нет — создать (True) или игнорировать (False)
    """
    in_path = (DATA_DIR / input_filename).resolve()
    out_path = (DATA_DIR / output_filename).resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {in_path.name}")

    try:
        obj = json.loads(in_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Некорректный JSON во входном файле: {e}")

    if not isinstance(obj, dict):
        raise ValueError("Ожидался JSON-объект верхнего уровня ({}).")

    if not isinstance(updates, dict):
        raise ValueError("updates должен быть объектом {key: value}.")

    changes = []
    for k, new_val in updates.items():
        if k in obj:
            old_val = obj[k]
            obj[k] = new_val
            changes.append({"key": k, "old": old_val, "new": new_val, "created": False})
        else:
            if create_missing:
                obj[k] = new_val
                changes.append({"key": k, "old": None, "new": new_val, "created": True})

    # Атомарная запись: сначала во временный файл, затем replace
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)

    return {
        "written": out_path.name,
        "changed": changes,
        "total_changed": len(changes),
    }


if __name__ == "__main__":
    server.run()   # для 2.11.3 это правильный запуск


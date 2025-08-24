import asyncio
import sys
import json
import argparse
from pathlib import Path
import requests
import re

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# -----------------------------
# Константы/пути
# -----------------------------
SCRIPT = Path(__file__).with_name("server_tools.py")
DATA_DIR = SCRIPT.parent / "data_dir"

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma3:4b"

INPUT_FILENAME = "input.txt"
MAX_INPUT_PREVIEW = 200_000
MAX_TOOLS_IN_PROMPT = 20

# -----------------------------
# Базовые утилиты
# -----------------------------

def log(msg: str) -> None:
    print(f"[agent] {msg}")

def load_input_json_text() -> str:
    path = DATA_DIR / INPUT_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"[Error loading {INPUT_FILENAME}: {e}]"
    if len(text) > MAX_INPUT_PREVIEW:
        return text[:MAX_INPUT_PREVIEW] + "... [truncated]"
    return text

def sanitize_relative_path(p: str) -> str:
    p = (p or "").strip()
    p = p.replace("/", "").replace("\\", "")
    if p.startswith("."):
        p = p.lstrip(".")
    return p or "Output.txt"

def json_safe(obj, _depth: int = 0):
    if _depth > 6:
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v, _depth + 1) for v in obj]
    for attr in ("model_dump", "dict", "to_dict"):
        try:
            if hasattr(obj, attr):
                dumped = getattr(obj, attr)()
                return json_safe(dumped, _depth + 1)
        except Exception:
            pass
    for attr in ("model_dump_json",):
        try:
            if hasattr(obj, attr):
                dumped_json = getattr(obj, attr)()
                return json.loads(dumped_json)
        except Exception:
            pass
    try:
        return json_safe(obj.__dict__, _depth + 1)
    except Exception:
        pass
    return str(obj)

# -----------------------------
# Приведение булевых значений
# -----------------------------

def _coerce_bool_scalar(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "true":  return True
        if s == "false": return False
    return v

def _coerce_booleans_in_dict(d: dict) -> dict:
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _coerce_booleans_in_dict(v)
        elif isinstance(v, list):
            out[k] = [ _coerce_bool_scalar(x) for x in v ]
        else:
            out[k] = _coerce_bool_scalar(v)
    return out

# -----------------------------
# Ollama
# -----------------------------

def ask_ollama(system: str, user: str, model: str = MODEL, temperature: float = 0.2) -> str:
    prompt = f"[SYSTEM]\n{system.strip()}\n\n[USER]\n{user.strip()}"
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", "")).strip()
    except Exception as e:
        return f"[LLM error: {e}]"

# -----------------------------
# Инструменты из MCP (динамика)
# -----------------------------

def _tool_to_record(t) -> dict:
    if isinstance(t, dict):
        name = t.get("name", "")
        desc = t.get("description", "") or ""
        schema = t.get("input_schema", {}) or {}
    else:
        name = getattr(t, "name", "") or ""
        desc = getattr(t, "description", "") or ""
        schema = getattr(t, "input_schema", {}) or {}
    return {"name": str(name), "description": str(desc), "schema": dict(schema)}

def _summarize_schema(schema: dict) -> str:
    props = (schema or {}).get("properties", {}) or {}
    required = set((schema or {}).get("required", []) or [])
    parts = []
    for k, v in props.items():
        typ = v.get("type", "any")
        opt = "" if k in required else "?"
        parts.append(f"{k}{opt}:{typ}")
    return ", ".join(parts) if parts else "no-args"

def _format_tool_line(t: dict) -> str:
    name = t["name"]
    desc = (t.get("description") or "").strip().splitlines()[0]
    args = _summarize_schema(t.get("schema") or {})
    return f"- {name} — {desc}\n  args: {args}"

def _select_relevant_tools(user_query: str, tools: list[dict], max_tools: int = MAX_TOOLS_IN_PROMPT) -> list[dict]:
    q = (user_query or "").lower()
    tokens = [tok for tok in re.split(r"[^\w]+", q) if tok]
    scored = []
    for t in tools:
        text = (t["name"] + " " + (t.get("description") or "")).lower()
        score = sum(1 for tok in tokens if tok in text)
        scored.append((score, t))
    scored.sort(key=lambda x: (-x[0], x[1]["name"]))
    selected = [t for s, t in scored[:max_tools] if s > 0]
    if not selected:
        selected = [t for _, t in scored[:max_tools]]
    return selected

# -----------------------------
# Промпт для ПЛАНА (двухшаговый сценарий)
# -----------------------------

DECISION_RULES = """
Отвечай СТРОГО в JSON. Разрешены формы:

1) ДВУХШАГОВЫЙ ПЛАН изменения JSON с последующей проверкой:
{
  "action": "plan",
  "steps": [
    {
      "name": "update_json_simple",
      "arguments": {
        "updates": { "<ТОПОВОЙ_КЛЮЧ>": <НОВОЕ_ЗНАЧЕНИЕ_БЕЗ_КАВЫЧЕК_ДЛЯ_TRUE_FALSE> },
        "input_filename": "input.txt",
        "output_filename": "Output.txt",
        "create_missing": false
      }
    },
    {
      "name": "verify_json_fields",
      "arguments": {
        "filename": "Output.txt",
        "expectations": { "<ТОПОВОЙ_КЛЮЧ>": <ТО_ЖЕ_ЗНАЧЕНИЕ_КАК_ВЫШЕ> }
      }
    }
  ],
  "final_say": "Короткое резюме для пользователя."
}

2) Финальный ответ без инструментов (для вопросов, где инструменты не нужны):
{ "action": "final", "final_say": "..." }

ВАЖНО:
- Используй ИМЕНА инструментов ТОЧНО как в списке ниже.
- Для изменения поля в JSON ВСЕГДА возвращай вариант (1) — план из двух шагов (обновить → проверить).
- Ключи только верхнего уровня (без точек/скобок). Не выдумывай новых ключей.
- Булевы значения должны быть без кавычек: true/false (а не "true"/"false").
- Если целевой ключ отсутствует, update_json_simple с create_missing=false должен вернуть ошибку — это корректный исход.
"""

def build_decide_prompt(user_question: str, input_json_text: str, tools_for_prompt: list[dict]) -> tuple[str, str]:
    system = (
        "Ты — диспетчер. Твоя задача — построить план из инструментов. "
        "Мы демонстрируем путь: изменить JSON, затем проверить изменение. "
        "Верни СТРОГИЙ JSON в указанных форматах."
    )
    tools_block = "\n".join(_format_tool_line(t) for t in tools_for_prompt)
    user = (
        f"{DECISION_RULES}\n\n"
        f"Доступные инструменты (фрагмент каталога из MCP):\n{tools_block}\n\n"
        f"Вопрос пользователя:\n{user_question}\n\n"
        f"Содержимое {INPUT_FILENAME} (обрезано при необходимости):\n{input_json_text}\n"
    )
    return system, user

def parse_llm_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start:end+1]
    try:
        data = json.loads(snippet)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None

def decide_action(user_question: str, input_json_text: str, tools_for_prompt: list[dict], model: str = MODEL) -> dict | str:
    system, user = build_decide_prompt(user_question, input_json_text, tools_for_prompt)
    raw = ask_ollama(system, user, model=model)
    log(f"LLM raw decision: {raw[:400]}{'...' if len(raw) > 400 else ''}")
    data = parse_llm_json(raw)
    if not data:
        return raw
    action = data.get("action")
    if action not in {"plan", "final"}:
        return raw
    return data

# -----------------------------
# НОРМАЛИЗАЦИЯ ПЛАНА ДО ВЫПОЛНЕНИЯ
# -----------------------------

def _normalize_plan(decision: dict) -> dict:
    """Принудительно правим шаги: create_missing=false и булевы как типы."""
    if not isinstance(decision, dict):
        return decision
    steps = decision.get("steps")
    if not isinstance(steps, list):
        return decision

    norm_steps = []
    for i, st in enumerate(steps, start=1):
        st = dict(st or {})
        name = st.get("name")
        args = dict((st.get("arguments") or {}))
        # санитайзинг путей
        for k in ("relative_path", "filename", "input_filename", "output_filename"):
            if k in args and isinstance(args[k], str):
                args[k] = sanitize_relative_path(args[k])

        if name == "update_json_simple":
            args["create_missing"] = False  # <--- ЖЁСТКО
            if isinstance(args.get("updates"), dict):
                args["updates"] = _coerce_booleans_in_dict(args["updates"])
        elif name == "verify_json_fields":
            if isinstance(args.get("expectations"), dict):
                args["expectations"] = _coerce_booleans_in_dict(args["expectations"])

        st["arguments"] = args
        norm_steps.append(st)

    norm = dict(decision)
    norm["steps"] = norm_steps

    # логируем до/после (коротко)
    try:
        before = json.dumps(decision.get("steps"), ensure_ascii=False)[:200]
        after  = json.dumps(norm_steps, ensure_ascii=False)[:200]
        log(f"Plan normalization:\n  before: {before}\n  after:  {after}")
    except Exception:
        pass

    return norm

# -----------------------------
# Исполнение плана
# -----------------------------

async def run_plan(session: ClientSession, tools_by_name: dict, plan_steps: list[dict]) -> list[dict]:
    results = []
    for idx, step in enumerate(plan_steps, start=1):
        name = (step or {}).get("name")
        args = (step or {}).get("arguments", {}) or {}

        # ещё раз гарантируем жёсткие правила ПЕРЕД самым вызовом
        for k in ("relative_path", "filename", "input_filename", "output_filename"):
            if k in args and isinstance(args[k], str):
                args[k] = sanitize_relative_path(args[k])

        if name == "update_json_simple":
            args["create_missing"] = False
            if isinstance(args.get("updates"), dict):
                args["updates"] = _coerce_booleans_in_dict(args["updates"])

        if name == "verify_json_fields":
            if isinstance(args.get("expectations"), dict):
                args["expectations"] = _coerce_booleans_in_dict(args["expectations"])

        if name not in tools_by_name:
            results.append({"step": idx, "tool": name, "error": f"Unknown tool: {name}"})
            break

        log(f"Calling tool[{idx}]: {name} args={args}")
        try:
            res = await session.call_tool(name, args)
            results.append({"step": idx, "tool": name, "args": args, "result": json_safe(res)})
        except Exception as e:
            results.append({"step": idx, "tool": name, "args": args, "error": str(e)})
            break

        last = results[-1]
        if "error" in last:
            break

    return results

# -----------------------------
# Основной обработчик запроса
# -----------------------------

async def handle_question(session: ClientSession, tools_by_name: dict, tools_for_prompt: list[dict], user_question: str, model: str = MODEL) -> None:
    input_json_text = load_input_json_text()
    decision = decide_action(user_question, input_json_text, tools_for_prompt, model=model)

    if isinstance(decision, str):
        print("Final answer:", decision)
        return

    action = decision.get("action")

    if action == "final":
        print("Final answer:", (decision.get("final_say") or "").strip())
        return

    if action == "plan":
        # НОРМАЛИЗУЕМ ПЛАН ПЕРЕД ВЫПОЛНЕНИЕМ
        decision = _normalize_plan(decision)
        steps = decision.get("steps", [])
        plan_results = await run_plan(session, tools_by_name, steps)

        # wrap-up
        system = (
            "Суммируй результат плана кратко. "
            "Если среди шагов есть verify_json_fields и его результат содержит ok=false или mismatches — сообщи об этом явно. "
            "Если update_json_simple вернул ошибку (например, ключ отсутствует), скажи об этом и не утверждай, что изменение применено."
        )
        user = json.dumps({"question": user_question, "plan_results": plan_results}, ensure_ascii=False)
        summary = ask_ollama(system, user, model=model)
        final_text = (summary or "").strip()
        if not final_text:
            final_text = json.dumps(plan_results, ensure_ascii=False)
        print("Final answer:", final_text)
        return

    print("Final answer:", json.dumps(decision, ensure_ascii=False))

# -----------------------------
# Точка входа
# -----------------------------

async def main():
    parser = argparse.ArgumentParser(description="MCP JSON agent — strict plan normalization")
    parser.add_argument("question", nargs="*", help="Вопрос одной строкой (если пусто — интерактивный режим)")
    parser.add_argument("--model", default=MODEL, help="Имя модели Ollama")
    parser.add_argument("--max-tools", type=int, default=MAX_TOOLS_IN_PROMPT, help="Сколько инструментов показывать LLM")
    args = parser.parse_args()

    model = args.model
    user_q = " ".join(args.question).strip()
    max_tools = int(args.max_tools)

    if not SCRIPT.exists():
        print(f"Не найден сервер инструментов: {SCRIPT}")
        sys.exit(1)

    server = StdioServerParameters(command=sys.executable, args=[str(SCRIPT)], env=None)

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            log("MCP session initialized.")

            list_result = await session.list_tools()
            raw_tools = getattr(list_result, "tools", list_result)
            tools = [_tool_to_record(t) for t in raw_tools]
            tools_by_name = {t["name"]: t for t in tools}

            async def run_once(question: str):
                tools_for_prompt = _select_relevant_tools(question, tools, max_tools=max_tools)
                await handle_question(session, tools_by_name, tools_for_prompt, question, model=model)

            if user_q:
                await run_once(user_q)
            else:
                print('Введите вопрос (пустая строка или "exit" — выход):')
                while True:
                    try:
                        line = input("> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if not line or line.lower() == "exit":
                        break
                    await run_once(line)

if __name__ == "__main__":
    asyncio.run(main())

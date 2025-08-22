import asyncio
import sys
from pathlib import Path
import requests
import json
import re
import argparse

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

SCRIPT = Path(__file__).with_name("server_tools.py")
DATA_DIR = SCRIPT.parent / "data_dir"
OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma3:4b"  # можешь сменить на "gpt-oss:20b"

def ask_ollama(prompt: str, temperature: float = 0.0) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": temperature}},
        timeout=180
    )
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data.get("response", "")

def sanitize_expr(expr: str) -> str:
    expr = expr.splitlines()[0]
    expr = expr.replace("^", "**")
    expr = re.sub(r"[^0-9+\-*/().\s]", "", expr)
    return expr.strip() or "0"

def sanitize_relative_path(p: str) -> str:
    p = p.splitlines()[0].strip()
    if p.startswith("\\") or p.startswith("/") or ":" in p or ".." in p:
        return "sample.txt"
    return p or "sample.txt"

def decide_action(user_question: str, input_json_text: str) -> dict:
    system = (
        "Ты офлайн-агент. ВСЕГДА отвечай СТРОГО валидным JSON без какого-либо дополнительного текста.\n"
        "Поддерживаемые форматы ответа:\n"
        '1) {"action":"tool","name":"math_eval","expr":"<арифметическое выражение>"}\n'
        '2) {"action":"tool","name":"read_text","relative_path":"<относительный_путь_в_data_dir>"}\n'
        '3) {"action":"tool","name":"update_json_simple","updates":{"<ключ>":"<значение>", "...":"..."},'
        '"input_filename":"input.txt","output_filename":"Output.txt"}\n'
        '4) {"action":"final","text":"<готовый ответ>"}\n'
        "\n"
        "ОЧЕНЬ ВАЖНО:\n"
        "- Используй РОВНО те имена полей (ключей), которые указал пользователь. Не заменяй на синонимы.\n"
        '- Если пользователь сказал \"title\", значит ключ ДОЛЖЕН быть ровно \"title\" (учитывая регистр).\n'
        "- JSON ниже дан только для анализа структуры. НЕ ПЕРЕПИСЫВАЙ его целиком в ответе, возвращай только JSON-решение (форматы выше).\n"
        "- Если инструмент не нужен — возвращай формат (4).\n"
        "\n"
        "Примеры для (3):\n"
        '  {"action":"tool","name":"update_json_simple","updates":{"title":"Поставщики"},"input_filename":"input.txt","output_filename":"Output.txt"}\n'
        '  {"action":"tool","name":"update_json_simple","updates":{"status":"done","priority":2},"input_filename":"input.txt","output_filename":"Output.txt"}\n'
    )

    raw = ask_ollama(
        system +
        "\nВот ТЕКУЩИЙ JSON из input.txt (для контекста):\n```json\n" +
        input_json_text +
        "\n```\n"
        f"Вопрос пользователя: {user_question}\nОтвети JSON:"
    )
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return {"action": "final", "text": raw.strip()}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "final", "text": raw.strip()}


    raw = ask_ollama(system + f"\nВопрос пользователя: {user_question}\nОтвети JSON:")
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return {"action": "final", "text": raw.strip()}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "final", "text": raw.strip()}


    raw = ask_ollama(system + f"\nВопрос пользователя: {user_question}\nОтвети JSON:")
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return {"action": "final", "text": raw.strip()}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "final", "text": raw.strip()}


def load_input_json_text(filename: str = "input.txt", max_bytes: int = 200_000) -> str:
    """
    Читает текст из data_dir/filename (UTF-8).
    Ограничивает размер (по умолчанию ~200 КБ), чтобы не переполнить контекст.
    """
    p = DATA_DIR / filename
    text = p.read_text(encoding="utf-8")
    if len(text.encode("utf-8")) > max_bytes:
        # аккуратно обрежем, но оставим валидный JSON фрагмент в подсказке
        truncated = text[:max_bytes // 2]
        return truncated + "\n\n/* ...TRUNCATED... */"
    return text

async def handle_question(session: ClientSession, user_question: str):
    input_json_text = load_input_json_text("input.txt")   # читаем текущий JSON
    decision = decide_action(user_question, input_json_text)
    # ↓↓↓ НОВОЕ: печатаем решение LLM как JSON, чтобы видеть, что именно она вернула
    print("LLM decision:", json.dumps(decision, ensure_ascii=False, indent=2))

    if decision.get("action") == "tool":
        name = decision.get("name")
        if name == "math_eval":
            expr = sanitize_expr(decision.get("expr", ""))
            res = await session.call_tool("math_eval", {"expr": expr})
            value = res.content[0].text if res.content else "Ошибка"
            final = ask_ollama(
                f"Результат вычисления {expr} = {value}. Сформулируй краткий ответ пользователю."
            )
            print("Final answer:", final.strip())
            

        elif name == "read_text":
            rel = sanitize_relative_path(decision.get("relative_path", "sample.txt"))
            res = await session.call_tool("read_text", {"relative_path": rel})
            content = res.content[0].text if res.content else ""
            final = ask_ollama(
                "Вот содержимое файла:\n"
                f"{content}\n\n"
                "Сформируй краткий пересказ для пользователя (1-2 предложения)."
            )
            print("Final answer:", final.strip())

        elif name == "update_json_simple":
            updates = decision.get("updates", {})
            input_filename = decision.get("input_filename", "input.txt")
            output_filename = decision.get("output_filename", "Output.txt")

            # простая валидация updates
            if not isinstance(updates, dict) or not updates:
                print("Final answer:", "Нужно передать updates как объект {key:value}.")
                return

            res = await session.call_tool("update_json_simple", {
                "updates": updates,
                "input_filename": input_filename,
                "output_filename": output_filename
            })
            result_text = res.content[0].text if res.content else ""

            # Попросим LLM оформить короткий итог (можно и просто вывести result_text)
            final = ask_ollama(
                "Инструмент применил обновления к JSON и сохранил файл. "
                f"Отчет инструмента:\n{result_text}\n"
                "Сформулируй краткий ответ пользователю одной строкой."
            )
            print("Final answer:", final.strip())


        else:
            print("Final answer:", json.dumps(decision, ensure_ascii=False))
    else:
        print("Final answer:", decision.get("text", "").strip())

async def main():
    parser = argparse.ArgumentParser(description="MCP offline agent with Ollama")
    parser.add_argument("question", nargs="*", help="Вопрос пользователя (если не передан — будет интерактивный режим)")
    args = parser.parse_args()

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SCRIPT)],
        cwd=str(SCRIPT.parent),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            if args.question:
                user_q = " ".join(args.question)
                await handle_question(session, user_q)
            else:
                # интерактивный режим
                print('Введите вопрос (пустая строка или "exit" — выход):')
                while True:
                    try:
                        user_q = input("> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if not user_q or user_q.lower() == "exit":
                        break
                    await handle_question(session, user_q)

if __name__ == "__main__":
    asyncio.run(main())
"""A customer service agent for Harbor Bean Café (FastAPI backend + HTML/JavaScript frontend).

Every question goes to the agent first. The agent decides which tool to use:
café information and the menu come from a SQLite database, the code works out the bill,
table bookings are written only after the customer confirms, and anything the tools
cannot answer is handed to a human.

The API key, the agent loop, the tools and the database all live here, on the server.
The page in static/ only sends the conversation to /api/chat and shows the answer and the tool trace.

Run it with:  python app.py      then open http://127.0.0.1:8000
"""

import csv
import json
import os
import re
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. API setup
# ---------------------------------------------------------------------------
# Settings come from the .env file in this folder (copy .env.example to .env first).
load_dotenv(HERE / ".env")
API_KEY = os.getenv("API_KEY")
API_ENDPOINT = os.getenv("LLM_API_URL")
CHAT_MODEL = os.getenv("MODEL_NAME")

client = OpenAI(api_key=API_KEY or "missing", base_url=API_ENDPOINT, timeout=60.0)


# ---------------------------------------------------------------------------
# 2. The café database: cafe_info, menu and reservations (SQLite)
# ---------------------------------------------------------------------------
DB_FILE = HERE / "cafe_app.db"

# FastAPI answers requests on worker threads, so all database work goes through one lock.
connection = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=10)
connection.row_factory = sqlite3.Row
db_lock = threading.Lock()


def query(sql, params=()):
    """Run one SQL statement with fixed text and ? parameters. The model never writes SQL."""
    with db_lock:
        rows = connection.execute(sql, params).fetchall()
        connection.commit()
        return rows


def build_database():
    """Rebuild the tables from the CSV files every time the program starts."""
    with db_lock:
        cursor = connection.cursor()
        cursor.execute("DROP TABLE IF EXISTS cafe_info")
        cursor.execute("""
            CREATE TABLE cafe_info (
                topic   TEXT PRIMARY KEY,
                content TEXT NOT NULL
            )
        """)
        cursor.execute("DROP TABLE IF EXISTS menu")
        cursor.execute("""
            CREATE TABLE menu (
                item_id     INTEGER PRIMARY KEY,
                name        TEXT NOT NULL,
                name_zh     TEXT NOT NULL,     -- Chinese name, for questions in Chinese
                category    TEXT NOT NULL,     -- coffee, non-coffee, dessert, meal, beans
                price       INTEGER NOT NULL,  -- New Taiwan dollars, Regular size
                large_extra INTEGER NOT NULL,  -- extra for Large; 0 = one size only
                milk_based  INTEGER NOT NULL,  -- 1 = can be made with oat milk
                available   INTEGER NOT NULL
            )
        """)
        # Bookings are not in a CSV file: the reserve_table tool adds them while the app runs.
        cursor.execute("DROP TABLE IF EXISTS reservations")
        cursor.execute("""
            CREATE TABLE reservations (
                reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                phone      TEXT NOT NULL,
                date       TEXT NOT NULL,
                time       TEXT NOT NULL,
                people     INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        with open(HERE / "cafe_info.csv", newline="", encoding="utf-8") as f:
            cursor.executemany(
                "INSERT INTO cafe_info VALUES (?, ?)",
                [(r["topic"], r["content"]) for r in csv.DictReader(f)],
            )
        with open(HERE / "menu.csv", newline="", encoding="utf-8") as f:
            cursor.executemany(
                "INSERT INTO menu VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(int(r["item_id"]), r["name"], r["name_zh"], r["category"], int(r["price"]),
                  int(r["large_extra"]), int(r["milk_based"]), int(r["available"]))
                 for r in csv.DictReader(f)],
            )
        connection.commit()


build_database()
CAFE_TOPICS = [r["topic"] for r in query("SELECT topic FROM cafe_info ORDER BY rowid")]
CATEGORIES = [r["category"] for r in query("SELECT DISTINCT category FROM menu ORDER BY item_id")]
MENU_SIZE = query("SELECT COUNT(*) AS n FROM menu")[0]["n"]
print(f"Ready: {len(CAFE_TOPICS)} cafe topics, {MENU_SIZE} menu items.")


# ---------------------------------------------------------------------------
# 3. The tools, as plain functions
# ---------------------------------------------------------------------------
def get_cafe_info(topic):
    """Return one piece of café information, such as the opening hours or the Wi-Fi."""
    rows = query("SELECT topic, content FROM cafe_info WHERE topic = ? COLLATE NOCASE", (topic.strip(),))
    if not rows:
        return {"found": False, "note": f"There is no café information about '{topic}'.",
                "topics_available": CAFE_TOPICS}
    return {"found": True, "topic": rows[0]["topic"], "content": rows[0]["content"]}


# What customers say -> words in the menu names or categories.
ALIASES = {"cake": "cheesecake", "croissant": "croissant", "coffee": "coffee",
           "tea": "tea", "chocolate": "chocolate", "cocoa": "chocolate", "pour": "hand-drip",
           "drip": "hand-drip", "bean": "beans", "pasta": "pasta", "sandwich": "panini"}
CATEGORY_ZH = {"咖啡": "coffee", "非咖啡": "non-coffee", "茶": "non-coffee", "甜點": "dessert",
               "蛋糕": "dessert", "餐點": "meal", "輕食": "meal", "正餐": "meal", "咖啡豆": "beans",
               "豆子": "beans", "coffee beans": "beans", "飲料": "drink", "drinks": "drink", "food": "food", "吃的": "food"}
GROUPS = {"drink": ("coffee", "non-coffee"), "food": ("dessert", "meal")}
# Size, temperature and filler words say how to make the item, not which item it is.
SKIP_EN = {"a", "an", "and", "the", "of", "hand", "cup", "glass", "slice", "piece", "iced", "ice", "hot",
           "large", "regular", "small", "medium", "with", "oat", "milk"}
SKIP_ZH = set("杯份個片塊的客冰熱溫大中小燕麥奶一二兩三四五六七八九十")


def words(text):
    """'2 Iced Lattes!' -> {'2', 'latte'};  '大杯冰拿鐵' -> {'拿', '鐵'}.

    English: lower case, no punctuation, no plural s, no size words.  Chinese: each character is one word.
    """
    result = set()
    for word in re.sub(r"[^a-z0-9\-]+", " ", text.lower()).split():
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]                              # lattes -> latte
        if word in SKIP_EN or word.isdigit():
            continue
        result.update(ALIASES.get(word, word).split())
    result.update(ch for ch in text if "一" <= ch <= "鿿" and ch not in SKIP_ZH)
    return result


def find_items(text):
    """Menu rows that match the text: the English name, the Chinese name, or the category.

    Customers type "iced latte" or "大杯拿鐵", not "Latte", so we compare words.
    """
    key = text.strip().lower()
    key = CATEGORY_ZH.get(key, key)
    rows = query("SELECT * FROM menu ORDER BY item_id")
    if key in GROUPS:                                     # "drinks" = coffee + non-coffee
        return [r for r in rows if r["category"] in GROUPS[key]]
    if key in CATEGORIES:
        return [r for r in rows if r["category"] == key]
    exact = [r for r in rows if key in (r["name"].lower(), r["name_zh"])]
    if exact:                                             # "拿鐵" is the Latte, not the Matcha Latte
        return exact
    wanted = words(text)
    if not wanted:
        return []
    matches = [r for r in rows if wanted <= words(r["name"]) | words(r["name_zh"])]
    plain = [r for r in matches if wanted in (words(r["name"]), words(r["name_zh"]))]
    return plain if len(plain) == 1 else matches          # "iced latte" is the Latte, not the Matcha Latte


def item_to_dict(r):
    return {"name": r["name"], "name_zh": r["name_zh"], "category": r["category"], "price": r["price"],
            "large_available": r["large_extra"] > 0, "oat_milk_possible": bool(r["milk_based"]),
            "available": bool(r["available"])}


def search_menu(keyword):
    """Find menu items by part of a name ('latte', '拿鐵') or by category ('dessert', '甜點')."""
    rows = find_items(keyword)
    if not rows:
        return {"found": 0, "items": [], "note": f"No menu item matches '{keyword}'.",
                "categories": CATEGORIES}
    return {"found": len(rows), "items": [item_to_dict(r) for r in rows]}


def get_item(item_name):
    """Return the price, sizes and availability of ONE item. The name does not have to be exact."""
    rows = find_items(item_name)
    if len(rows) == 1:
        r = rows[0]
        result = {"found": True, **item_to_dict(r)}
        if r["large_extra"]:
            result["large_price"] = r["price"] + r["large_extra"]
        return result
    if len(rows) > 1:
        return {"found": False, "note": f"'{item_name}' matches several items. Ask the customer which one.",
                "candidates": [f"{r['name']} ({r['name_zh']})" for r in rows]}
    return {"found": False, "note": f"'{item_name}' is not on the menu."}


OAT_MILK_EXTRA = 15     # the same numbers as size_and_milk, own_cup and membership in cafe_info.csv
OWN_CUP_DISCOUNT = 5
MEMBER_RATE = 0.10


def calculate_order(items, own_cup=False, member=False):
    """Work out a café bill: size, oat milk, own-cup discount and member discount, all in code.

    No total unless every item is clear and can be sold: a total of only some items is a wrong bill.
    """
    lines, problems, subtotal, drinks = [], [], 0, 0
    for entry in items:
        name = str(entry.get("name", ""))
        quantity = int(entry.get("quantity", 1))
        size = str(entry.get("size", "regular")).lower()
        oat_milk = bool(entry.get("oat_milk", False))
        rows = find_items(name)
        if len(rows) != 1:
            problems.append({"item": name, **get_item(name)})
            continue
        r = rows[0]
        if not r["available"]:
            problems.append({"item": name, "note": f"{r['name']} is sold out today."})
            continue
        if quantity < 1:
            problems.append({"item": name, "note": "The quantity must be at least 1."})
            continue
        unit = r["price"]
        if size == "large":
            if not r["large_extra"]:
                problems.append({"item": name, "note": f"{r['name']} comes in one size only."})
                continue
            unit += r["large_extra"]
        if oat_milk:
            if not r["milk_based"]:
                problems.append({"item": name, "note": f"{r['name']} has no milk, so oat milk does not apply."})
                continue
            unit += OAT_MILK_EXTRA
        line_total = unit * quantity
        lines.append({"name": r["name"], "name_zh": r["name_zh"], "size": size if r["large_extra"] else "one size",
                      "oat_milk": oat_milk, "unit_price": unit, "quantity": quantity, "subtotal": line_total})
        subtotal += line_total
        if r["category"] in GROUPS["drink"]:
            drinks += quantity

    if problems:
        return {"total": None, "problems": problems,
                "note": "No total yet. Ask the customer about these items, then call calculate_order again."}
    if not lines:
        return {"total": None, "note": "The order is empty. Ask the customer what they would like."}

    cup_discount = OWN_CUP_DISCOUNT * drinks if own_cup else 0
    after_cup = subtotal - cup_discount
    member_discount = round(after_cup * MEMBER_RATE) if member else 0
    return {"lines": lines, "subtotal": subtotal,
            "own_cup_discount": cup_discount, "drinks_counted_for_own_cup": drinks if own_cup else 0,
            "member_discount": member_discount, "total": after_cup - member_discount}


OPEN_TIME, LAST_BOOKING_TIME = "08:00", "20:00"
MAX_PEOPLE = 8
pending_bookings = set()   # bookings summarised to a customer but not yet confirmed


def reserve_table(name, phone, date_text, time_text, people, confirmed=False):
    """Book a table. The first call only checks and summarises; the booking is saved only after
    the customer has seen that summary and said yes (confirmed=true with exactly the same details).
    """
    problems = []
    try:
        day = datetime.strptime(date_text.strip(), "%Y-%m-%d").date()
        if day < date.today():
            problems.append("The date is in the past.")
        elif day.weekday() == 0:
            problems.append("We are closed on Mondays.")
    except ValueError:
        problems.append("The date must look like 2026-10-24.")
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", time_text.strip()):
        problems.append("The time must look like 14:30.")
    elif not OPEN_TIME <= time_text.strip() <= LAST_BOOKING_TIME:
        problems.append(f"Bookings are possible from {OPEN_TIME} to {LAST_BOOKING_TIME}.")
    people = int(people)
    if people < 1:
        problems.append("The number of people must be at least 1.")
    elif people > MAX_PEOPLE:
        problems.append(f"Groups of more than {MAX_PEOPLE} people must call the café: use transfer_to_human.")
    if not name.strip():
        problems.append("The customer's name is missing.")
    if not re.fullmatch(r"[0-9\-+ ]{8,15}", phone.strip()):
        problems.append("The phone number is missing or not valid.")
    if problems:
        return {"saved": False, "problems": problems, "note": "Ask the customer to fix these details."}

    booking = (name.strip(), phone.strip(), date_text.strip(), time_text.strip(), people)
    summary = {"name": booking[0], "phone": booking[1], "date": booking[2], "time": booking[3], "people": people}
    if not confirmed or booking not in pending_bookings:
        # Our code, not the model, makes sure the customer saw the summary before anything is written.
        pending_bookings.add(booking)
        return {"saved": False, "summary": summary,
                "note": ("Not saved yet. Show this summary to the customer and ask them to confirm. "
                         "Only if they say yes, call reserve_table again with the same details and confirmed=true.")}

    pending_bookings.discard(booking)
    query("INSERT INTO reservations (name, phone, date, time, people, created_at) VALUES (?, ?, ?, ?, ?, ?)",
          (*booking, datetime.now().isoformat(timespec="seconds")))
    number = query("SELECT MAX(reservation_id) AS n FROM reservations")[0]["n"]
    return {"saved": True, "reservation_number": number, **summary,
            "note": "We keep a booked table for 15 minutes."}


def transfer_to_human(reason=""):
    """Hand the conversation to a person."""
    return {"message": "Please call our staff at 07-765-4321, Tuesday to Sunday, 09:00 to 20:00.",
            "reason_logged": reason,
            "note": ("The hand-over is done. First tell the customer in one short sentence why: "
                     "you do not have that information, or the request needs a person. "
                     "Then give this message in the customer's language. Do not say that you will connect them "
                     "or pass them on: you cannot. Do not call this tool again.")}


TOOL_FUNCTIONS = {
    "get_cafe_info": get_cafe_info,
    "search_menu": search_menu,
    "get_item": get_item,
    "calculate_order": calculate_order,
    "reserve_table": reserve_table,
    "transfer_to_human": transfer_to_human,
}

# What the model sees: names, purpose, arguments. Most "it did not use my tool"
# problems are really "my description did not say when to use it".
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_cafe_info",
        "description": ("Get one piece of café information: opening hours (營業時間), address (地址), phone (電話), "
                        "payment (付款方式), Wi-Fi and power outlets (網路、插座), time limit (限時), "
                        "minimum order (低消), how to book (訂位規定), own cup discount (自備杯), "
                        "membership (會員), sizes and oat milk prices (大杯、燕麥奶), pets (寵物), "
                        "parking (停車) or delivery (外送)."),
        "parameters": {"type": "object",
                       "properties": {"topic": {"type": "string", "enum": CAFE_TOPICS}},
                       "required": ["topic"]}}},
    {"type": "function", "function": {
        "name": "search_menu",
        "description": ("Search the menu by keyword or category (coffee, non-coffee, drink, dessert, meal, food, "
                        "beans). Use it when the customer asks what is available, "
                        "or names an item only partly, such as 'cake' or 'tea'."),
        "parameters": {"type": "object",
                       "properties": {"keyword": {"type": "string"}},
                       "required": ["keyword"]}}},
    {"type": "function", "function": {
        "name": "get_item",
        "description": ("Get the price of ONE item, whether it comes in Large or with oat milk, and whether it is "
                        "available today. Use the customer's own words and do not translate them, "
                        "for example 'iced latte' or '冰拿鐵'."),
        "parameters": {"type": "object",
                       "properties": {"item_name": {"type": "string"}},
                       "required": ["item_name"]}}},
    {"type": "function", "function": {
        "name": "calculate_order",
        "description": ("Work out the bill for one or more items, including Large size, oat milk, "
                        "the own-cup discount and the member discount. Always use this for any total or "
                        "any price with a size, oat milk or a discount. Never add or discount prices yourself. "
                        "Call it directly with the item names: it looks up the prices itself, "
                        "so do not call get_item first."),
        "parameters": {"type": "object",
                       "properties": {
                           "items": {"type": "array", "items": {
                               "type": "object",
                               "properties": {
                                   "name": {"type": "string", "description": "The item in the customer's own words. Do not translate it."},
                                   "quantity": {"type": "integer"},
                                   "size": {"type": "string", "enum": ["regular", "large"],
                                            "description": "Use large only if the customer asks for Large (大杯)."},
                                   "oat_milk": {"type": "boolean", "description": "True if the customer wants oat milk (燕麥奶)."}},
                               "required": ["name", "quantity"]}},
                           "own_cup": {"type": "boolean", "description": "True if the customer brings their own cup (自備杯)."},
                           "member": {"type": "boolean", "description": "True if the customer says they are a member (會員)."}},
                       "required": ["items"]}}},
    {"type": "function", "function": {
        "name": "reserve_table",
        "description": ("Book a table for 1 to 8 people. Ask for the name, phone, date, time and number of people first. "
                        "Call it with confirmed=false to check the details and get a summary for the customer. "
                        "Call it with confirmed=true only after the customer has said yes to that summary."),
        "parameters": {"type": "object",
                       "properties": {
                           "name": {"type": "string"},
                           "phone": {"type": "string"},
                           "date_text": {"type": "string", "description": "YYYY-MM-DD"},
                           "time_text": {"type": "string", "description": "HH:MM, 24-hour clock"},
                           "people": {"type": "integer"},
                           "confirmed": {"type": "boolean"}},
                       "required": ["name", "phone", "date_text", "time_text", "people", "confirmed"]}}},
    {"type": "function", "function": {
        "name": "transfer_to_human",
        "description": ("Use this when the customer asks for a manager, makes a complaint, wants to book for more "
                        "than 8 people, or asks about the café something no other tool can answer."),
        "parameters": {"type": "object",
                       "properties": {"reason": {"type": "string", "description": "A few words on why."}},
                       "required": []}}},
]

SYSTEM_PROMPT = """
You are "Bean", the online customer service agent of Harbor Bean Café (港豆咖啡), a small café near
NKUST Nanzih campus in Kaohsiung. Your customers are mostly students and people working nearby.
All prices are in New Taiwan dollars. Write them like NT$110.

GROUNDING
1. Answer only from tool results. Never answer from memory, and never invent anything.
2. If a tool reports that nothing was found, say so plainly. Do not guess.
   If the customer asks about the café and no tool has the answer (for example the owner's name or where
   the beans come from), say you do not have that information and call transfer_to_human.

SCOPE
3. Only talk about this café. Politely refuse other topics, such as the weather, homework or other shops,
   and offer to help with the café instead.

ROUTING
4. Café questions (hours, address, Wi-Fi, outlets, time limit, minimum order, payment, discounts, pets,
   parking, delivery) -> get_cafe_info.
5. Menu questions (what is there, one item's price, sizes, sold out or not) -> search_menu or get_item.
6. Any total, or any price with Large, oat milk, own cup or member discount -> calculate_order.
   Never add, multiply or discount prices yourself.
7. Table bookings -> reserve_table. Collect all details first, show the summary, and save only after the
   customer says yes.
8. Complaints, a request for a manager, groups over 8 people, or café questions no tool can answer
   -> transfer_to_human.

LANGUAGE AND FORMAT
9. Answer in the customer's language: English, or Traditional Chinese if the customer writes in Chinese.
   Write item names in the language of the answer: English names in English, Chinese names (name_zh) in Chinese.
10. Be warm and polite, and use three sentences or less, unless you are listing menu items or a bill.
"""

MAX_STEPS = 6          # a confused model must not loop forever


def language_note(text):
    """The program, not the model, decides the reply language: Chinese characters -> Chinese, else English."""
    if re.search(r"[一-鿿]", text):
        return [{"role": "system", "content": "The customer wrote in Chinese. Reply in Traditional Chinese."}]
    return [{"role": "system", "content": "The customer wrote in English. Reply in English."}]


def today_note():
    """The model does not know today's date, but bookings need it."""
    today = date.today()
    return {"role": "system", "content": f"Today is {today.isoformat()} ({today.strftime('%A')})."}


def run_agent(chat_history):
    """Answer the last question in chat_history. Returns (answer, trace)."""
    messages = ([{"role": "system", "content": SYSTEM_PROMPT}, today_note()] + chat_history[:-1]
                + language_note(chat_history[-1]["content"]) + chat_history[-1:])
    trace = []

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=TOOL_SPECS, temperature=0.2,
        )
        message = response.choices[0].message

        if not message.tool_calls:                     # plain text: the turn is finished
            return (message.content or "").strip(), trace

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name,
                                         "arguments": c.function.arguments}}
                           for c in message.tool_calls],
        })

        for call in message.tool_calls:
            name = call.function.name
            function = TOOL_FUNCTIONS.get(name)        # only our own names are allowed
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"raw": call.function.arguments}
                function = None if function is None else "bad-arguments"

            if function is None:
                result = {"error": f"unknown tool: {name}. Use only the tools you were given."}
            elif function == "bad-arguments":
                result = {"error": "The arguments were not valid JSON. Try again."}
            else:
                try:
                    result = function(**arguments)
                except Exception as error:
                    result = {"error": f"{type(error).__name__}: {error}"}

            trace.append({"step": step + 1, "tool": name, "arguments": arguments, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})

    return "Sorry, I could not finish that. Please call our staff at 07-765-4321.", trace


# ---------------------------------------------------------------------------
# 4. The web server: one API route for the chat, and the static page
# ---------------------------------------------------------------------------
GREETING = ("Hi, I'm Bean, the online helper of Harbor Bean Café ☕ "
            "Ask me about our hours, the menu, prices, your bill or booking a table.\n\n"
            "您好！我是港豆咖啡的線上客服 Bean，用中文或英文問我都可以。")

MAX_HISTORY = 20       # only the last turns are sent to the model


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


app = FastAPI(title="Harbor Bean Café Agent")


@app.get("/api/config")
def config():
    """What the page needs to start: the greeting. The sample questions live in the page itself."""
    return {"greeting": GREETING, "model": CHAT_MODEL}


@app.post("/api/chat")
def chat(request: ChatRequest):
    """Run the agent on the conversation the page sends. A plain def: FastAPI runs it on a worker thread."""
    history = [{"role": m.role, "content": m.content[:2000]} for m in request.messages
               if m.role in ("user", "assistant") and m.content.strip()][-MAX_HISTORY:]
    if not history or history[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="The last message must be from the user.")
    if not API_KEY:
        return {"answer": "(system message) API_KEY is empty. Copy .env.example to .env and add your key.",
                "trace": []}
    try:
        answer, trace = run_agent(history)
    except Exception as error:
        return {"answer": f"(system message) API error: {error}", "trace": [], "error": True}
    return {"answer": answer, "trace": trace}


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

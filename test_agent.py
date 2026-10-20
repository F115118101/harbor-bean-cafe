"""Tool selection test: does the agent pick an acceptable tool for each question?

Each test lists the tools we accept as a first choice ("none" = no tool should be needed),
and optionally one tool that must appear somewhere in the trace (for example transfer_to_human
when the data has no answer).
It calls the real model, so it needs a working .env. Run it with:  python test_agent.py
"""

import sys
import time

from app import run_agent

sys.stdout.reconfigure(encoding="utf-8")

TESTS = [
    # (question, acceptable first tools, question type[, tool that must be used])
    ("What time do you open on Sunday?", {"get_cafe_info"}, "in the data"),
    ("有插座可以讀書嗎？", {"get_cafe_info"}, "in the data"),
    ("Is there a time limit on weekends?", {"get_cafe_info"}, "in the data"),
    ("How much is an iced latte?", {"get_item", "search_menu"}, "in the data"),
    ("檸檬塔今天有嗎？", {"get_item", "search_menu"}, "in the data"),
    ("What desserts do you have?", {"search_menu"}, "in the data"),
    ("Two large lattes with oat milk and a tiramisu. I brought my own cups. How much?", {"calculate_order"}, "calculation"),
    ("我是會員，一杯美式咖啡和一個司康，總共多少？", {"calculate_order"}, "calculation"),
    ("我們讀書會 5 個人，每人一杯美式咖啡和一個司康，多少錢？", {"calculate_order"}, "calculation"),
    ("Do you sell bubble tea?", {"search_menu", "get_item"}, "in the data (answer: no)"),
    ("Which farm do your Ethiopia beans come from?", {"transfer_to_human", "get_item", "search_menu"}, "not in the data", "transfer_to_human"),
    ("你們老闆叫什麼名字？", {"transfer_to_human", "get_cafe_info"}, "not in the data", "transfer_to_human"),
    ("My coffee was cold and the staff were rude. I want to talk to the manager.", {"transfer_to_human"}, "complaint"),
    ("Can you help me with my calculus homework?", {"none"}, "not our job"),
    ("明天高雄會下雨嗎？", {"none"}, "not our job"),
]

PAUSE_SECONDS = 5      # free plans allow only a few calls per minute (429 errors)

passed = 0
for number, (question, accepted, kind, *must_use) in enumerate(TESTS, 1):
    try:
        answer, trace = run_agent([{"role": "user", "content": question}])
    except Exception as error:
        answer, trace = f"API error: {error}", []
    tools = [t["tool"] for t in trace]
    first = tools[0] if tools else "none"
    ok = first in accepted and all(tool in tools for tool in must_use)
    passed += ok
    print(f"{number:2}. [{'PASS' if ok else 'FAIL'}] ({kind}) {question}")
    must = f"   must use: {must_use[0]}" if must_use else ""
    print(f"    tools: {' -> '.join(tools) or 'none'}   accepted: {', '.join(sorted(accepted))}{must}")
    print(f"    answer: {answer[:220]}")
    time.sleep(PAUSE_SECONDS)

print(f"\nTool selection accuracy: {passed}/{len(TESTS)} = {passed / len(TESTS):.0%}")

"""Check the house rules search (RAG) without the chat model.

For each question it prints the best chunks and their cosine scores, and whether the right chunk came first.
Use it to choose MIN_SCORE: questions the rules cover should score above it, questions they do not cover below it.
Run it with:  python check_rag.py
"""

import sys

import numpy as np

import app

sys.stdout.reconfigure(encoding="utf-8")

if app.RULE_VECTORS is None:
    sys.exit(f"The rules index could not be built: {app.RAG_ERROR}")

# (question, the chunk title that should come first; None = the rules do not cover it)
QUESTIONS = [
    ("Does the pesto pasta have nuts?", "Allergens and special diets"),
    ("你們有素食餐點嗎？", "Allergens and special diets"),
    ("Can our study group book the back room?", "The study room (back room)"),
    ("包廂一次可以坐幾個人？", "The study room (back room)"),
    ("I will be 20 minutes late for my booking.", "Booking changes, late arrival and no-shows"),
    ("我要取消訂位", "Booking changes, late arrival and no-shows"),
    ("I left my umbrella in the café.", "Lost and found"),
    ("我的學生證掉在你們店裡", "Lost and found"),
    ("Can I bring my own birthday cake?", "Outside food, drinks and birthday cakes"),
    ("Do you have decaf?", "Caffeine, decaf and children"),
    ("小孩可以去嗎？有兒童椅嗎？", "Caffeine, decaf and children"),
    ("My latte was wrong, can I get a refund?", "Wrong drinks, refunds and complaints"),
    ("Can I have a video meeting there?", "Laptops, phone calls and noise"),
    ("我想電話先點外帶", "Takeaway pre-orders by phone"),
    ("Can you grind the beans for my moka pot?", "Coffee beans: grinding, freshness and returns"),
    ("禮券可以找零嗎？", "Gift vouchers"),
    ("颱風天有營業嗎？", "Typhoons and special closing days"),
    # not covered by the rules: these should score below MIN_SCORE
    ("Who is the owner of the café?", None),
    ("Where do your Ethiopia beans come from?", None),
    ("明天高雄會下雨嗎？", None),
    ("Can you help me with my calculus homework?", None),
    ("你們有在徵工讀生嗎？", None),
]

vectors = app.embed([q for q, _ in QUESTIONS])
right, covered_scores, other_scores = 0, [], []
for (question, expected), q in zip(QUESTIONS, vectors):
    scores = app.RULE_VECTORS @ q
    order = np.argsort(scores)[::-1]
    top = app.RULE_CHUNKS[order[0]]["title"]
    best = float(scores[order[0]])
    if expected is None:
        other_scores.append(best)
        ok = best < app.MIN_SCORE
    else:
        covered_scores.append(best)
        ok = top == expected and best >= app.MIN_SCORE
    right += ok
    print(f"[{'OK ' if ok else 'BAD'}] {best:.3f}  {question}")
    print(f"       1st: {top}   2nd: {app.RULE_CHUNKS[order[1]]['title']} ({scores[order[1]]:.3f})"
          + ("" if expected in (None, top) else f"   expected: {expected}"))

print(f"\n{right}/{len(QUESTIONS)} as expected with MIN_SCORE = {app.MIN_SCORE}")
print(f"covered questions:     lowest best score {min(covered_scores):.3f}")
print(f"not-covered questions: highest best score {max(other_scores):.3f}")
print("A good MIN_SCORE lies between these two numbers.")

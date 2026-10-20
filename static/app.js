// The page only shows the chat. It keeps the conversation and sends all of it to /api/chat
// with every question; the API key, the agent and the database stay on the server.

const SAMPLES = {
  en: {
    "Café information (get_cafe_info)": [
      "What time do you open on Sunday?",
      "Do you have Wi-Fi and power outlets?",
      "Is there a time limit on weekends?",
      "Can I bring my dog?",
      "Where can I park my car?",
    ],
    "Menu and items (search_menu, get_item)": [
      "What desserts do you have?",
      "How much is an iced latte?",
      "Is the lemon tart available today?",
      "Do you sell coffee beans?",
    ],
    "The bill (calculate_order)": [
      "Two large lattes with oat milk and a tiramisu. I brought my own cups. How much?",
      "I'm a member. One Americano, one Basque cheesecake and one pesto pasta, please. Total?",
      "Three cappuccinos and some cake, how much?",
    ],
    "Booking (reserve_table)": [
      "I'd like to book a table for 4 this Saturday at 14:00.",
    ],
    "Complaint, not on the menu, not in the data, not our job": [
      "My coffee was cold and the staff were rude. I want to talk to the manager.",
      "Do you sell bubble tea?",
      "Which farm do your Ethiopia beans come from?",
      "Can you help me with my calculus homework?",
    ],
  },
  zh: {
    "店家資訊（get_cafe_info）": [
      "星期一有開嗎？",
      "有插座可以讀書嗎？",
      "有低消嗎？",
      "自備杯有折扣嗎？",
      "你們有外送嗎？",
    ],
    "菜單與品項（search_menu、get_item）": [
      "有什麼甜點？",
      "冰拿鐵多少錢？",
      "檸檬塔今天有嗎？",
      "抹茶拿鐵可以換燕麥奶嗎？",
    ],
    "算錢（calculate_order）": [
      "我是會員，兩杯大杯拿鐵換燕麥奶、一個巴斯克乳酪蛋糕，我有自備杯，總共多少？",
      "我們讀書會 5 個人，每人一杯美式咖啡和一個司康，多少錢？",
      "一杯手沖要大杯，多少錢？",
    ],
    "訂位（reserve_table）": [
      "我想訂這週六下午兩點，3 位。",
    ],
    "客訴、菜單沒有、資料沒有、不是客服的事": [
      "我要客訴，餐點等了 40 分鐘。",
      "有賣珍珠奶茶嗎？",
      "你們老闆叫什麼名字？",
      "明天高雄會下雨嗎？",
    ],
  },
};

const chatBox = document.getElementById("chat");
const traceBox = document.getElementById("trace");
const form = document.getElementById("ask");
const input = document.getElementById("question");
const sendButton = document.getElementById("send");
const groupsBox = document.getElementById("sample-groups");

let greeting = "Hi! 您好！";
let history = [];          // [{role, content}] — the whole conversation, sent with every question
let busy = false;

function addBubble(role, text, extraClass = "") {
  const div = document.createElement("div");
  div.className = `msg ${role} ${extraClass}`.trim();
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function showTrace(trace) {
  traceBox.innerHTML = "";
  if (!trace || trace.length === 0) {
    traceBox.innerHTML = '<p class="empty">(no tool was needed for this answer)</p>';
    return;
  }
  for (const entry of trace) {
    const card = document.createElement("div");
    card.className = "call" + (entry.result && entry.result.error ? " bad" : "");
    card.innerHTML = `
      <div class="call-head"><span class="call-name"></span><span class="call-step"></span></div>
      <div class="label">arguments</div><pre class="args"></pre>
      <div class="label">result</div><pre class="result"></pre>`;
    card.querySelector(".call-name").textContent = entry.tool;
    card.querySelector(".call-step").textContent = `step ${entry.step}`;
    card.querySelector(".args").textContent = pretty(entry.arguments);
    card.querySelector(".result").textContent = pretty(entry.result);
    traceBox.appendChild(card);
  }
}

function startChat() {
  history = [];
  chatBox.innerHTML = "";
  addBubble("assistant", greeting);
  traceBox.innerHTML = '<p class="empty">(nothing yet)</p>';
}

async function ask(question) {
  question = question.trim();
  if (!question || busy) return;
  busy = true;
  sendButton.disabled = true;
  input.value = "";

  history.push({ role: "user", content: question });
  addBubble("user", question);
  const waiting = addBubble("assistant", "Bean is checking… 查詢中…", "thinking");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    waiting.remove();
    addBubble("assistant", data.answer, data.error ? "error" : "");
    if (!data.error) history.push({ role: "assistant", content: data.answer });
    showTrace(data.trace);
  } catch (error) {
    waiting.remove();
    addBubble("assistant", `(system message) Could not reach the server: ${error.message}`, "error");
    history.pop();         // the question was not answered, so it is not part of the conversation
  } finally {
    busy = false;
    sendButton.disabled = false;
    input.focus();
  }
}

function showSamples(lang) {
  groupsBox.innerHTML = "";
  for (const [title, questions] of Object.entries(SAMPLES[lang])) {
    const group = document.createElement("div");
    group.className = "group";
    group.innerHTML = '<div class="group-title"></div><div class="chips"></div>';
    group.querySelector(".group-title").textContent = title;
    const chips = group.querySelector(".chips");
    for (const q of questions) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = q;
      chip.addEventListener("click", () => { input.value = q; input.focus(); });
      chips.appendChild(chip);
    }
    groupsBox.appendChild(group);
  }
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.lang === lang));
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(input.value);
});
document.getElementById("clear").addEventListener("click", startChat);
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showSamples(t.dataset.lang)));

fetch("/api/config")
  .then((r) => r.json())
  .then((data) => { greeting = data.greeting; })
  .catch(() => {})
  .finally(startChat);
showSamples("en");

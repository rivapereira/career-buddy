# 🚀 Career Buddy – Your AI-Powered Career Planner

![image](https://github.com/user-attachments/assets/5692550d-2c0b-478c-a9f5-259fbf7270e2)


> Personalized, motivational, and intelligent — Career Buddy helps you move from **goals to actions** with weekly plans, smart rewards, and AI-generated roadmaps.

[👉 **Launch App on Hugging Face**](https://huggingface.co/spaces/rivapereira123/career-buddy)

---

## 📍 Author

**Riva Pereira**  
AI Residency Cohort 4, University of Wollongong (Dubai)  
Built as a capstone project exploring agentic AI, human motivation, and career tools that go beyond static dashboards — into **personalized, goal-driven coaching**.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🧠 **AI Roadmap Generator** | Personalized 4–6 week plan using GPT-4o, Tavilly Search, and dynamic summarization |
| 🔁 **RAG-Based Memory** | Retrieves your past goals, summaries, and tasks using Pinecone + LlamaIndex |
| 📚 **Course Recommender** | Suggests top courses for career goals (e.g. Coursera, edX, Codecademy) |
| 🎯 **Gamified Task Manager** | Add weekly tasks (Critical 🔴, Important 🟠, Optional 🟢), mark progress |
| 🎁 **Reward System** | Claim motivational rewards when you complete plans |
| 📅 **Google Calendar Sync** | Auto-schedule plans with OAuth-based integration |
| 👥 **LinkedIn & GitHub Analyzer** | AI feedback for your profile and README quality |
| 🧪 **Gradio Interface** | Simple UI for fast testing or deployment |

---

## 🙋 Why Career Buddy?

Most tools stop at inspiration or tracking. Career Buddy helps you **plan, act, and follow through**:

- Converts broad goals into weekly action steps
- Rewards consistency with dopamine-boosting milestones
- Provides actionable content, not just advice

---

## 🧠 Tech Stack

- **Frontend:** Gradio (Python)
- **Backend:** Python, LangChain, LangGraph
- **LLMs:** GPT-4o, HuggingFace FLAN-T5
- **Agents:** TaskPlanner, ResourceFinder, MotivationAgent
- **Memory / RAG:** LlamaIndex + Pinecone
- **Auth:** Firebase Auth + Google OAuth
- **Scheduling:** Google Calendar API
- **Web Search:** Tavilly Career Contextual API

---

## 🧩 Agentic AI Design

Career Buddy uses modular AI agents with distinct roles:

| Agent | Role |
|-------|------|
| 🎓 `TaskPlannerAgent` | Generates weekly roadmap |
| 📚 `ResourceFinderAgent` | Searches learning content via Tavilly |
| 💬 `MotivationAgent` | Gives feedback and unlocks rewards |
| 🧠 `MemoryAgent` | Stores goals, summaries, completions |

Agents talk to each other via `LangGraph`, and you can extend to `CrewAI` or `Swarm`.

---

## 🔗 Integrations

- 🗂 Pinecone VectorDB
- 🔍 Tavilly Real-Time Web Search
- 🎓 Coursera, Udemy, Class Central (via scraper)
- 🔒 Firebase Auth
- 📅 Google Calendar API

---

## 🧠 Example Use Cases

- Generate a roadmap to become a **UX Designer** in 6 weeks  
- Schedule weekly goals in **Google Calendar**  
- Get AI feedback on your **LinkedIn** or **GitHub**  
- Gamify your week with visual progress + rewards  
- Track progress on your favorite **course** or bootcamp

---

## ⚠️ API Keys & Credentials

To run locally:
- Add `.env` with your OpenAI + Pinecone keys
- Add `credentials.json` for Google Calendar
- Demo works without login

---

## 📂 Folder Structure

career-buddy/
├── app.py
├── requirements.txt
├── .env
├── credentials.json # (not committed)
├── data/
│ └── memo/ # Embedded memory snapshots
└── firebase_helpers.py, langgraph_*.py, progress_tracker.py

---

## 🧪 Run Locally

```
git clone https://github.com/rivapereira123/career-buddy.git
cd career-buddy
pip install -r requirements.txt
python app.py  
```

📰 Blog / Docs (optional)
Coming soon — or add a Medium/Decoding Data Science link here

## ❤️ Enjoyed It?

If you liked Career Buddy:

- ⭐ [Star this repos on GitHub](https://github.com/rivapereira123/career-buddy)
- 📰 [Subscribe to my AI & Learning newsletter](https://www.linkedin.com/pulse/between-coffee-code-issue-1-just-surviving-why-small-wins-pereira-nklkf)*
- ☕ [Buy me a Ko-fi](https://ko-fi.com/your-kofi-name) and fuel more ideas or commission art-!
- 💼 [Connect on LinkedIn](https://linkedin.com/in/riva-pereira/) — always open to collabs!


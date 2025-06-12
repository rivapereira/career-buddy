# 🚀 Career Buddy – Your AI-Powered Career Planner

Career Buddy is a personalized AI career coaching app that blends smart roadmap generation, weekly task planning, reward systems, and real-time web search. Designed for students, early-career professionals, and anyone looking to upskill, it uses a combination of Retrieval-Augmented Generation (RAG), LLM agents, and motivational gamification to keep you consistent and inspired.

---

## 📍 Author
Riva Pereira
AI Residency Cohort 4
University of Wollongong (Dubai)
This app was built as a capstone project to explore agentic AI and educational tools that go beyond static dashboards — into personalized, goal-driven guidance for real people.


## ✨ Features

- 🧠 **AI Roadmap Generator**: Create a personalized 4–6 week plan based on your career goal using GPT-4o, Tavilly search, and dynamic summarization.
- 🔁 **RAG-Based Memory**: Retrieve your past goals, summaries, and courses using Pinecone vector search + LlamaIndex embeddings.
- 📚 **Course Recommender**: Auto-suggests top courses (Coursera, edX, Codecademy, etc.) for Data Analyst, UX Designer, Developer & more.
- 🎯 **Gamified Task Manager**: Add weekly tasks with tags (Critical 🔴, Important 🟠, Optional 🟢), mark completion, and earn rewards.
- 🎁 **Reward System**: Claim motivation boosters like spa day, watch party, or ice cream when milestones are hit.
- 📅 **Calendar Sync**: Integrate with Google Calendar via OAuth to auto-schedule your weekly roadmap tasks.
- 🧑‍💻 **LinkedIn & GitHub Analyzer**: AI feedback on your professional profile and README quality.
- 🧪 **Gradio Interface**: Clean, simple UI using Gradio for demo deployment or local development.

---

## 🛠 Tech Stack

- **Frontend**: Gradio (Python)
- **Backend**: Python (LangChain, LangGraph, OpenAI API, Transformers)
- **Agents**: TaskPlanner, ResourceFinder, MotivationAgent (LangGraph-based)
- **LLMs**: OpenAI GPT-4o, HuggingFace FLAN-T5
- **RAG Engine**: LlamaIndex + Pinecone
- **Auth**: Firebase Auth + Google OAuth
- **Scheduling**: Google Calendar API
- **Search**: Tavilly API (contextual career search)
- **Gamification**: Custom rules + stateful reward logic

## Folder Structure
career-buddy/
├── app.py
├── requirements.txt
├── data/
│   └── memo/                 # Embedded user memory for RAG
├── .env
└── credentials.json          # Google OAuth config (not committed)

##🧩 Agentic AI Design
Career Buddy uses modular AI agents with distinct responsibilities. Example agents:
- 🎓 TaskPlannerAgent: Creates step-by-step weekly roadmaps
- 📚 ResourceFinderAgent: Searches Tavilly for relevant learning content
- 💬 MotivationAgent: Provides rewards and feedback messages
- 🧠 MemoryAgent: Saves summaries and recall data with Pinecone vector embeddings
  
These agents communicate via LangGraph or your preferred multi-agent orchestration framework (e.g., CrewAI or Swarm).

## 🔗 Integrations
- 🗂 Pinecone VectorDB
- 🔍 Tavilly Real-Time Search
-🎓 Coursera, Udemy, Class Central (via links)
- 🔒 Firebase Auth
- 📊 Gradio Dashboard

## 🧠 Example Use Cases
Generate a roadmap to become a UX Designer in 6 weeks
Schedule weekly career goals directly into Google Calendar
Get actionable feedback on your LinkedIn or GitHub README
Gamify your productivity with visual progress tracking and rewards
Add your favorite course and track its completion over time





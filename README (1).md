
# 🎓 Career Buddy: AI-Powered Career Planning Assistant

Career Buddy is an AI-driven productivity and career guidance tool for students and early-career professionals. Designed as a weekly planner and personal coach, it merges task gamification, profile analyzers, LLMs, and retrieval-augmented generation into one seamless platform.

---

## 🚀 Features

### 🔁 Weekly Career Roadmap Generator
- User inputs a career goal (e.g., "UX Designer")
- Uses Tavily → GPT-4o → FLAN-T5 to generate weekly plans
- Stores memory in Pinecone for personalized retrieval
- Visualizes progress with ASCII-style diagrams

### 🧠 Memo-Based RAG System
- User notes are stored in `data/memo/`
- Indexed with HuggingFace embeddings via LlamaIndex
- Enables context-based queries using vector similarity

### 👔 LinkedIn Profile Analyzer
- Input: Headline, About, Experience, Skills, etc.
- Output: Heuristic + regex-based feedback and tips

### 💻 GitHub README Analyzer
- Highlights missing sections and offers improvement checklists

### ✅ Gamified Task & Reward System
- Add tasks with tags like `Critical 🔴`, `Optional 🟢`
- Weekly plan generator with progress tracking
- Rewards like “Ice Cream” and “Spa Day” for 100% completion

### 📅 Google Calendar Sync
- OAuth-integrated event logging
- Fallback option to book via Calendly

### 📚 Course Discovery Engine
- Pulls top courses from Class Central
- Uses Tavily + summarizer to generate course-based plans

---

## ⚙️ Tech Stack

- **Frontend**: Gradio
- **LLM**: OpenAI GPT-4o, Google FLAN-T5
- **RAG**: LlamaIndex + HuggingFace + Pinecone
- **Calendar Integration**: Google Calendar API
- **Search Retrieval**: Tavily API

---

## 🧩 Planned Agentic Architecture

- `TaskPlannerAgent`
- `HabitCoachAgent`
- `GitAnalyzerAgent`, `LinkedInAgent`
- `ResourceFinderAgent`

Designed to evolve into a multi-agent system using LangGraph or CrewAI with memory-enabled task routing.

---

## ✅ What Works Well
- Fast UI prototyping using Gradio
- GPT-4o + T5 combo gives layered insights
- Pinecone memory is responsive and reliable
- Low token cost with high personalization

## ⚠️ What Needs Improvement
- No login/auth; relies on manually re-entered user ID
- No persistent backend (currently ephemeral)
- No export/import of progress
- Agent routing and orchestration not yet live

---

## 🛣️ Roadmap

- [ ] Add agent routing (LangGraph / CrewAI)
- [ ] Firebase Auth & session storage
- [ ] JSON export/import
- [ ] Dynamic reward personalization
- [ ] Token-efficient summarization (emoji/icon-based)

---

## 💡 Why It Matters

- Bridges mentorship gap for underserved students
- Reduces career confusion and procrastination
- Encourages consistent growth via gamified accountability
- Easily integrable into student ecosystems or edtech platforms

---

## 🏁 Conclusion

Career Buddy is more than a productivity app. It's a modular, extensible AI ecosystem designed to scale from a college hackathon MVP into a real-world, enterprise-grade coaching platform. With future upgrades, it can become a multi-agent, memory-augmented tool tailored to modern learning and working environments.

---

🔗 **Built by Riva Pereira | AI Residency Capstone 2025**

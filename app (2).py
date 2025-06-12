import gradio as gr
import datetime
import random
import os
import requests
from pathlib import Path
import openai
from openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from pinecone import Pinecone
import re

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pine_index = pc.Index("career-buddy-memo")
APIFY_TOKEN = os.environ.get("APIFY_TOKEN")

# ==== Constants ====
TASK_DIFFICULTIES = ["Simple", "Moderate", "Challenging"]
TASK_TAGS = ["Critical 🔴", "Important 🟠", "Optional 🟢"]
reward_pool = ["Ice Cream 🍦", "Watch Party 🎬", "Spa Day 💆‍♀️"]
task_data, claimed_rewards, available_rewards = [], [], []
completed_tasks = set()
completed_steps_box = set ()

visual_steps = []
last_reset = datetime.date.today()


#---------------
course_suggestions = {
        "data analyst": [
            ("Google Data Analytics Professional Certificate", "https://www.coursera.org/professional-certificates/google-data-analytics"),
            ("IBM Data Analyst Professional Certificate", "https://www.coursera.org/professional-certificates/ibm-data-analyst"),
            ("Introduction to Data Analytics by IBM", "https://www.coursera.org/learn/introduction-to-data-analytics"),
            ("Excel Basics for Data Analysis by IBM", "https://www.coursera.org/learn/excel-basics-data-analysis"),
            ("Data Analysis using Excel and Tableau by EntryLevel", "https://www.entrylevel.net/post/beginner-data-analysis-courses-by-platform-with-certificates")
        ],
        "ux designer": [
            ("Google UX Design Professional Certificate", "https://www.coursera.org/professional-certificates/google-ux-design"),
            ("Introduction to UI and UX Design by Codecademy", "https://www.codecademy.com/learn/intro-to-ui-ux"),
            ("UX Design Institute's Introduction to UX Design", "https://www.uxdesigninstitute.com/blog/best-free-ux-design-courses-in-2022/"),
            ("Introduction to User Experience Design by Georgia Tech", "https://www.coursera.org/learn/user-experience-design"),
            ("CareerFoundry UX Design Program", "https://careerfoundry.com/en/blog/ux-design/ux-design-course-online/")
        ],
        "software engineer": [
            ("Introduction to Software Engineering by IBM", "https://www.coursera.org/learn/introduction-to-software-engineering"),
            ("Python for Everybody Specialization by University of Michigan", "https://www.coursera.org/specializations/python"),
            ("Full-Stack Engineer Career Path by Codecademy", "https://www.codecademy.com/learn/paths/full-stack-engineer-career-path"),
            ("Software Engineering for Beginners by Udemy", "https://www.udemy.com/course/software-engineering-for-beginners/"),
            ("Software Engineering Bootcamp by TripleTen", "https://tripleten.com/software-engineer/")
        ],
        "digital marketing": [
            ("Fundamentals of Digital Marketing by Google Digital Garage", "https://learndigital.withgoogle.com/digitalgarage/course/digital-marketing"),
            ("Digital Marketing Specialization by Coursera", "https://www.coursera.org/specializations/digital-marketing"),
            ("The Complete Digital Marketing Course by Udemy", "https://www.udemy.com/course/learn-digital-marketing-course/"),
            ("Digital Marketing Fundamentals by University of Edinburgh on edX", "https://www.edx.org/course/digital-marketing-fundamentals"),
            ("Digital Marketing Course by CareerFoundry", "https://careerfoundry.com/en/blog/digital-marketing/online-digital-marketing-courses/")
        ],
        "project manager": [
            ("Google Project Management Professional Certificate", "https://www.coursera.org/professional-certificates/google-project-management"),
            ("Foundations of Project Management by Coursera", "https://www.coursera.org/learn/project-management-foundations"),
            ("Project Management Basics by PMI", "https://www.pmi.org/learning/free-online-courses"),
            ("Introduction to Project Management by University of Adelaide on edX", "https://www.edx.org/course/introduction-to-project-management"),
            ("Project Management Principles and Practices Specialization by Coursera", "https://www.coursera.org/specializations/project-management")
        ]
    }

import difflib

def get_courses_for_goal(goal_key):
    if goal_key not in course_suggestions:
        match = difflib.get_close_matches(goal_key, course_suggestions.keys(), n=1, cutoff=0.6)
        if match:
            goal_key = match[0]
    return course_suggestions.get(goal_key, [])



#-------

class RoadmapUnlockManager:
    def __init__(self):
        self.weekly_steps = {}
        self.current_week = "Week 1"
        self.completed_tasks = set()

    def load_steps(self, steps: list[str]):
        self.weekly_steps = {}
        current_label = None

        for step in steps:
            stripped = step.strip().strip("*")
            if stripped.lower().startswith("week"):
                current_label = stripped.split(":")[0].strip()  # e.g. "Week 1"
                self.weekly_steps[current_label] = []
            elif current_label:
                # Only store valid actionable items
                self.weekly_steps[current_label].append(stripped)

        self.current_week = list(self.weekly_steps.keys())[0] if self.weekly_steps else "Week 1"
        self.completed_tasks.clear()

    def get_current_choices(self):
        return [
            s for s in self.weekly_steps.get(self.current_week, [])
            if not s.lower().startswith("week") and not s.startswith("**")
        ]

    

    def get_current_week_title(self):
        return f"**📅 Current Focus: {self.current_week}**"

    def get_current_choices(self):
        return self.weekly_steps.get(self.current_week, [])

    def update_completion(self, selected):
        self.completed_tasks.update(selected)
        all_current = set(self.get_current_choices())
        if all_current.issubset(self.completed_tasks):
            return self._unlock_next_week()
        return f"✅ Progress: {len(self.completed_tasks)}/{len(all_current)}"

    def _unlock_next_week(self):
        weeks = list(self.weekly_steps.keys())
        current_index = weeks.index(self.current_week)
        if current_index + 1 < len(weeks):
            self.current_week = weeks[current_index + 1]
            self.completed_tasks.clear()
            return f"🎉 All tasks done! Unlocked: {self.current_week}"
        return "✅ All weeks completed!"


#----------

def greet_user(uid, goal):
    feedback = f"✅ Welcome back, **{uid}**!"
    recalled = recall_from_memory(uid, goal)
    return feedback, recalled


# ==== Embedding & Summarizer Setup ====
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
SCOPES = ["https://www.googleapis.com/auth/calendar"]
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
print("Tavilly key (first 5 chars):", TAVILY_API_KEY[:5] if TAVILY_API_KEY else "NOT FOUND")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model
summarizer = pipeline("text2text-generation", model="google/flan-t5-base")


Path("data/memo").mkdir(parents=True, exist_ok=True)

import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()



# ==== Load RAG Vector Index ====
def load_docs():
    try:
        docs = SimpleDirectoryReader("data/memo").load_data()
        return VectorStoreIndex.from_documents(docs).as_query_engine()
    except Exception as e:
        print("❌ Error loading RAG docs:", e)
        return None

memo_rag_engine = load_docs()



# ==== Ingest Popular Courses from Class Central ====
def batch_ingest_from_classcentral():
    course_descriptions = [
        "CS50’s Introduction to Computer Science from Harvard University",
        "Google Data Analytics from Google",
        "Neural Networks and Deep Learning from DeepLearning.AI",
        "Python for Everybody from University of Michigan",
        "Introduction to Psychology from Yale University",
        "Foundations of User Experience (UX) Design from Google",
        "Financial Markets from Yale University",
        "Introduction to Data Science in Python from University of Michigan",
        "AI For Everyone from DeepLearning.AI",
        "Introduction to HTML5 from University of Michigan"
    ]
    for title in course_descriptions:
        try:
            response = requests.post("https://api.tavily.com/search", json={
                "api_key": TAVILY_API_KEY,
                "query": title,
                "include_answer": True
            }, timeout=15)
            response.raise_for_status()
            answer = response.json().get("answer", "")
            if not answer:
                continue
                summary = summarizer(f"Summarize this course for roadmap purposes:\n{answer}", max_new_tokens=300)[0]["generated_text"]
            goal = title.split(" from ")[0].strip().lower().replace(" ", "_")
            save_to_rag(goal, answer + "\n\n---\n" + summary)

            print(f"✅ Ingested: {title}")
        except Exception as e:
            print(f"❌ Failed to ingest {title}: {e}")
                        

# ==== Save Tavilly Result to RAG ====
def save_to_rag(goal, content):
    goal_slug = goal.lower().replace(" ", "_")
    path = Path(f"data/memo/{goal_slug}_tavilly.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"📄 Saved to: {path}")

#===============
def is_headless():
    return os.environ.get("HF_SPACE_ID") is not None

# ==== RAG from Memo ====
def call_rag(goal):
    # Load saved content
    path = Path(f"data/memo/{goal.lower().replace(' ', '_')}_tavilly.txt")
    if not path.exists():
        return "❌ No memory found for this goal yet. Try running Tavilly first."

    # Example fixed formula-based roadmap
    base_plan = f"""
## 📅 4-Week Roadmap for Becoming a {goal.title()}
### 🎓 Step 1: Choose a Top-Rated Course
- Search for a course on Coursera, edX, or Class Central.
- Prefer those with ★★★★☆ or ★★★★★.
- Example: Google {goal.title()} Certificate.
### 💰 Step 2: Check Accessibility
- ✅ Can you audit it for free?
- 💳 Can you afford a paid certificate?
- 🎓 See if your university provides access.
### 🧠 Step 3: Weekly Breakdown
- **Week 1–3**: Complete 75% of the course.
- **Week 4**: Build a project related to the course topic.
    - Example: For UX → Design a landing page wireframe
    - For Data → Create a dashboard in Google Sheets or Tableau
### 📌 Tip:
Document your work in Notion or a public portfolio. Practice explaining your learnings.
---
📚 Course inspiration: https://www.classcentral.com/report/most-popular-online-courses/
"""
    return clean_text(base_plan)

#=========GOOGLE CALENDAR BUISNESS ==========
#====================================================

def save_to_memory(user_id, goal, summary, steps, courses):
    try:
        from datetime import datetime
        text_blob = f"Goal: {goal}\nSummary: {summary}\nSteps: {' | '.join(steps)}\nCourses: {' | '.join([c[0] for c in courses])}"
        embedding = embed_model.embed_query(text_blob)
        metadata = {
            "user_id": user_id,
            "goal": goal,
            "summary": summary,
            "steps": steps,
            "courses": [f"{c[0]} | {c[1]}" for c in courses],
            "timestamp": datetime.utcnow().isoformat()
        }
        pine_index.upsert([(user_id + ":" + goal.replace(" ", "_"), embedding, metadata)])
        return True
    except Exception as e:
        print(f"❌ Failed to save memory: {e}")
        return False

def recall_from_memory(user_id, goal):
    try:
        query = user_id + ":" + goal.replace(" ", "_")
        result = pine_index.fetch([query])  # ✅ returns a FetchResponse object
        
        if query not in result.vectors:
            return "❌ No saved plan found for this goal."

        metadata = result.vectors[query].get("metadata", {})
        steps = metadata.get("steps", [])
        summary = metadata.get("summary", "")
        courses = metadata.get("courses", [])
        course_section = ""

        diagram = render_text_roadmap(goal, steps)
        
        if courses:
            course_section = "\n\n### 📚 Recommended Courses\n" + "\n".join([f"- [{c['name']}]({c['url']})" for c in courses if 'name' in c and 'url' in c])

        return f"""### 🔁 Recalled Plan for {goal}

{diagram}

{summary}{course_section}

**🗓 Book your weekly study check-in:** [Click here]({CALENDLY_LINK})
"""
    except Exception as e:
        return f"❌ Error recalling memory: {e}"

        
# ==== Tavilly + Summary + Course Suggestion ====
# UI wiring reminder:
# rag_button.click(fn=call_tavilly_rag, inputs=career_goal, outputs=rag_output)
# Ensure gr.Markdown() is assigned to rag_output
def call_tavilly_rag(user_id, goal):
    completed_tasks.clear()

    if not TAVILY_API_KEY:
        return "❌ Tavilly API key not found.", "", []

    try:
        headers = {"Authorization": TAVILY_API_KEY}
        payload = {
            "query": f"{goal} career weekly roadmap",
            "search_depth": "advanced",
            "include_answer": True
        }
        response = requests.post("https://api.tavily.com/search", headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        web_content = result.get("answer", "")
        if len(web_content.split()) < 100:
            web_content += "\n\nSuggested steps: Learn Figma, build portfolio, network, and apply for internships."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Tavilly error: {str(e)}", "", []


    try:
        # Get 6 week short roadmap steps
        messages = [
            {"role": "system", "content": "Create a personalized 6-step weekly career roadmap. The roadmap should be goal-focused and iterative — each step should build upon the previous one. Encourage the user to start by selecting a course from the recommended list, then move toward applying that knowledge through projects, certifications, or content creation. End the roadmap by demonstrating expertise (e.g., GitHub repo, portfolio update, mock interview). Each step should be 1–2 sentences and mention a clear action, resource, and milestone outcome."}
        ]

        client = openai.OpenAI()
        res = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=300, temperature=0.5)
        response_text = res.choices[0].message.content
        raw_steps = response_text.split("\n")

        steps = [s.strip("* ").strip() for s in raw_steps if s.strip() and not s.strip().lower().startswith("**week")]

        if not steps:
            print("⚠️ No valid steps found from LLM — using fallback tasks. Tavilly can't find it, maybe RAG can-?")
            steps = [
                "Action: Find a course or learn the skill by hand. Head to Memo with your research.",
                "Resource: Watch the first 2 modules.",
                "Milestone: Create a short reflection post on what you learned."
            ]

        diagram = render_text_roadmap(goal, steps)

        # Summarize with FLAN-T5
        prompt = f"Create a weekly roadmap for someone becoming a {goal}. Use:\n{web_content}"
        summary = summarizer(prompt, max_new_tokens=300, do_sample=False)[0]["generated_text"]

        goal_key = goal.lower().strip()
        courses = course_suggestions.get(goal_key, [])
        course_section = "\n\n### 📚 Recommended Courses\n" + "\n".join([f"- [{name}]({url})" for name, url in courses]) if courses else ""

        save_to_memory(user_id, goal, summary, steps, courses)

        return f"""
### 🧠 Weekly Plan for {goal}

```
{diagram}
```

{summary}{course_section}

**🗓 Book your weekly study check-in:** [Click here]({CALENDLY_LINK})
""", "", steps

    except Exception as e:
        print(f"❌ GPT-4o fallback failed: {e}")
        fallback_steps = [
            "Action: Search YouTube or Coursera for a beginner course.",
            "Resource: Choose any free learning platform.",
            "Milestone: Finish one hour of learning and reflect."
        ]
        diagram = render_text_roadmap(goal, fallback_steps)
        fallback_summary = "This is a basic roadmap you can follow to get started until dynamic generation is fixed."

        return f"""
### 🧠 Starter Plan for {goal}

```
{diagram}
```

{fallback_summary}

**🗓 Book your weekly study check-in:** [Click here]({CALENDLY_LINK})
""", "", fallback_steps

#===================================================================================================================
def mark_step_completed(user_id, step):
    completed_tasks.add(step)
    return f"✅ Completed: {step}"

def calculate_visual_progress(user_id, selected_steps):
    completed_tasks.clear()  # ✅ Reset on change
    for step in selected_steps:
        if step in visual_steps:
            completed_tasks.add(step)
    return f"✅ Progress: {len(completed_tasks)} steps done!"


    
def reset_weekly_data():
    global task_data, claimed_rewards, available_rewards, last_reset
    if datetime.date.today() != last_reset:
        task_data.clear()
        claimed_rewards.clear()
        available_rewards.clear()
        last_reset = datetime.date.today()

def add_task(user_id, task, duration, difficulty, tag):
    reset_weekly_data()
    if not task:
        return "⚠️ Please enter a task.", gr.update()
    for t in task_data:
        if t['Task'].strip().lower() == task.strip().lower():
            return clean_text("⚠️ Task already exists. Please enter a unique task."), gr.update()
    task_data.append({"Task": task.strip(), "Duration": duration, "Difficulty": difficulty, "Tag": tag})
    return display_tasks(), ""

def display_tasks():
    if not task_data:
        return "No tasks yet."
    display = "| Task | Duration | Difficulty | Tag |\n|------|----------|------------|-----|\n"
    for t in task_data:
        emoji = {"Critical 🔴": "🔴", "Important 🟠": "🟠", "Optional 🟢": "🟢"}.get(t["Tag"], "")
        display += f"| {t['Task']} | {t['Duration']}hr | {t['Difficulty']} | {t['Tag']} {emoji} |\n"
    return display

def add_reward(new_reward):
    if new_reward and new_reward not in reward_pool:
        reward_pool.append(new_reward)
    return gr.update(choices=reward_pool, value=reward_pool)

def run_gpt_fallback(goal):
    messages = [
        {
            "role": "system",
            "content": (
                "Create a personalized 6-step weekly career roadmap for becoming a "
                f"{goal}. The roadmap should be goal-focused and iterative — each step "
                "should build upon the previous one. Encourage the user to start with a course, then move toward applying that knowledge. "
                "Each step must include an action, a resource, and a milestone."
            )
        }
    ]

    client = openai.OpenAI()
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300,
        temperature=0.5
    )

    response_text = res.choices[0].message.content
    raw_steps = response_text.split("\n")
    steps = [s.strip("* ").strip() for s in raw_steps if s.strip() and not s.strip().lower().startswith("**week")]

    if not steps:
        raise ValueError("GPT fallback returned no usable steps")

    return steps




def run_gpt_fallback(goal):
    messages = [
        {
            "role": "system",
            "content": (
                "Create a personalized 6-step weekly career roadmap for becoming a "
                f"{goal}. The roadmap should be goal-focused and iterative — each step "
                "should build upon the previous one. Encourage the user to start with a course, then move toward applying that knowledge. "
                "Each step must include an action, a resource, and a milestone."
            )
        }
    ]

    client = openai.OpenAI()
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300,
        temperature=0.5
    )

    response_text = res.choices[0].message.content
    raw_steps = response_text.split("\n")
    steps = [s.strip("* ").strip() for s in raw_steps if s.strip() and not s.strip().lower().startswith("**week")]

    if not steps:
        raise ValueError("GPT fallback returned no usable steps")

    return steps



def generate_smart_plan(user_id, start_date_str, goal):
    import datetime
    import difflib

    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    
    if not user_id:
        return "❌ Please enter a nickname in the Welcome tab.", "", "📅 Week 1"
    user_key = user_id.strip().lower()
    goal_key = goal.lower().strip()
    
    if goal_key not in course_suggestions:
        close_matches = difflib.get_close_matches(goal_key, course_suggestions.keys(), n=1, cutoff=0.6)
    if close_matches:
        goal_key = close_matches[0]

    
    steps = []
    plan_markdown = ""
    plan_source = ""
    summary = ""
    courses = []

    # 1️⃣ Try RAG
    recalled = recall_from_memory(user_key, goal_key)
    if recalled and "❌" not in recalled:
        print("✅ RAG hit")
        steps = recalled.split("\n")
        plan_markdown = f"### 🧠 Plan for {goal}\n\n_Recalled from memory_\n\n" + "\n".join(steps)
        plan_source = "🔁 Loaded from memory (RAG)"
        
    if not steps:
        print("🧪 Trying Tavilly fallback for:", goal_key)

    try:
        result = call_tavilly_rag(user_id, goal_key)
        print("✅ Tavilly returned something:", result)

        tav_plan = result[0]
        tav_plan = re.sub(r'(Create a weekly roadmap.*?)\1+', r'\1', tav_plan, flags=re.DOTALL)

        tav_steps = result[2]

        if tav_steps and len(tav_steps) > 1:
            print("🌐 Tavilly used")
            steps = tav_steps
            plan_markdown = tav_plan
            plan_source = "🌐 Generated via Tavilly (web search)"

            courses = get_courses_for_goal(goal_key)  # <-- use the fuzzy look

        if courses:
            course_section = (
        "\n\n### 📚 Recommended Courses\n" +
        "\n".join([f"- [{name}]({url})" for name, url in courses]))
            
        else:
            course_section = ""

        plan_markdown += course_section
        pass 

    except Exception as e:
        print(f"⚠️ Tavilly failed: {e}")


    # 3️⃣ Try GPT-4o fallback
    if not steps:
        try:
            steps = run_gpt_fallback(goal_key)
            print("🤖 GPT-4o fallback used")
            plan_source = "🤖 Generated with GPT-4o fallback"

            diagram = render_text_roadmap(goal, steps)
            summary = summarizer(f"Create a weekly roadmap for {goal}.", max_new_tokens=300, do_sample=False)[0]["generated_text"]

            courses = course_suggestions.get(goal_key, [])
            if courses:
                course_section = ("\n\n### 📚 Recommended Courses\n" +"\n".join([f"- [{name}]({url})" for name, url in courses]))
            else:
                course_section = ""

            plan_markdown = f"""
### 🧠 Career Plan for {goal}

```
{diagram}
```

{summary}{course_section}

📅 To schedule your plan or check-ins, visit the **Schedule Tab**.

---

_Source: {plan_source}_
"""
            save_to_memory(user_id, goal, summary, steps, courses)

        except Exception as e:
            print(f"❌ GPT fallback failed: {e}")

    # 4️⃣ Static fallback
    if not steps:
        print("📦 Using static backup")
        steps = [
            "Action: Search Coursera for a beginner course.",
            "Resource: Watch at least 1 hour and reflect.",
            "Milestone: Write a journal entry on what you learned."
        ]
        plan_markdown = f"""
### 🧠 Static Plan for {goal}

Start with some basic research and courses. Track progress in the Courses tab.

📅 To schedule your plan or check-ins, visit the **Schedule Tab**.

---

_Source: 📦 Static backup used_
"""
        plan_source = "📦 Static backup used"

    # 🔁 Push steps into Memo Task List
    for step in steps:
        add_task(user_id, task=step, duration=2, difficulty="Moderate", tag="Important 🟠")

    return plan_markdown, gr.update(choices=steps, value=[]), "📅 Week 1"



def calculate_progress(user_id, completed):
    completed_count = len(completed)
    total = len(task_data)
    percent = int((completed_count / total) * 100) if total else 0
    points = completed_count * 25
    bar = f"[{'█' * (percent // 10)}{'-' * (10 - percent // 10)}]"
    global available_rewards
    available_rewards = reward_pool if percent == 100 else reward_pool[:2] if percent >= 50 else reward_pool[:1]
    return f"Progress: {bar} {percent}%  Points: {points} / {total * 25}", completed, task_data

def claim_reward(completed, tasks):
    if not available_rewards:
        return gr.update(value="🔒 No rewards unlocked yet.")

    if len(claimed_rewards) >= 1:
        return gr.update(value="⛔ Already claimed reward this week.")

    chosen = random.choice(available_rewards)
    claimed_rewards.append(chosen)

    return gr.update(value=f"""
    <div style='border: 2px solid #FFD700; padding: 12px; background: #fff3cd; font-size: 18px; border-radius:10px;'>
        🎉 <strong>Reward Unlocked!</strong><br><br>
        <span style='font-size: 22px;'>✨ You claimed: <strong>{chosen}</strong> 🎁</span><br><br>
        Past Rewards: {" ".join(claimed_rewards)}
    </div>
    """)


def add_course_to_memo(course_title):
    reset_weekly_data()
    task = f"Finish Week 1 of {course_title}"
    for t in task_data:
        if t['Task'].strip().lower() == task.strip().lower():
            return "⚠️ Course task already added."
    task_data.append({
        "Task": task,
        "Duration": 3,
        "Difficulty": "Moderate",
        "Tag": "Critical 🔴"
    })
    return display_tasks()

#------------------------------------------------------------

from pathlib import Path
import re
from transformers import pipeline

# Initialize summarizer again after state reset
summarizer = pipeline("text2text-generation", model="google/flan-t5-base")



# Clean noisy LinkedIn input text
def clean_linkedin_input(text):
    junk_patterns = [
        r"Add verification badge", r"Contact info", r"followers", r"connections",
        r"Add profile section", r"Enhance profile", r"Open to work.*?Show details",
        r"Show all analytics", r"Get started", r"Edit", r"See more", r"…see more",
        r"Subscribe", r"View .*? graphic link", r"Activate to view larger image",
        r"Create a post", r"Loaded .*? posts", r"Visible to anyone.*?", r"· Remote",
        r"\d+\s+(followers|connections|comments|likes?)", r"Issued .*?",
        r"Posts.*?Comments.*?Videos.*?Images.*?Newsletter", r"Show all .*?",
        r"–", r"—", r"…"
    ]
    for pattern in junk_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def clean_name_headline_section(text):
    lines = text.splitlines()
    debug = []

    name = None
    headline = None
    location = None
    followers = None
    open_roles = None

    for line in lines:
        line = line.strip()

        # Name is usually first and alphabetic
        if not name and line and line[0].isalpha() and " " in line:
            name = line
            debug.append(f"👤 Name: {name}")
            continue

        # Headline: usually comes after name or has "•" or "|"
        if not headline and ("•" in line or "|" in line):
            headline = line
            debug.append(f"🧠 Headline: {headline}")
            continue

        # Location
        if "united arab emirates" in line.lower() or "city" in line.lower():
            location = line
            debug.append(f"📍 Location: {location}")
            continue

        # Followers
        if "followers" in line.lower():
            followers = line
            debug.append(f"📊 {followers}")
            continue

        # Open to work
        if "open to" in line.lower() and "roles" in line.lower():
            open_roles = line
            debug.append(f"💼 {open_roles}")
            continue

    feedback = []
    if not name:
        feedback.append("⚠️ Your full name is missing or unclear.")
    if not headline:
        feedback.append("⚠️ Headline/tagline is missing. Add a short, keyword-rich sentence.")
    if not location:
        feedback.append("⚠️ Location info not found. Add your city for recruiters.")
    if not followers:
        feedback.append("🔍 Tip: Add or grow your follower count for visibility.")
    if not open_roles:
        feedback.append("📣 Mention your 'Open to work' roles clearly in your profile.")

    feedback.append("📸 Bonus: Did you upload a banner and cover photo? If not, add one to personalize your profile!")

    return "\n".join(feedback), "\n".join(debug)



def analyze_apify_name_headline(row):
    feedback = []
    debug = []

    name = row.get("fullName", "").strip()
    headline = row.get("headline", "").strip()
    location = row.get("location", "").strip()
    followers = row.get("followersCount", "")
    open_to_work = row.get("openToWork", "")

    # Debug info
    if name:
        debug.append(f"👤 Name: {name}")
    if headline:
        debug.append(f"🧠 Headline: {headline}")
    if location:
        debug.append(f"📍 Location: {location}")
    if followers:
        debug.append(f"📊 Followers: {followers}")
    if open_to_work:
        debug.append(f"💼 Open to: {open_to_work}")

    # Feedback logic
    if not name:
        feedback.append("⚠️ Your full name is missing or unclear.")
    if not headline:
        feedback.append("⚠️ Headline/tagline is missing. Add a short, keyword-rich sentence.")
    if not location:
        feedback.append("⚠️ Location info not found. Add your city for recruiters.")
    if not followers:
        feedback.append("🔍 Tip: Add or grow your follower count for visibility.")
    if not open_to_work:
        feedback.append("📣 Mention your 'Open to work' roles clearly in your profile.")
    
    feedback.append("📸 Bonus: Did you upload a banner and cover photo? If not, add one to personalize your profile!")

    return f"### 🧾 Name & Headline\n" + "\n".join(feedback) + "\n\n<details><summary>Debug</summary>\n" + "\n".join(debug) + "\n</details>"


def clean_about_section(text):
    feedback = []
    debug_info = []

    # Remove repeated lines
    sentences = list(dict.fromkeys(text.strip().split('.')))
    cleaned_text = '. '.join([s.strip() for s in sentences if s.strip()])

    debug_info.append(f"🧹 Cleaned Sentences Count: {len(sentences)}")
    debug_info.append(f"📝 Cleaned Text:\n{cleaned_text[:500]}...")

    # Heuristics
    if len(cleaned_text) < 200:
        feedback.append("⚠️ Your About section seems short. Aim for 3-5 strong paragraphs.")
    if cleaned_text.lower().count("i am") + cleaned_text.lower().count("i’m") == 0:
        feedback.append("🤔 Add more personal voice. Use 'I am...' or 'I'm...' to connect with the reader.")
    if "impact" in cleaned_text.lower() and "mentor" in cleaned_text.lower():
        feedback.append("✅ Nice! You’re showing leadership and purpose.")

    # Detect keyword stuffing
    keywords = ['python', 'machine learning', 'data', 'power bi', 'ai', 'artificial intelligence']
    keyword_hits = [kw for kw in keywords if cleaned_text.lower().count(kw) > 2]
    if keyword_hits:
        feedback.append(f"⚠️ These keywords are mentioned too often: {', '.join(keyword_hits)}. Avoid overusing them.")

    return "\n".join(feedback), "\n".join(debug_info)


def analyze_apify_about_section(row):
    feedback = []
    debug_info = []

    about = row.get("about", "").strip()

    if not about or len(about) < 10:
        return "### 📘 About Section\n⚠️ No About section found or it’s too short."

    # Clean repetitions and spacing
    sentences = list(dict.fromkeys(about.strip().split('.')))
    cleaned_text = '. '.join([s.strip() for s in sentences if s.strip()])

    debug_info.append(f"🧹 Cleaned Sentences Count: {len(sentences)}")
    debug_info.append(f"📝 Cleaned Text:\n{cleaned_text[:500]}...")

    # Heuristics
    if len(cleaned_text) < 200:
        feedback.append("⚠️ Your About section seems short. Aim for 3–5 strong paragraphs.")

    personal_voice = cleaned_text.lower().count("i am") + cleaned_text.lower().count("i’m")
    if personal_voice == 0:
        feedback.append("🤔 Add more personal voice. Use 'I am...' or 'I'm...' to connect with the reader.")

    if "impact" in cleaned_text.lower() and "mentor" in cleaned_text.lower():
        feedback.append("✅ Nice! You’re showing leadership and purpose.")

    # Keyword stuffing check
    keywords = ['python', 'machine learning', 'data', 'power bi', 'ai', 'artificial intelligence']
    keyword_hits = [kw for kw in keywords if cleaned_text.lower().count(kw) > 2]
    if keyword_hits:
        feedback.append(f"⚠️ These keywords are mentioned too often: {', '.join(keyword_hits)}. Try spreading them more naturally.")

    return (
        "### 📘 About Section\n" +
        "\n".join(feedback) +
        "\n\n<details><summary>Debug</summary>\n" +
        "\n".join(debug_info) +
        "\n</details>"
    )





import re

def analyze_experience_section(text):
    feedback, debug = [], []
    total_skills = set()

    # Normalize
    text = re.sub(r"(logo|pdf).*?\.pdf", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.?\s*see more", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    entries = re.split(r"(?:\d{4}.*?(?:mo|mos|yr|yrs))", text)

    work_types = {"remote": 0, "hybrid": 1, "onsite": 2}
    format_score = 0
    short_roles = 0
    roles_found = 0

    for entry in entries:
        if len(entry.strip()) < 50:
            continue
        roles_found += 1

        # Determine format
        format_detected = "onsite"
        if "remote" in entry.lower():
            format_detected = "remote"
        elif "hybrid" in entry.lower():
            format_detected = "hybrid"
        format_score += work_types[format_detected]

        # Detect time span
        if re.search(r"(\d+\s*(mo|mos|yr|yrs))", entry):
            months = sum([
                int(x) if "mo" in unit else int(x) * 12
                for x, unit in re.findall(r"(\d+)\s*(mo|mos|yr|yrs)", entry)
            ])
            if months < 3:
                short_roles += 1
        else:
            feedback.append("⚠️ One experience entry is missing a time span.")

        # Remove duplicated bullets and body
        cleaned_entry = re.sub(r"(▶️.*?)(\1)+", r"\1", entry)

        # Extract skills
        skill_matches = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b", cleaned_entry)
        for s in skill_matches:
            if len(s) <= 20:
                total_skills.add(s.strip().lower())

    # Summary logic
    if roles_found == 0:
        feedback.append("⚠️ Couldn't find valid experience entries. Double-check formatting.")
    else:
        feedback.append(f"✅ Found **{roles_found}** experience roles.")
        if short_roles > 0:
            feedback.append(f"🕒 {short_roles} roles seem too short (<3 months). Consider explaining a little on these roles, Use Harvard Referencing Words.")
        if format_score / max(1, roles_found) < 1.2:
            feedback.append("📍 Most of your roles are **Remote** or **Hybrid**. Consider getting (if you can-easier said then done) onsite or longer-term internships for variety.")
        feedback.append(f"🧠 Extracted **{len(total_skills)}** possible skills so far.")

    debug.append("🛠 Extracted Sample Skills:\n" + ", ".join(list(total_skills)[:20]))
    debug.append(f"🔎 Total Raw Experience Entries: {roles_found}")

    return "\n".join(feedback), "\n".join(debug), total_skills



def analyze_apify_experience_section(row):
    import re

    text = row.get("experience", "")
    feedback, debug = [], []
    total_skills = set()

    # Normalize
    text = re.sub(r"(logo|pdf).*?\.pdf", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.?\s*see more", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    entries = re.split(r"(?:\d{4}.*?(?:mo|mos|yr|yrs))", text)

    work_types = {"remote": 0, "hybrid": 1, "onsite": 2}
    format_score = 0
    short_roles = 0
    roles_found = 0

    for entry in entries:
        if len(entry.strip()) < 50:
            continue
        roles_found += 1

        # Determine format
        format_detected = "onsite"
        if "remote" in entry.lower():
            format_detected = "remote"
        elif "hybrid" in entry.lower():
            format_detected = "hybrid"
        format_score += work_types[format_detected]

        # Detect time span
        if re.search(r"(\d+\s*(mo|mos|yr|yrs))", entry):
            months = sum([
                int(x) if "mo" in unit else int(x) * 12
                for x, unit in re.findall(r"(\d+)\s*(mo|mos|yr|yrs)", entry)
            ])
            if months < 3:
                short_roles += 1
        else:
            feedback.append("⚠️ One experience entry is missing a time span.")

        # Clean and extract skills
        cleaned_entry = re.sub(r"(▶️.*?)(\1)+", r"\1", entry)
        skill_matches = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b", cleaned_entry)
        for s in skill_matches:
            if len(s) <= 20:
                total_skills.add(s.strip().lower())

    # Summary
    if roles_found == 0:
        feedback.append("⚠️ Couldn't find valid experience entries. Double-check formatting.")
    else:
        feedback.append(f"✅ Found **{roles_found}** experience roles.")
        if short_roles > 0:
            feedback.append(f"🕒 {short_roles} roles seem too short (<3 months). Consider explaining these briefly.")
        if format_score / max(1, roles_found) < 1.2:
            feedback.append("📍 Most roles are **Remote** or **Hybrid**. If possible, seek an onsite or longer-term internship.")
        feedback.append(f"🧠 Extracted **{len(total_skills)}** possible skills from your experience.")

    debug.append("🛠 Sample Skills:\n" + ", ".join(list(total_skills)[:20]))
    debug.append(f"🔎 Raw Experience Entries Count: {roles_found}")

    return (
        "### 💼 Experience Insight\n" +
        "\n".join(feedback) +
        "\n\n<details><summary>Debug</summary>\n" +
        "\n".join(debug) +
        "\n</details>"
    )




import re

def analyze_education_section(text):
    if not text or len(text.strip()) < 30:
        return "⚠️ Your education section looks empty or too short. Add your university, field of study, and time period."

    suggestions = []

    # Check for institution and degree/field
    has_university = re.search(r"(university|college|institute|school)", text, re.IGNORECASE)
    has_field = re.search(r"(computer|data|science|engineering|business|design|marketing|ai|big data|cs|it)", text, re.IGNORECASE)
    has_dates = re.search(r"\b20\d{2}\b", text)

    if not has_university:
        suggestions.append("🎓 Add your **university or institution name**.")
    if not has_field:
        suggestions.append("📘 Add your **field of study** like Data Science, Business, or AI.")
    if not has_dates:
        suggestions.append("📅 Include your **education timeline**, like 2022–2025.")

    # Skill extraction (optional)
    skills_found = re.findall(r"[A-Za-z]{3,}", text)
    if len(skills_found) < 3:
        suggestions.append("🧠 List **a few relevant skills** you learned (e.g., Python, SQL, Problem Solving).")

    if not suggestions:
        return "✅ Your education section looks complete and informative!"
    else:
        return "\n".join(suggestions), "No debug info"



import re

def analyze_apify_education_section(row):
    text = row.get("education", "")
    
    if not text or len(text.strip()) < 30:
        return "### 🎓 Education Insight\n⚠️ Your education section looks empty or too short. Add your university, field of study, and time period."

    suggestions = []

    # Checks
    has_university = re.search(r"(university|college|institute|school)", text, re.IGNORECASE)
    has_field = re.search(r"(computer|data|science|engineering|business|design|marketing|ai|big data|cs|it)", text, re.IGNORECASE)
    has_dates = re.search(r"\b20\d{2}\b", text)

    if not has_university:
        suggestions.append("🎓 Add your **university or institution name**.")
    if not has_field:
        suggestions.append("📘 Add your **field of study** like Data Science, Business, or AI.")
    if not has_dates:
        suggestions.append("📅 Include your **education timeline**, like 2022–2025.")

    # Skill words (optional heuristic)
    skills_found = re.findall(r"[A-Za-z]{3,}", text)
    if len(skills_found) < 3:
        suggestions.append("🧠 List **a few relevant skills** you learned (e.g., Python, SQL, Problem Solving).")

    if not suggestions:
        return "### 🎓 Education Insight\n✅ Your education section looks complete and informative!"
    else:
        return "### 🎓 Education Insight\n" + "\n".join(suggestions)




import re

def analyze_skills_section(text):
    if not text or len(text.strip()) < 20:
        return "😬 DUDE PUT SOMETHING IN THERE. Add your technical, analytical, or soft skills. This helps with visibility and matching."

    # Clean noisy parts
    clean = re.sub(r"Company logo.*?", "", text, flags=re.DOTALL)
    clean = re.sub(r"Show all \d+ details", "", clean)
    clean = re.sub(r"\b\d+ endorsement[s]?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s{2,}", " ", clean)
    clean = re.sub(r"[^\x00-\x7F]+", "", clean)  # remove emojis, logos, etc.

    # Extract skills
    lines = clean.splitlines()
    skills = set()
    endorsements = 0
    for line in lines:
        skill = line.strip()
        if skill.lower().endswith("endorsement"):
            endorsements += 1
        elif len(skill.split()) < 5 and not re.search(r'\d', skill) and len(skill) > 2:
            skills.add(skill)

    feedback = []
    skill_count = len(skills)

    # Skill quantity logic
    if skill_count == 0:
        feedback.append("😬 You didn’t list any skills. Add at least 5–10 to improve discoverability.")
    elif skill_count < 10:
        feedback.append(f"🧠 You listed {skill_count} skills. Maybe add more as you study and grow.")
    elif skill_count < 50:
        feedback.append(f"✅ You have {skill_count} skills — solid! Most professionals have up to 50 over time.")
    else:
        feedback.append(f"🔥 You’ve listed {skill_count}+ skills — that's fantastic!")

    # Endorsements check
    if endorsements == 0:
        feedback.append(
            "📣 None of your skills are endorsed. Ask your friends, lecturers, or mentors to endorse them. "
            "They just need to visit your profile, scroll to skills, and click **Endorse**."
        )
    else:
        feedback.append(f"👍 You’ve got {endorsements} endorsement{'s' if endorsements > 1 else ''} — nice!")

    return "\n".join(feedback), "No debug info"



import re

def analyze_apify_skills_section(row):
    text = row.get("skills", "")
    
    if not text or len(text.strip()) < 20:
        return "### 🧠 Skills Insight\n😬 DUDE PUT SOMETHING IN THERE. Add your technical, analytical, or soft skills. This helps with visibility and matching."

    # Clean up the scraped junk
    clean = re.sub(r"Company logo.*?", "", text, flags=re.DOTALL)
    clean = re.sub(r"Show all \d+ details", "", clean)
    clean = re.sub(r"\b\d+ endorsement[s]?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s{2,}", " ", clean)
    clean = re.sub(r"[^\x00-\x7F]+", "", clean)

    # Parse out skills
    lines = clean.splitlines()
    skills = set()
    endorsements = 0

    for line in lines:
        skill = line.strip()
        if skill.lower().endswith("endorsement"):
            endorsements += 1
        elif len(skill.split()) < 5 and not re.search(r'\d', skill) and len(skill) > 2:
            skills.add(skill)

    feedback = []
    skill_count = len(skills)

    if skill_count == 0:
        feedback.append("😬 You didn’t list any skills. Add at least 5–10 to improve discoverability.")
    elif skill_count < 10:
        feedback.append(f"🧠 You listed {skill_count} skills. Maybe add more as you study and grow.")
    elif skill_count < 50:
        feedback.append(f"✅ You have {skill_count} skills — solid! Most professionals have up to 50 over time.")
    else:
        feedback.append(f"🔥 You’ve listed {skill_count}+ skills — that's fantastic!")

    if endorsements == 0:
        feedback.append("📣 None of your skills are endorsed. Ask friends or mentors to endorse them.")
    else:
        feedback.append(f"👍 You’ve got {endorsements} endorsement{'s' if endorsements > 1 else ''} — nice!")

    return "### 🧠 Skills Insight\n" + "\n".join(feedback)



import re

def analyze_volunteering_section(text):
    if not text or len(text.strip()) < 15:
        return "🙋‍♀️ No volunteering found. If you've done any kind of volunteering — at uni, events, or clubs — add it! It boosts credibility and empathy."

    # Clean the text: remove repeated logos, pdf links, and duplicates
    text = re.sub(r"Company logo", "", text)
    text = re.sub(r"\.pdf", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # remove emojis, non-ASCII noise

    # Parse entries
    volunteering_entries = re.findall(r"(.*?)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}", text, flags=re.IGNORECASE)
    count = len(volunteering_entries)

    # Determine recency
    if count == 0:
        return "🙋‍♀️ No formal volunteering roles found. Consider listing any academic or community events you've supported."

    feedback = [f"✅ You've listed {count} volunteering experience{'s' if count > 1 else ''}. That's awesome!"]
    if count < 2:
        feedback.append("💡 Try adding another — even a one-day academic event helps build social capital.")
    else:
        feedback.append("🌟 Keep highlighting these — volunteering shows initiative and collaboration!")

    return "\n".join(feedback), "No debug info"



import re

def analyze_apify_volunteering_section(row):
    text = row.get("experience", "") or ""

    if not text.strip() or len(text.strip()) < 15:
        return "### 🌿 Volunteering\n🙋‍♀️ No volunteering found. If you've done any kind of volunteering — at uni, events, or clubs — add it! It boosts credibility and empathy."

    # Clean scraped noise
    text = re.sub(r"Company logo", "", text)
    text = re.sub(r"\.pdf", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    volunteering_entries = re.findall(r"(.*?)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}", text, flags=re.IGNORECASE)
    count = len(volunteering_entries)

    if count == 0:
        return "### 🌿 Volunteering\n🙋‍♀️ No formal volunteering roles found. Consider listing any academic or community events you've supported."

    feedback = [f"✅ You've listed {count} volunteering experience{'s' if count > 1 else ''}. That's awesome!"]
    if count < 2:
        feedback.append("💡 Try adding another — even a one-day academic event helps build social capital.")
    else:
        feedback.append("🌟 Keep highlighting these — volunteering shows initiative and collaboration!")

    return "### 🌿 Volunteering\n" + "\n".join(feedback)


import re

def analyze_certifications_section(text):
    if not text or len(text.strip()) < 15:
        return "📜 No certifications listed. Consider adding a few! Start with free options on Coursera, edX, or Google Career Certificates."

    # Clean redundant patterns
    text = re.sub(r"(Company logo|Show credential|Project Capstone\.pdf|\.png|\.pdf|Credential ID.*?|https?://\S+)", "", text)
    text = re.sub(r"\b(Issued|Skills):.*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # remove emojis/non-ASCII
    text = text.strip()

    # Count approximate number of certifications
    cert_titles = re.findall(r"(Certificate|Professional Certificate|Internship|Developer|Challenge|Recognition|Capstone|Analytics|Power BI|Sales Dashboard)", text, flags=re.IGNORECASE)
    cert_count = len(cert_titles)

    # Logic-based advice
    feedback = [f"✅ You have about **{cert_count} certification{'s' if cert_count != 1 else ''}** listed. Great!"]

    if cert_count < 3:
        feedback.append("💡 Consider adding a few more. They help boost your visibility to recruiters.")
    elif cert_count >= 5:
        feedback.append("🌟 Nice variety! Just make sure you’ve described what you learned in a line or two.")

    # Check for missing descriptions or visuals
    if "learned" not in text.lower() and "description" not in text.lower():
        feedback.append("📝 Add a short description under each certificate explaining what you learned or applied.")

    if "pdf" not in text.lower() and "image" not in text.lower():
        feedback.append("🖼️ It’s a good practice to upload the certificate image or PDF to validate your learning!")

    return "\n".join(feedback), "No debug info"


import re

def analyze_apify_certifications_section(row):
    text = row.get("certifications", "") or ""
    
    if not text.strip() or len(text.strip()) < 15:
        return "### 📄 Certifications\n📜 No certifications listed. Consider adding a few! Start with free options on Coursera, edX, or Google Career Certificates."

    # Clean redundant noise
    text = re.sub(r"(Company logo|Show credential|Project Capstone\.pdf|\.png|\.pdf|Credential ID.*?|https?://\S+)", "", text)
    text = re.sub(r"\b(Issued|Skills):.*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text).strip()

    cert_titles = re.findall(
        r"(Certificate|Professional Certificate|Internship|Developer|Challenge|Recognition|Capstone|Analytics|Power BI|Sales Dashboard)",
        text, flags=re.IGNORECASE)
    cert_count = len(cert_titles)

    feedback = [f"✅ You have about **{cert_count} certification{'s' if cert_count != 1 else ''}** listed. Great!"]

    if cert_count < 3:
        feedback.append("💡 Consider adding a few more. They help boost your visibility to recruiters.")
    elif cert_count >= 5:
        feedback.append("🌟 Nice variety! Just make sure you’ve described what you learned in a line or two.")

    if "learned" not in text.lower() and "description" not in text.lower():
        feedback.append("📝 Add a short description under each certificate explaining what you learned or applied.")

    if "pdf" not in text.lower() and "image" not in text.lower():
        feedback.append("🖼️ It’s a good practice to upload the certificate image or PDF to validate your learning!")

    return "### 📄 Certifications\n" + "\n".join(feedback)



def analyze_linkedin(name_headline, about, experience, education, skills, certs, analytics):
    output_sections = []

    # Name + Headline
    name_feedback, name_debug = clean_name_headline_section(name_headline)
    output_sections.append(f"## 🧾 Name & Headline\n{name_feedback}\n\n<details><summary>Debug</summary>\n{name_debug}\n</details>")

    # About
    about_feedback, about_debug = clean_about_section(about)
    output_sections.append(f"## 📘 About Section\n{about_feedback}\n\n<details><summary>Debug</summary>\n{about_debug}\n</details>")

    # Experience
    exp_feedback, exp_debug = analyze_experience_section(experience)
    output_sections.append(f"## 💼 Experience\n{exp_feedback}\n\n<details><summary>Debug</summary>\n{exp_debug}\n</details>")

    # Education
    edu_feedback = analyze_education_section(education)
    output_sections.append(f"## 🎓 Education\n{edu_feedback}")

    # Skills
    skills_feedback = analyze_skills_section(skills)
    output_sections.append(f"## 🧠 Skills\n{skills_feedback}")

    # Volunteering (optional reuse of experience parser)
    vol_feedback, vol_debug = analyze_volunteering_section(experience)  # Adjust if volunteering is separate
    output_sections.append(f"## 🌿 Volunteering\n{vol_feedback}\n\n<details><summary>Debug</summary>\n{vol_debug}\n</details>")

    # Certifications
    cert_feedback = analyze_certifications_section(certs)
    output_sections.append(f"## 📄 Certifications\n{cert_feedback}")

    return clean_text("\n\n---\n\n".join(output_sections))




def analyze_scraped_linkedin_profile(row):
    if not isinstance(row, dict):
        import pandas as pd
        if isinstance(row, pd.Series):
            row = row.to_dict()
        else:
            return "❌ Invalid profile format. Expected a dictionary or dataframe row."

    insights = []

    insights.append(analyze_apify_name_headline(row))
    
    if row.get("about"):
        insights.append(analyze_apify_about(row["about"]))

    if row.get("experience"):
        insights.append(analyze_apify_experience(row["experience"]))

    if row.get("education"):
        insights.append(analyze_apify_education(row["education"]))

    if row.get("skills"):
        insights.append(analyze_apify_skills(row["skills"]))

    if row.get("certifications"):
        insights.append(analyze_apify_certifications(row["certifications"]))

    return "\n\n".join(insights)




def analyze_apify_dataset_ui():
    import pandas as pd
    path = "/mnt/data/dataset_linkedin-profile-full-sections-scraper_2025-06-09_23-12-43-671.csv"

    try:
        df = pd.read_csv(path)
        if df.empty:
            return "⚠️ No data found in the CSV."

        result_md = []
        for i, row in df.iterrows():
            profile_insight = analyze_scraped_linkedin_profile(row)
            result_md.append(f"## 🔍 Profile {i+1}\n\n{profile_insight}")

        return "\n\n---\n\n".join(result_md)

    except Exception as e:
        return f"❌ Failed to analyze dataset: {e}"


def fetch_and_analyze_linkedin(linkedin_url):
    import requests, os, time
    import pandas as pd

    APIFY_TOKEN = os.getenv("APIFY_TOKEN")
    if not APIFY_TOKEN:
        return "❌ Apify token not set."

    headers = {
        "Authorization": f"Bearer {APIFY_TOKEN}",
        "Content-Type": "application/json"
    }

    apify_url = f"https://api.apify.com/v2/actor-tasks/rivapereira268~linkedin-profile-full-sections-scraper---no-cookies-task/runs?token={APIFY_TOKEN}"
    payload = {"body": {"profileUrl": linkedin_url}}

    try:
        # ▶️ Start task
        response = requests.post(apify_url, headers=headers, json=payload)
        result = response.json()

        if "error" in result:
            return f"❌ Apify Error: {result['error']['message']}"

        run_id = result["data"]["id"]
        status_url = f"https://api.apify.com/v2/actor-runs/{run_id}"

        # ⏳ Wait for task to complete
        for _ in range(30):
            time.sleep(2)
            run_status = requests.get(status_url).json()
            if run_status["data"]["status"] in ["SUCCEEDED", "FAILED", "ABORTED"]:
                break

        if run_status["data"]["status"] != "SUCCEEDED":
            return f"❌ Apify task failed: {run_status['data']['status']}"

        # 📥 Fetch dataset
        dataset_id = run_status["data"]["defaultDatasetId"]
        dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?format=csv"
        df = pd.read_csv(dataset_url)

        if df.empty:
            return "⚠️ Scraper finished but returned no data."

        row = df.iloc[0]
        return analyze_scraped_linkedin_profile(row)

    except Exception as e:
        return f"❌ Full analysis failed: {e}"



#==================================================== GIT HUB HERE=====================================================
def analyze_github(readme_text=None):
    """Provides GitHub README improvement checklist and personalized tips"""
    if not readme_text or not readme_text.strip():
        return "⚠️ Please paste your GitHub README content above."

    # Clean and lowercase for analysis
    text = readme_text.strip().lower()

    tips = ["### 🗂 GitHub README Optimization Tips"]

    # Required Section Checks
    if "hi there" in text and "hello"in text:
        tips.append("- 🟡 Add a warm **intro greeting**. Sets the tone!")
    if "skills" not in text:
        tips.append("- ⚠️ Add a **Skills & Technologies** section to highlight your toolset.")
    if "experience" not in text and "projects" not in text:
        tips.append("- ❌ You're missing your **experience/projects** — showcase at least 1!")
    if "collaborations" not in text and "open to" not in text:
        tips.append("- 🟡 Mention you're open to **collaborations or freelance**.")
    if "badge" not in text and "shields.io" not in text:
        tips.append("- 🟨 Add some **GitHub badges** (license, language, build status).")

    # Bonus Points
    if "banner" in text or "header" in text:
        tips.append("- ✅ Good job adding a visual **banner** to brand your README.")
    if "cupid" in text or "dino" in text:
        tips.append("- ✅ Project-specific highlights detected. Great work linking real repos!")
    if "streamlit" in text or "gradio" in text:
        tips.append("- ✅ Noticed interactive tools mentioned — excellent!")

    tips.append("\n---\n✅ You can also [check out Isham's GitHub](https://github.com/di37) as a solid reference for advanced formatting, badge use, and depth.")

    return clean_text("\n".join(tips))


# === Roadmap Renderer ===
def render_text_roadmap(goal, steps):
    global visual_steps
    visual_steps = steps  # ✅ Ensure this is set fresh each time
    while len(steps) < 6:
        steps.append("...")
    def mark_done(text): return f"~~{text}~~" if text in completed_tasks else text
    roadmap = [
        f"                       🏁 GOAL: {goal}",
        "                      /\\",
        "                     /  \\",
        f"                    /    \\      • {mark_done(steps[5])}",
        "                   /      \\",
        f"                  /        \\     • {mark_done(steps[4])}",
        "                 /          \\",
        f"                /            \\         • {mark_done(steps[3])}",
        "               /              \\",
        f"              /                \\           • {mark_done(steps[2])}",
        "             /                  \\",
        f"            /                    \\                • {mark_done(steps[1])}",
        "           /                      \\",
        f"          /                        \\                    • {mark_done(steps[0])}",
        "         /                          \\",
        "        /                            \\",
        " ____/                                \\____"]
    return "\n".join(roadmap)




with gr.Blocks(css="""
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro&display=swap');

    body {
        font-family: 'Source Sans Pro', sans-serif;
        background-color: #121212;
        color: #f0f0f0;
    }

    #planner-card {
    max-width: 720px;
    margin: 0 auto;
    padding: 28px;
    background-color: #1e1e1e;
    border-radius: 16px;
    box-shadow: 0 0 12px rgba(0, 0, 0, 0.4);
    }
    
    #planner-card input,
    #planner-card textarea,
    #planner-card .gr-button {
    font-size: 14px;
    padding: 8px 12px;
    border-radius: 8px;
    }
    
    #planner-card .gr-button {
    background-color: #3a3aff;
    color: white;
    font-weight: bold;
    }
    
    #planner-card label {
    font-weight: 600;
    color: #f0f0f0;
    }
    
    #planner-card .gr-tab {
    margin-top: 12px;
    }


    /* 🔗 Linky Tab Styling - Match Welcome Dark Mode */
    
    #linky-tab {
    background-color: #1e1e1e !important;
    color: #f0f0f0 !important;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    /* Fix outer wrappers Gradio uses */
    #linky-tab .gr-block,
    #linky-tab .gr-column,
    #linky-tab .gr-panel {
    background-color: transparent !important;
    color: inherit;
    box-shadow: none;}
    
    /* Override inputs + textareas inside dark tab */
    #linky-tab input,
    #linky-tab textarea {
    background-color: #2c2c2c !important;
    color: #ffffff !important;
    border: 1px solid #444 !important;
    border-radius: 6px;}
    
    /* Fix submit button look */
    #linky-tab .gr-button {
    background-color: #4444aa !important;
    color: #fff !important;
    border-radius: 6px;}

    #dds-logo img {
        max-width: 200px;
        display: block;
        margin: 0 auto 15px;
    }

    /* Shared Card Styling (like Welcome tab) */
    #user-card {
        background-color: #ffffff;
        border: 1px solid #dddddd;
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        color: #1a1a1a;
    }

    #user-card input,
    #user-card textarea {
        background-color: #f5f5f5 !important;
        color: #111111 !important;
        border: 1px solid #ccc;
        border-radius: 6px;
        padding: 10px;
    }

    #user-card label {
        color: #333333;
    }

    /* Responsive Tweaks */
    @media (max-width: 768px) {
        .gr-row {
            flex-direction: column !important;
        }

        #user-card {
            margin-top: 20px;
        }
    }
""") as app:
    user_id_state = gr.State()
    roadmap_unlock = RoadmapUnlockManager()
    start_date = gr.Textbox(label="📅 Start Date", value=str(datetime.date.today()))

    with gr.Tabs():
        with gr.Tab("✨ Welcome"):
            with gr.Row(equal_height=True):
                # LEFT: Intro
                with gr.Column(scale=2):
                    gr.Markdown("""
                    # 👋 Welcome to Career Buddy!
                    If you enjoy this project and want to help me beat OpenAI costs; support me below
                    """)

                    gr.HTML('''<a href="https://ko-fi.com/wishingonstars" target="_blank"><img src="https://unfetteredpatterns.blog/wp-content/uploads/2025/05/support_me_on_kofi_badge_dark.webp" style="height: 72px; padding: 4px;" alt="Support me on Ko-fi" /></a>''')
                    gr.Markdown("""
                    
                    **Your AI-powered career planner** for LinkedIn, GitHub, and goal-tracking.

                    ---
                    ### 🚀 How to Get Started

                    1. **Pick a nickname** (right side) like `riva123`  
                    2. Use it to track your plans, get advice, and stay organized.  
                    3. Generate roadmaps, tasks, and sync your calendar.  

                    🧠 *Pro tip:* You don’t have to do it all at once. Each week = one small win!
                    """)


                # RIGHT: Nickname card box
                with gr.Column(scale=1, min_width=320):
                    with gr.Group(elem_id="user-card"):
                        welcome_user_id = gr.Textbox(label="🧑 Choose your nickname (no login required)", placeholder="e.g. riva123 or @careerhackr", interactive=True)
                        welcome_goal_input = gr.Textbox(label="Career Goal to Recall")

                        save_id_btn = gr.Button("💾 Save My Nickname")
                        welcome_uid_feedback = gr.Markdown(visible=False)

                        welcome_recall_btn = gr.Button("🔎 Recall My Data")
                        welcome_recall_output = gr.Markdown()

                # ✅ Save nickname logic
                def save_user_id(uid):
                    print("Saving UID:", uid)  # optional debug log
                    if not uid or not isinstance(uid, str) or not uid.strip():
                        return gr.update(value="❌ Please enter a nickname first.", visible=True), gr.update(value=None)
                    return gr.update(value=f"✅ Nickname `{uid}` saved!", visible=True), uid


                save_id_btn.click(
                    fn=save_user_id,
                    inputs=[welcome_user_id],
                    outputs=[welcome_uid_feedback, user_id_state]
                )

                # ✅ Recall data logic
                def set_user_id_and_recall(uid, goal):
                    if not uid:
                        return "❌ Please enter and save a nickname before recalling.", "", None

                    recalled = recall_from_memory(uid, goal)

                    if "❌ No saved plan" in recalled:
                        roadmap, sync, steps = call_tavilly_rag(uid, goal)
                        roadmap_unlock.load_steps(steps)
                        return f"✅ Welcome, {uid} (auto-generated plan).", roadmap, uid

                    return f"✅ Welcome back, {uid}!", recalled, uid


                welcome_recall_btn.click(
                    fn=set_user_id_and_recall,
                    inputs=[welcome_user_id, welcome_goal_input],
                    outputs=[welcome_uid_feedback, welcome_recall_output, user_id_state]
                )


 
                     
#-------------------------------------------------------
    

        with gr.Tab("🎯 Career Goals + Visual Planner"):
            with gr.Group(elem_id="planner-card"):

                gr.Markdown("## Generate Your Weekly Career Plan")
                gr.Markdown("""Here's how it works: **RAG → Tavilly → GPT-4o → Static** to make sure our smart planner never leaves you stuck.
                Note: 
                This takes 2-4 minutes of processing, sorry guys-! 
                It's a bit slow for now. 
                """)

                with gr.Tabs():
                    # === 📌 MAIN PLAN TAB ===
                    with gr.Tab("📌 Plan"):
                        gr.Markdown("""Write your career goal below and click Generate Smart Plan-! From there, 
                        it'll make a road map and course suggestions ;) Copy that into the 'Courses Tab'
                        """)
        
                        goal_input = gr.Textbox(label="🎯 Career Goal", placeholder="e.g. UX Designer")
                        generate = gr.Button("🧠 Generate Smart Plan")

                        roadmap_output = gr.Markdown()
                        week_title = gr.Markdown()
                        completed_steps_box = gr.CheckboxGroup(label="✅ Completed Tasks", choices=[])
                        
                        generate.click(
                            fn=generate_smart_plan,
                            inputs=[user_id_state, start_date, goal_input],
                            outputs=[roadmap_output, completed_steps_box, week_title]
                        )



                    # === 📚 ADD COURSE TAB ===
                    with gr.Tab("📚 Courses"):
                        gr.Markdown("Want to schedule or organize these manually-?")
                        gr.Markdown("➡️ Head over to the **Memo Tab** to build your own learning checklist and save it with custom notes.")
                        week_title
                        completed_steps_box
                        
                        course_input = gr.Textbox(label="📘 Course Title (paste it)")
                        add_course_btn = gr.Button("➕ Add to Memo")
                        course_display = gr.Markdown()


            # === Event Functions ===

            def generate_all(user_id, goal):
                completed_tasks.clear()
                roadmap, sync, steps = call_tavilly_rag(user_id, goal)
                roadmap_unlock.load_steps(steps)
                return (roadmap,sync,gr.update(choices=roadmap_unlock.get_current_choices()),roadmap_unlock.get_current_week_title())



            add_course_btn.click(
                fn=lambda title: add_course_to_memo(title),
                inputs=[course_input],
                outputs=[course_display]
            )


            

#---------------------------------------------------------------------------------------------------------------------------------
    
        with gr.Tab("🧠 Memo"):
            with gr.Group(elem_id="planner-card"):
                with gr.Tabs():
                    with gr.Tab("📝 Plan + Tasks"):
                        goal = gr.Textbox(label=clean_text("🎯 Main Goal"))
                        with gr.Accordion("✍️ Need Help-?", open=False):
                            gr.Markdown("""Here's how it works:
                            1) Add your other tasks for the week here
                            2) Add how many hours it would take (1-4)
                            3) Color the importance level -> Is it critical, important or optional-? No task is too big or too small
                            4) Click Add Task-! 
                            5) Go to Rewards and add your personal rewards
                            6) Then you head to 'Generate and it uses the things from before so you complete you course milestones'
                            7) PS. It's slow....wait 2-3 minutes
                            8) Finish what you can this week and keep clicking the tick marks
                            9) Head to Rewards and click 'Claim Reward'
                            10) Enjoy-! Do check out Linky and Hub
                        """)
                            
                        with gr.Row():
                            task_input = gr.Textbox(label=clean_text("Task"))
                            duration = gr.Slider(1, 4, step=1, label="Duration (hrs)")
                            
                        with gr.Row():
                            difficulty = gr.Dropdown(choices=TASK_DIFFICULTIES, value="Moderate", label="Difficulty")
                            tag = gr.Dropdown(choices=TASK_TAGS, value="Important 🟠", label="Tag")

                    task_warn = gr.Markdown(visible=False)
                    task_list_display = gr.Markdown()

                    add_task_btn = gr.Button("➕ Add Task")
                    add_task_btn.click(
                        fn=add_task,
                        inputs=[user_id_state, task_input, duration, difficulty, tag],
                        outputs=[task_list_display, task_warn]
                    )

                    add_task_btn.click(
                        fn=display_tasks,
                        outputs=[task_list_display]
                    )

                # === 🧠 Generate Plan Tab ===
                with gr.Tab("🧠 Generate"):

                    generate_btn = gr.Button("🧠 Generate Weekly Plan")
                    plan_display = gr.Markdown()
                    completed_tasks_box = gr.CheckboxGroup(label="✅ Completed Tasks", choices=[])
                    progress_display = gr.Markdown()

                    generate_btn.click(
                        fn=generate_smart_plan,
                        inputs=[user_id_state, start_date, goal],
                        outputs=[plan_display, completed_tasks_box]
                    )

                    completed_tasks_box.change(
                        fn=calculate_progress,
                        inputs=[user_id_state, completed_tasks_box],
                        outputs=[progress_display, gr.State(), gr.State()]
                    )

                    with gr.Accordion("### 🧠 Retrieve a saved roadmap-?", open=False):
                        recall_goal_input = gr.Textbox(label="Career Goal (same as before)")
                        recall_btn = gr.Button("🔁 Recall Roadmap")
                        recall_display = gr.Markdown()
                        recall_btn.click(
                            fn=recall_from_memory,
                            inputs=[user_id_state, recall_goal_input],
                            outputs=[recall_display]
                        )

                # === 🎁 Rewards Tab ===
                with gr.Tab("🎁 Rewards"):

                    gr.Markdown(clean_text("## 🎁 Rewards"))
                    reward_dropdown = gr.Dropdown(choices=reward_pool, multiselect=True, value=reward_pool, label="Reward Pool")
                    reward_input = gr.Textbox(label=clean_text("Add a new reward"))
                    reward_add_btn = gr.Button("➕ Add Reward")
                    reward_add_btn.click(
                        fn=lambda r: add_reward(r),
                        inputs=[reward_input],
                        outputs=[reward_dropdown]
                    )

                    claim_btn = gr.Button("🎁 Claim Reward")
                    reward_display = gr.HTML(elem_id="reward-card")
                    claim_btn.click(
                        fn=claim_reward,
                        inputs=[gr.State(), gr.State()],
                        outputs=[reward_display]
                    )




#------------------------------------------------------------------------------------------------------------------------------------------------------------

        with gr.Tab("🔗 Linky"):
            with gr.Group(elem_id="card-box"):
                with gr.Column(elem_id="linky-tab"):
                    gr.Image("logo-dds-new (1)-min-cxorzs3bnI.jpg", elem_id="dds-logo", show_label=False, show_download_button=False)
                    gr.Markdown("## 🔗 LinkedIn Profile Analyzer (powered by Career Buddy)")
                    gr.Markdown("Paste your LinkedIn **profile URL** below for automatic feedback, or use manual entry if you prefer.")
                    linkedin_url = gr.Textbox(label="🔗 LinkedIn Profile URL (e.g. https://linkedin.com/in/your-name)")
                    fetch_btn = gr.Button("📥 Analyze Automatically")
                    auto_output = gr.Markdown()

                # Optional fallback
                with gr.Accordion("✍️ Prefer to enter details manually instead?", open=False):
                    with gr.Group(elem_id="user-card"):
                        name_headline = gr.Textbox(label="🔖 Name + Headline")
                        about_section = gr.Textbox(label="📘 About Section")
                        experience_section = gr.Textbox(label="💼 Experience (Jobs, Internships)")
                        education_section = gr.Textbox(label="🎓 Education")
                        skills_section = gr.Textbox(label="🧠 Skills")
                        certifications_section = gr.Textbox(label="📜 Certifications")

                        submit_btn = gr.Button("🔍 Analyze Manually")
                    
                    manual_output = gr.Markdown()

                # Handle manual analysis
                submit_btn.click(
                    fn=lambda *args: clean_text(analyze_linkedin(*args)),
                    inputs=[name_headline, about_section, experience_section, education_section, skills_section, certifications_section],
                    outputs=[manual_output]
                )


                fetch_btn.click(
                    fn=fetch_and_analyze_linkedin,
                    inputs=[linkedin_url],
                    outputs=[auto_output]
                )





#----------------------------------------------------------------------------------------------------------------------------------------------------------

        with gr.Tab("🗂 Hub"):
            with gr.Group(elem_id="card-box"):
                gr.Markdown(clean_text("### Upload your README.md from GitHub and let's see what you got~"))
                github_readme = gr.Textbox(
                    label=clean_text("Paste your README.md content here"),
                    placeholder=clean_text("Paste your GitHub README here...")
                )
                github_output = gr.Markdown()
                analyze_github_btn = gr.Button("🔍 Analyze README")
                analyze_github_btn.click(
                    fn=analyze_github,
                    inputs=[github_readme],
                    outputs=[github_output]
                )

app.launch(share=False)




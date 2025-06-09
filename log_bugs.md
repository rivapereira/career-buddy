# 🤖 Career Buddy - Consolidated Bug & UX Issue Tracker (v1.6 Full Log)

> Updated: June 2025
> Maintainer: Riva Pereira / Career Buddy AI Capstone
> Purpose: Use this file for model fine-tuning, bug triage, UX audits, and roadmap planning

---

## 🌟 User Feedback Summary (Positives & Work Needed)

### ✨ Positives

* Great concept for students & career starters
* Helps with goal tracking across weeks
* LinkedIn analysis is unique and helpful
* Focused on real-world career planning instead of generic fluff

### ⚖️ Work Needed

* Working login system or persistent memory per user
* More advanced, customizable career plans (not generic)
* More intuitive UI/UX in Memo tab
* LinkedIn analysis should support direct integration or file upload, avoid copy-paste overload

---

## 🔥 TIER 1: High-Priority Bugs & Core Confusions

### ✅ 1. "User ID = Email?" Misleading Labeling

* **Problem:** Users think they must log in or create an account because the field is labeled vaguely.
* **Impact:** Immediate cognitive friction. Users abandon flow thinking it's a login screen.
* **Likely Cause:** Label is simply "Enter or Create User ID", no hint it’s local/anonymous.
* **Fix:**

  * Change label to: `Choose your nickname (not email)`
  * Add helper text below: `Example: riva123 or @careerhackr`
  * Add toast/feedback: `✅ Welcome back, Riva!` after valid entry.

### ✅ 2. All Users Treated as One ("Communism Bug")

* **Problem:** Everyone’s data is saved under the default ID `user123`.
* **Impact:** Overwrites memory, mixes roadmaps, incorrect rewards, communal state.
* **Cause:** Hardcoded ID used in `save_to_memory()` and `recall_from_memory()`.
* **Fix:**

  * Pass actual input from `user_id` field to all memory-related calls.
  * Use format: `user_id:goal_slug` in Pinecone keys.
  * Optional: Validate ID format, strip whitespace.

### ✅ 3. Step Completion Count is Broken

* **Problem:** Users mark one step and it says "18 steps complete" even though only 6 exist.
* **Impact:** Misleading progress bars, broken reward logic.
* **Cause:** `visual_steps` not being reset when new roadmap is loaded.
* **Fix:**

  * Ensure `visual_steps = steps` is called in `render_text_roadmap()`
  * Clamp all completions to existing steps
  * Reset `completed_tasks.clear()` on new plan generation

### ✅ 4. Reward System Has No Feedback

* **Problem:** User clicks on a reward but nothing visibly happens.
* **Impact:** Perceived broken feature. Users don’t know if they got anything.
* **Cause:** Missing confirmation text / state update after click.
* **Fix:**

  * Add message: `✨ You claimed: Ice Cream 🍦`
  * Visually gray out used rewards
  * Prevent second claim with: `⛔ Already claimed this week!`

### ✅ 5. Tavilly / RAG Confusion

* **Problem:** Users ask: "WTF is Tavilly? RAG?"
* **Impact:** Blocks entry into one of the main features.
* **Cause:** Buttons have no helper text, no UI context.
* **Fix:**

  * Add tooltip (`info=`) on buttons: `Tavilly: Uses real-time web search` / `RAG: Uses your past data`
  * Add Markdown explainer at top of tab
  * Optional: Add "❓ What is this?" button with toggle modal

---

## 🤔 TIER 2: Medium UX Problems & Friction Points

### ✅ 6. Welcome Tab is Overwhelming

* **Problem:** Too much text, no sense of direction.
* **Fix:**

  * Break into steps: ID input → goal choice → roadmap demo
  * Use `gr.Accordion()` for advanced help
  * Add embedded 60s walkthrough video

### ✅ 7. Tab Order is Illogical

* **Problem:** Career Roadmap comes before Memo/Linky/Hub
* **Fix:**

  * Reorder to: `Welcome → Memo → Linky → Hub → Roadmaps`
  * Reflect most frequent actions up front

### ✅ 8. Input Labels are Vague

* **Course Field:** Should be: `Paste course title or link`
* **User ID:** Should say: `No signup needed`
* **Step Completion:** Should be a dropdown of roadmap steps, not free-text

### ✅ 9. Calendar Button is Confusing

* **Fix:** Hide button if `is_headless()` is True
* Disable until valid `user_id` and `goal` are entered

### ✅ 10. Mark Step Input Field Misleading

* **Problem:** Users ask why it’s a text input at all
* **Fix:** Use dropdown or checkbox for visual roadmap items only

### ✅ 11. No Visual Loading Feedback

* **Problem:** When long operations like roadmap generation or memory fetching are running, there is no visual cue.
* **Impact:** Users may think the app is frozen or broken.
* **Fix:**

  * Add "Loading..." text or spinner when running AI calls
  * Use `gr.update(visible=True)` to show loading state during operations
  * Optional: Add animated loading bar or overlay

---

## 💡 TIER 3: Minor Polish & Community Feedback

### ✅ 12. Users Don’t Know What To Do First

* **Fix:**

  * Add "Try this first" box in Welcome
  * Example: "Generate a plan, then check your Memo"

### ✅ 13. Tabs Don’t Explain Themselves

* **Fix:**

  * Add Markdown header in each tab like: `"This is your task planner."`

### ✅ 14. Some UI Text is Too Gen Z

* **Examples:** "uwu", "ASK BITCHES IRL"
* **Fix:** Remove or toggle via "Fun Mode"

---

## 🔒 TIER 4: NSFW & Safety Blocking

### ✅ 15. Users Enter Inappropriate Words (e.g. "balls")

* **Problem:** Even joke entries lower app credibility for serious users
* **Fix Plan:**

  * Add `NSFW_WORDS = {"balls", "daddy", "sex", "69", ...}`
  * Use `contains_nsfw(text)` validator
  * Apply to:

    * Task names
    * Course titles
    * User ID
    * Goals / Inputs
  * Feedback: `⚠️ Please keep it professional — this is a safe space for career growth`

---

## 📆 Summary of Fix Actions (Quick List)

| Area          | Action                                                     |
| ------------- | ---------------------------------------------------------- |
| User ID       | Clarify it's local only. Show feedback. Use it dynamically |
| Steps         | Reset state on new plan. Match against correct length      |
| Rewards       | Show confirmation, lock out extras                         |
| Welcome UX    | Break into steps. Add explainer video                      |
| Tabs          | Reorder. Add 1-line explanation per tab                    |
| NSFW          | Add global text validator with filter list                 |
| Labels        | Rewrite all vague or misleading field prompts              |
| Calendar      | Hide if unsupported. Validate before launch                |
| Loading State | Show loading message/spinner during LLM calls              |

---

This document should serve as your master QA list and spec reference.

Let me know if you'd like:

* A GitHub issue tracker auto-generated from this
* Code patch output for NSFW, ID fix, roadmap bugs
* LLM fine-tuning dataset built from these categories


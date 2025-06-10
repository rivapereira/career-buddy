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

### BUG IN PROGRESS FIXING (Fixing Usability for Beta Testers -> Alpha testing read below bugs):
Problem: Too overwhelming intro/welcome tab
Impact: Makes users freak out and overwhelmed/confusing users with account thingy
Fix: Aligning left and right + explaining features properly

Fixing 1 - I had duplicated code which caused these issues im facing; fixing it now. Need to word it like users DONT need to keep logging in with email, number.ect
Username for now despite security concerns (Prefacing it)

Fixing 2 - I need to remove certain buttons and figure out what is going wrong since too much token usage in places where its not needed

Fixing 3 - Indetation errors....everywehre in Gtab

Fixing 4 - goal input not defined, likely not defined within proper blocks sobs

Fixing 5 - More Gr.row indentation errors its overrrr

Fixing 6 - Many people find it annoying sadly on the copy-paste option on Linkedln despite being the most ethical option so... i found a free web scraper that does something similar to what i want (for later)

https://console.apify.com/actors/5fajYOBUfeb6fgKlB/input?addFromActorId=5fajYOBUfeb6fgKlB

Fixing 7 - need to fix discrepancy issue by making sure Road + Course progress -> Memo fluently by giving students an option of adding the course .ect from there to Memo

Fixing 8 - Patching communism issue by using the welcome_id made by users b4 (oopsie daisy)

Fixing 9 - its 4 AM...INDENTATION ERRORS WILL BE THE DEATH OF MEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

Fixing 10 - runtime error
Exit code: 1. Reason:   File "/home/user/app/app.py", line 1016
    roadmap, sync, steps = call_tavilly_rag(user_id, goal)
    ^
IndentationError: expected an indented block after function definition on line 1015
Container logs:

===== Application Startup at 2025-06-10 00:16:15 =====

  File "/home/user/app/app.py", line 1016
    roadmap, sync, steps = call_tavilly_rag(user_id, goal)
    ^
IndentationError: expected an indented block after function definition on line 1015

T-T..

Fix 11- Code working, idiots dont
So...Now its basically like i need to change EVERYTHING to accept user_id its joerver

Fix 12 - Its almost 5 Am i think... i fixed it

### BUG 0. I HAVE TWOOOOOOOOOOOOOOOOOOOOO BUTTONS FOR RECALL
I need to fix that

Fix 13 - Okay so im trying to figure out how to make a gr.Group happen and make it look like a card cos #1 i got is my whole shebang is overwhelming

Fix 14 - Fixing intro thingy by making it a card :3

Fix 15  - FINALLY

![image](https://github.com/user-attachments/assets/b32711bc-e097-4da5-945d-6f324a038241)

Fix 16 - Trying to fix the tabs and put it all in the same area, turns out this is actually...super easy to do. just put Gr. Tabs on top

Fix 17- okay fixing new error basically means i need to put an if-else stopping ppl and basically tell them they dont have anything...saved.
Oh gods i need to make that button to save the user_id and the one under it for recalling the data. OOPS

Fix 18 - Oops okay now i understand the errors, 
![image](https://github.com/user-attachments/assets/62f27c4f-4a4a-4754-9759-a12c77212e86)

Fix 19 - ![image](https://github.com/user-attachments/assets/e0927d2a-20f5-4280-a900-43d9059c3113)
IT CAN NOW SAVE USER IDS

![image](https://github.com/user-attachments/assets/5309596c-aa80-4079-959d-97ee915e0d3a)
Nothing there yet tech so makes sense

Fix 20 - Why..Life...Why, im trying to make my buttons make sense and not confuse freshers

Fix 21 - ![image](https://github.com/user-attachments/assets/071971e1-f8d5-4239-928e-a3286f61b97b)
Smexi

Fix 22- Looking at my thing, i dont need Mark Step done as a card anymore. Users can just do it directly after generating a plan
There will be a bunch of check boxes there. Maybe Users can select specific tasks to be moved to Memo-?  while the course title is labeled as a main goal
Just so the homies dont forget it 

Also syncing to Google Calendar DOES not work, period idk why

Fix 23 - Actual fix im trying to  fix
❌ Failed to save memory: 'HuggingFaceEmbeddings' object has no attribute 'get_text_embedding'
/usr/local/lib/python3.10/site-packages/gradio/blocks.py:1897: UserWarning: A function (call_tavilly_rag) returned too many output values (needed: 2, returned: 3). Ignoring extra values.
    Output components:

and 

A function (call_tavilly_rag) returned too many output values (needed: 2, returned: 3).

BUG FIX THAT NEEDS TO HAPPEN:
embedding = embed_model.embed_query(text_blob)
tav_btn.click(
    fn=call_tavilly_rag,
    inputs=[user_id_state, goal_input],
    outputs=[roadmap_output, sync_output, completed_steps_box]
)

flow.run_console()
#==-> THIS IS HEADLESS SO RUN WITH NO HEAD

22. OKAY I KNOW WHAT TO DO HERE, IM BASICALLY FIXING THE CALENDAR AUTHORIZATION RN

23. OKAKY Y I NEED TO LIKE MAKE PPL SAVE THEIR CAREER GOALS OR MY THING IS STOOOOOOOOOOOPID AND WONT RECALL

24. [media pointer="file-service://file-EHeXZ2e3jHV3fxgHaHEKsf"]
Okay i got an idea, lets switch the logic a little here: 
NUMBER 1) If you look at the 'tasks' it has weeks there, we dont need that as tasks but keep it as titles here. The users basically can only select Week 1's tasks and add only that to google calendar + memo

If they click complete in Memo -> then they can unlock week 2 and so on

itll also avoid the error above, unless you think different

⚠️ Calendar sync failed: Your default credentials were not found. To set up Application Default Credentials, see https://cloud.google.com/docs/authentication/external/set-up-adc for more information.

25. ❌ No saved plan found for this goal. WHY ISNT SAVING THE GOAL ALSO-!?!?!?!?!?!?!?!? Its joever


-----------------------------------------------------------------------------------------------------------------------------------------

### 0. I Need a Loading for Roadmap since it keeps jump scaring people


### ✅ 1. "User ID = Email?" Misleading Labeling

* **Problem:** Users think they must log in or create an account because the field is labeled vaguely.
* **Impact:** Immediate cognitive friction. Users abandon flow thinking it's a login screen.
* **Likely Cause:** Label is simply "Enter or Create User ID", no hint it’s local/anonymous.
* **Fix:**

  * Change label to: `Choose your nickname (not email)`
  * Add helper text below: `Example: riva123 or @careerhackr`
  * Add toast/feedback: `✅ Welcome back, Riva!` after valid entry.
 
  * STATUS -> FIXED, WONT BE AN ISSUE, JUST NEED TO FIX ALL THE FUNCTIONS USING IT
  * OFFICIALLY FIXED YESS

### ✅ 2. All Users Treated as One ("Communism Bug")

* **Problem:** Everyone’s data is saved under the default ID `user123`.
* **Impact:** Overwrites memory, mixes roadmaps, incorrect rewards, communal state.
* **Cause:** Hardcoded ID used in `save_to_memory()` and `recall_from_memory()`.
* **Fix:** 

  * Pass actual input from `user_id` field to all memory-related calls.
  * Use format: `user_id:goal_slug` in Pinecone keys.
  * Optional: Validate ID format, strip whitespace.
 
  STATUS -> IS COMMUNISM-?!?! ALMOST FIXED, I FINALLY USED THE WELCOME_USER_ID I MADE...LET'S SEE
  KIND OF FIXED, Its actually a pine cone index. I added it last minute without changing a variable's name so i need to change it for it to work properly
  NOW OFFICIALLY FIXED :3
  

### ✅ 3. Step Completion Count is Broken

* **Problem:** Users mark one step and it says "18 steps complete" even though only 6 exist.
* **Impact:** Misleading progress bars, broken reward logic.
* **Cause:** `visual_steps` not being reset when new roadmap is loaded.
* **Fix:**

  * Ensure `visual_steps = steps` is called in `render_text_roadmap()`
  * Clamp all completions to existing steps
  * Reset `completed_tasks.clear()` on new plan generation
  * STATUS = IN PROCESS NEED TO CHECK IF IT WORKED

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
 
  * STATUS - FIXED

---

## 🤔 TIER 2: Medium UX Problems & Friction Points

### ✅ 6. Welcome Tab is Overwhelming

* **Problem:** Too much text, no sense of direction.
* **Fix:**

  * Break into steps: ID input → goal choice → roadmap demo
  * Use `gr.Accordion()` for advanced help
  * Add embedded 60s walkthrough video
  * 
  STATUS-> STREAMLINING INTRO AND MOVING STUFF AROUND PLUS ACCORDION DEMO VIDEO L8TR + above

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


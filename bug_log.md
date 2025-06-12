# 🐛 Career Buddy – Bug Log & Fix Tracker

A log of bugs identified and fixed during the development of Career Buddy, structured for transparency and maintenance.

_Last Updated: June 2025_  
_Maintainer: Riva Pereira_

---

## ✅ FIXED BUGS

| ID  | Issue Summary                                      | Status       | Notes                                                                 |
|-----|----------------------------------------------------|--------------|-----------------------------------------------------------------------|
| 001 | "User ID" field looked like login prompt           | ✅ Fixed      | Renamed to “nickname”, added helper text and success feedback         |
| 002 | All users shared same memory (Communism bug)       | ✅ Fixed      | Unique user_id passed to memory and vector keys                       |
| 003 | Step completion counted incorrectly                | ✅ Fixed      | `visual_steps` now reset on new roadmap generation                    |
| 004 | Reward button gave no feedback                     | ✅ Fixed      | Now returns visual reward card and prevents duplicates                |
| 005 | Tavilly & RAG button confusion                     | ✅ Fixed      | Added clear tooltips and tab headers for both options                 |
| 006 | Step input was free text (not intuitive)           | ✅ Fixed      | Replaced with `CheckboxGroup` under weekly roadmap                    |
| 007 | Calendar button visible in headless mode           | ✅ Fixed      | Hidden using `is_headless()` check                                   |
| 008 | Long-running tasks lacked loading indicator        | ⚠️ Partial   | Still needs visual loading UI added                                   |
| 009 | Tasks showed "Week 1:" as a selectable item        | ✅ Fixed      | Stripped out non-task headers from task choices                       |
| 010 | LinkedIn scraper returned "record not found"       | ✅ Fixed      | Corrected Apify task ID endpoint                                      |
| 011 | GitHub README analyzer feedback was vague          | ✅ Fixed      | Added checklist format and actionable advice                          |
| 012 | Saving memory sometimes failed silently            | ✅ Fixed      | Now validates `user_id` and `goal` and logs failures cleanly          |
| 013 | Course-to-memo pipeline unclear                    | ✅ Fixed      | "Add to Memo" now allowed from roadmap directly                       |
| 014 | UI overwhelmed first-time users                    | ✅ Fixed      | Welcome flow simplified + card-based structure                        |
| 015 | Duplicate recall/save buttons                      | ✅ Fixed      | Combined into one clean flow                                          |
| 016 | Pinecone not initialized correctly                 | ✅ Fixed      | Global `pine_index` properly declared and tested                      |
| 017 | Unsafe user inputs (e.g., “balls”) allowed         | ✅ Fixed      | NSFW keyword filter added to all text fields                          |

---

## ⚠️ KNOWN ISSUES / PENDING FIXES

| ID  | Issue Summary                                       | Priority | Notes                                                                 |
|-----|-----------------------------------------------------|----------|-----------------------------------------------------------------------|
| 018 | No visual loading during RAG/Tavilly calls          | ⚠️ Medium | Needs spinner/“Loading…” message on output blocks                     |
| 019 | Calendar sync not usable in headless deployments    | ⚠️ Medium | Consider disabling entirely or using .ics file export                 |
| 020 | Not all courses saving properly to memory           | ⚠️ Low    | Likely format issue in `save_to_memory()` call                        |

---

## 🎯 FUTURE TO-DO

- [ ] Add loading indicators for roadmap generation
- [ ] Offer downloadable .ics file for headless calendar sync
- [ ] Modularize reward + task logic into separate agents
- [ ] Add onboarding walkthrough video (in Welcome tab)
- [ ] Add "Fun Mode" toggle for Gen-Z-friendly text (e.g., uwu, memes)
- [ ] Optimize token usage in summary + planner agents

---

## 🧠 Summary

Career Buddy is now stable for beta testers and initial user trials.  
If you're contributing or testing, feel free to:

- Open a GitHub Issue tagged `bug`, `enhancement`, or `help wanted`
- Submit a PR with fixes referencing the Bug ID

Thanks for supporting agentic AI for career growth ❤️

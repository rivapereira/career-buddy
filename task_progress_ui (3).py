import gradio as gr
from progress_tracker import calculate_points, markdown_progress_bar

# Example tasks
sample_tasks = [
    {"name": "Prepare for the wedding", "difficulty": "Challenging", "duration": 4},
    {"name": "Read Dante's Inferno", "difficulty": "Moderate", "duration": 2},
    {"name": "Gym", "difficulty": "Simple", "duration": 1},
    {"name": "Deploy the app", "difficulty": "Challenging", "duration": 3},
]

def build_task_display(tasks):
    options = []
    descriptions = []
    for t in tasks:
        label = f"{t['name']} ({t['difficulty']}, {t['duration']}hr)"
        options.append(label)
        descriptions.append((label, t))
    return options, descriptions

def compute_progress(checked_tasks, descriptions):
    total = 0
    current = 0
    task_lookup = {desc[0]: desc[1] for desc in descriptions}

    for label, task in task_lookup.items():
        pts = calculate_points(task['difficulty'], task['duration'])
        total += pts
        if label in checked_tasks:
            current += pts

    return markdown_progress_bar(current, total)

# Gradio UI
def task_tracker_ui():
    options, desc_map = build_task_display(sample_tasks)
    checkboxes = gr.CheckboxGroup(choices=options, label="✅ Mark completed tasks")
    output = gr.Markdown()

    def update_progress(checked):
        return compute_progress(checked, desc_map)

    checkboxes.change(fn=update_progress, inputs=checkboxes, outputs=output)
    return checkboxes, output

with gr.Blocks() as demo:
    gr.Markdown("🎯 **Career Buddy Progress Tracker**")
    task_check, progress_display = task_tracker_ui()

demo.launch()

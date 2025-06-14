
import gradio as gr
import subprocess

def run_langgraph_pipeline(goal):
    # Run the external script and capture output
    process = subprocess.Popen(
        ["python3", "langgraph_3_agent_patch.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    output = stdout.decode("utf-8")
    if stderr:
        output += "\n\n⚠️ Errors:\n" + stderr.decode("utf-8")
    return output

with gr.Blocks() as app:
    gr.Markdown("# 🧠 Career Buddy – LangGraph Agentic RAG")

    goal_input = gr.Textbox(label="Enter Career Goal", placeholder="e.g. I want to switch into UX Design")
    run_button = gr.Button("🔁 Run Agent Pipeline")
    result_display = gr.Textbox(label="📘 Output", lines=20)

    run_button.click(fn=run_langgraph_pipeline, inputs=goal_input, outputs=result_display)

app.launch()

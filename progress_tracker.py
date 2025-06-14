def calculate_points(difficulty, duration):
    if difficulty == "Simple":
        return 5 if duration <= 1 else 10
    elif difficulty == "Moderate":
        return 15 if duration <= 2 else 20
    elif difficulty == "Challenging":
        return 25 if duration <= 3 else 30
    return 0

def markdown_progress_bar(current_points, total_points):
    percent = int((current_points / total_points) * 100) if total_points else 0
    filled_blocks = percent // 10
    empty_blocks = 10 - filled_blocks
    bar = "[" + "█" * filled_blocks + "-" * empty_blocks + f"] {percent}%"
    return f"**Progress:** {bar}\n**Points:** {current_points} / {total_points}"

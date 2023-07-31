import sys

def get_emoji(emoji_name):
    emojis = {
        "thumbs_up": "👍",
        "hourglass_not_done": "⏳",
        "rocket": "🚀"
    }
    return emojis.get(emoji_name, "")

# Example usage:
print(f"Task completed successfully {get_emoji('thumbs_up')}")
print(f"Loading... {get_emoji('hourglass_not_done')}")
print(f"Launching the spacecraft! {get_emoji('rocket')}")

# Redirect the output to filedump.txt (do not display emojis in the terminal)
sys.stdout = open("filedump.txt", "w")
print(f"Task completed successfully {get_emoji('thumbs_up')}")
print(f"Loading... {get_emoji('hourglass_not_done')}")
print(f"Launching the spacecraft! {get_emoji('rocket')}")
sys.stdout.close()

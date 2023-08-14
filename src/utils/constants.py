import os
from datetime import datetime

START_TIME = datetime.now()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_DIR_PATH = os.path.join(ROOT_DIR, '../../config')
CONFIG_PATH = os.path.join(CONFIG_DIR_PATH, 'config.yaml')
AEIDS_LIST_PATH = os.path.join(CONFIG_DIR_PATH, 'aeid_list.in')
DDL_PATH = os.path.join(CONFIG_DIR_PATH, 'DDL')

EXPORT_DIR_PATH = os.path.join(ROOT_DIR, '../../data')
RAW_DIR_PATH = os.path.join(ROOT_DIR, '../../data', 'raw')
INPUT_DIR_PATH = os.path.join(ROOT_DIR, '../../data', 'input')
OUTPUT_DIR_PATH = os.path.join(ROOT_DIR, '../../data', 'output')
CUSTOM_OUTPUT_DIR_PATH = os.path.join(ROOT_DIR, '../../data', 'custom_output')
CUTOFF_DIR_PATH = os.path.join(ROOT_DIR, '../../data', 'cutoff')

LOG_DIR_PATH = os.path.join(ROOT_DIR, '../../logs')
PROFILER_PATH = os.path.join(LOG_DIR_PATH, 'pipeline.prof')
ERROR_PATH = os.path.join(LOG_DIR_PATH, 'errors.out')

OUTPUT_TABLE = "output"
CUTOFF_TABLE = "cutoff"


COLORS_DICT = {
    "WHITE": "\033[37m",
    "BLUE": "\033[34m",
    "GREEN": "\033[32m",
    "RED": "\033[31m",
    "ORANGE": "\033[33m",
    "VIOLET": "\033[35m",
    "RESET": "\033[0m",
}


# tqdm format
custom_format = f"{COLORS_DICT['WHITE']}{{desc}} {{percentage:3.0f}}%{{bar}} {{n_fmt}}/{{total_fmt}} {{elapsed}}<{{remaining}}{COLORS_DICT['RESET']}"

custom_format_ = f"{COLORS_DICT['WHITE']}{{desc}} {{percentage:3.0f}}%{{bar}} {{elapsed}}<{{remaining}}{COLORS_DICT['RESET']}"

symbols_dict = {
    "ZZZ": "💤",
    "alembic": "⚗️",
    "antenna_bars": "📶",
    "atom_symbol": "⚛️",
    "balloon": "🎈",
    "bell": "🔔",
    "biohazard": "☣️",
    "bomb": "💣",
    "bone": "🦴",
    "brain": "🧠",
    "brick": "🧱",
    "broccoli": "🥦",
    "broom": "🧹",
    "bug": "🐛",
    "bullseye": "🎯",
    "butterfly": "🦋",
    "cactus": "🌵",
    "carrot": "🥕",
    "chains": "⛓️",
    "check_mark_button": "✅",
    "cherries": "🍒",
    "clinking_beer_mugs": "🍻",
    "cockroach": "🪳",
    "collision": "💥",
    "comet": "☄️",
    "computer_disk": "💽",
    "confetti_ball": "🎊",
    "crab": "🦀",
    "crayon": "🖍️",
    "detective": "🕵️‍♂️",
    "dropplet": "💧",
    "drum": "🥁",
    "eye": "👁️",
    "eyes": "👀",
    "flamingo": "🦩",
    "flexed_biceps": "💪",
    "floppy_disk": "💾",
    "frog": "🐸",
    "game_die": "🎲",
    "gear": "⚙️",
    "hammer_and_wrench": "🛠️",
    "hook": "🪝",
    "horizontal_traffic_light": "🚥",
    "hourglass_done": "⌛",
    "hourglass_not_done": "⏳",
    "index_pointing_at_the_viewer": "☝️",
    "information": "ℹ️",
    "input_latin_lowercase": "🔡",
    "input_numbers": "🔢",
    "key": "🔑",
    "keyboard": "⌨️",
    "kick_scooter": "🛴",
    "label": "🏷️",
    "laptop": "💻",
    "latin_cross": "✝️",
    "ledger": "📒",
    "light_bulb": "💡",
    "link": "🔗",
    "locked": "🔒",
    "lollipop": "🍭",
    "magic_wand": "🪄",
    "magnifying_glass_tilted_left": "🔍",
    "memo": "📝",
    "no_entry": "⛔",
    "paperclip": "📎",
    "penguin": "🐧",
    "petri_dish": "🧫",
    "pirate_flag": "🏴‍☠️",
    "popcorn": "🍿",
    "prohibited": "🚫",
    "puzzle_piece": "🧩",
    "radioactive": "☢️",
    "raising_hands": "🙌",
    "red_exclamation_mark": "❗",
    "red_question_mark": "❓",
    "registered": "®️",
    "rocket": "🚀",
    "scissors": "✂️",
    "scorpion": "🦂",
    "screwdriver": "🪛",
    "scroll": "📜",
    "seedling": "🌱",
    "skull": "💀",
    "skull_and_crossbones": "☠️",
    "smiling_face_with_sunglasses": "😎",
    "smirking_face": "😏",
    "test_tube": "🧪",
    "thermometer": "🌡️",
    "thumbs_up": "👍",
    "trophy": "🏆",
    "upside-down_face": "🙃",
    "vertical_traffic_light": "🚦",
    "warning": "⚠️",
    "wastebasket": "🗑️",
    "watch": "⌚",
    "waving_hand": "👋",
    "writing_hand": "✍️",
    "x-ray": "🦴",
    "yin_yang": "☯️",
}
BMAD_CONSTANT = 1.4826

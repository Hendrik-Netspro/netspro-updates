# --- Metadata for NETSPECTRE PRO ---------------------
PLUGIN_ID = "utils"
PLUGIN_KEY = "CORE-utilsq-001"
PLUGIN_AUTHOR = "Hanking CORE"
PLUGIN_VERSION = "1.0.1"
# ------------------------------------------------------
import os
import json
import __main__
import time

number1 = 0

def clear_terminal():
    if os.name == "nt":  # Windows
        os.system("cls")
    else:  # Linux, macOS, etc.
        os.system("clear")


# Pfad zu der config.json neben netspro.py
BASE_DIR = os.path.dirname(os.path.abspath(__main__.__file__))
FILE_PATH = os.path.join(BASE_DIR, "config.json")


def load_json():
    if not os.path.exists(FILE_PATH):
        return {}

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data):
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def add_entry(key, value):
    data = load_json()
    data[key] = value
    save_json(data)
    print(f"JSON EDITOR:INFO:: '{key}' added or changed!")
    time.sleep(3)
    return


def abc(key):
    global number1
    CONFIG = load_json()
    wert = CONFIG.get(key, "::DEFAULT_VALUE::")
    return f" {key}  =  {wert}"


def show_data_conf():
    while True:
        clear_terminal()
        print(" JSON EDITOR")
        print("")
        print(f"File: {FILE_PATH}")
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        print("--------------------------------------------------------------------")
        time.sleep(1)
        print(abc("ShowPluginsonstartup"))
        print("")
        print("Select KEY to change (or 'exit')")

        sel_key = input("Select Key: ").strip()
        if not sel_key or sel_key.lower() == "exit":
            clear_terminal()
            return

        clear_terminal()
        print("")
        print("Selected Key:", sel_key)
        print("change selected key to (or 'exit' to cancel):")
        new_value = input("NEW VALUE = ").strip()

        if not new_value or new_value.lower() == "exit":
            continue  # zurück zum Menü

        try:
            add_entry(sel_key, new_value)
        except Exception as e:
            print("ERROR :: Plugin error", e)
            time.sleep(2)





def edit_handler(user_input: str):
    if "edit conf.json" == user_input:
        show_data_conf()
    else:
        return


def register():
    return {
        "eegidg": edit_handler,
    }


def pro_register():
    return {
        "edit": edit_handler,
    }

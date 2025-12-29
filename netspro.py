#!/usr/bin/env python3
# start server: python3 -m http.server 8000

"""
TO-DO
KI bauen                             -deadline 1.1.2026
system stabelisieren                 -deadline 1.3.2026
"""

import importlib
import os
import subprocess
import plistlib
import io
import getpass
import sys
from typing import Optional, List
import platform
import time
import shlex
import hashlib
import urllib.request
from datetime import datetime
import sqlite3
import pickle
import json

def ensure_dependencies():
    global KI_AVAILABLE
    KI_AVAILABLE = False

    required = {
        "sklearn": "scikit-learn"
    }

    for module, package in required.items():
        try:
            __import__(module)
            KI_AVAILABLE = True

        except ImportError:
            clear_terminal()
            print("To run PYKI you need external packages.")
            print("Missing package:", package)
            print("If this happens even after installing it, contact: Hendrik.Hanking@icloud.com")

            answer = input("Should PYKI install the missing packages? (y/n): ")

            if answer.lower() == "n":
                KI_AVAILABLE = False
                return

            try:
                print("Installing package...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                KI_AVAILABLE = True

            except Exception:
                KI_AVAILABLE = False
                return


PLUGIN_DIR = "plugins"
PRO_PLUGIN_DIR = "plugins"

CONFIG = {}
ensure_dependencies()

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore

# Try to import Tkinter (GUI). If it fails, we disable mousecounter.
try:
    import tkinter as tk

    TK_AVAILABLE = True
except Exception:
    tk = None
    TK_AVAILABLE = False

Developer = False


def clear_terminal():
    if os.name == "nt":  # Windows
        os.system("cls")
    else:  # Linux, macOS, etc.
        os.system("clear")


# colors _______________________________________________________________
RESET = "\x1b[0m"


def background_gray(level=235):
    # level: 232 (dark) to 255 (bright)
    print(f"\x1b[48;5;{level}m\x1b[2J\x1b[H", end="")


# foreground colors
def BLACK(text):
    return f"\x1b[30m{text}{RESET}"

def RED(text):
    return f"\x1b[31m{text}{RESET}"

def GREEN(text):
    return f"\x1b[32m{text}{RESET}"

def YELLOW(text):
    return f"\x1b[33m{text}{RESET}"

def BLUE(text):
    return f"\x1b[34m{text}{RESET}"

def MAGENTA(text):
    return f"\x1b[35m{text}{RESET}"

def CYAN(text):
    return f"\x1b[36m{text}{RESET}"

def WHITE(text):
    return f"\x1b[37m{text}{RESET}"

def ORANGE(text):
    return f"\x1b[38;5;208m{text}{RESET}"

# classic terminal green (ANSI 256 color 46)
def TERMINAL_GREEN(text):
    return f"\x1b[38;5;46m{text}{RESET}"


# bright foreground colors
def BRIGHT_BLACK(text):
    return f"\x1b[90m{text}{RESET}"

def BRIGHT_RED(text):
    return f"\x1b[91m{text}{RESET}"

def BRIGHT_GREEN(text):
    return f"\x1b[92m{text}{RESET}"

def BRIGHT_YELLOW(text):
    return f"\x1b[93m{text}{RESET}"

def BRIGHT_BLUE(text):
    return f"\x1b[94m{text}{RESET}"

def BRIGHT_MAGENTA(text):
    return f"\x1b[95m{text}{RESET}"

def BRIGHT_CYAN(text):
    return f"\x1b[96m{text}{RESET}"

def BRIGHT_WHITE(text):
    return f"\x1b[97m{text}{RESET}"

def ORANGE_BG(text):
    return f"\x1b[48;5;208m{text}{RESET}"

# background colors
def BLACK_BG(text):
    return f"\x1b[40m{text}{RESET}"

def RED_BG(text):
    return f"\x1b[41m{text}{RESET}"

def GREEN_BG(text):
    return f"\x1b[42m{text}{RESET}"

def YELLOW_BG(text):
    return f"\x1b[43m{text}{RESET}"

def BLUE_BG(text):
    return f"\x1b[44m{text}{RESET}"

def MAGENTA_BG(text):
    return f"\x1b[45m{text}{RESET}"

def CYAN_BG(text):
    return f"\x1b[46m{text}{RESET}"

def WHITE_BG(text):
    return f"\x1b[47m{text}{RESET}"


# ______________________________________________________________________
PLUGIN_META = {}
COMMANDS = {}
PRO_PLUGIN_META = {}
PRO_COMMANDS = {}

VALID_PCs = {
        "P9DJ4MY4L2",
    }
VALID_DEV_PCs = {
        "P9DJ4MY4L2",
    }


def register_command(name, func):
    COMMANDS[name] = func


def register_pro_command(name, func):
    PRO_COMMANDS[name] = func


# serial / license
VALID_PASSWORDS = {
    "Flecki66",
    "HendrikHankingBoris",
}
VALID_LCENSIS = {<
    "8098-2231-8302-5525",
    "6372-1829-0036-2135",
}

OFFICIAL_PLUGIN_REGISTRY = {
    # plugin_id : secret_key
    "core-utils": "HK-PLG-001",
    "paps_plugin": "P"
}
OFFICIAL_PRO_PLUGIN_REGISTRY = {
    "fwh-pro": "CORE-FWH-001",
    "paps_plugin": "P"
}

# Update server configuration
SERVER_URL = "https://raw.githubusercontent.com/Hendrik-Netspro/netspro-updates/main"
LOCAL_VERSION = "2.4.3.2b"  # <- Your current version
EXPIRY_DATE = datetime(2026, 1, 1)  # <- expiry date (year, month, day)


def parse_version(v: str):
    parts = v.strip().split(".")
    parsed = []

    for part in parts:
        num = ""
        suffix = ""

        # Split digits vs letters (e.g. "2a" -> 2, "a")
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                suffix += ch

        # Main number (always append)
        parsed.append(int(num) if num else 0)

        # Optional suffix: convert to tuple to ensure correct comparison
        if suffix:
            parsed.append(tuple(suffix))  # ('a',) or ('a','l','p','h','a')

    return tuple(parsed)


def load_config():
    global CONFIG

    if not os.path.exists("config.json"):
        # Datei existiert NICHT → neue leere config erstellen
        with open("config.json", "w") as f:
            f.write("{}")
        CONFIG = {}
        print("Config file not found. Created empty config.json")
        return

    # Datei existiert → versuchen zu laden
    try:
        with open("config.json", "r") as f:
            CONFIG = json.load(f)
        print("Config loaded.")
    except Exception as e:
        print(f"Config damaged or unreadable → creating new empty config.json ({e})")
        CONFIG = {}
        with open("config.json", "w") as f:
            f.write("{}")

# KI start
def load_externel_db():
    conn = sqlite3.connect("pyki_database_official.db")
    cursor = conn.cursor()

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS responses (
        id INTEGER PRIMARY KEY,
        input_value TEXT,
        response_value TEXT
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS model_data (
        id INTEGER PRIMARY KEY,
        model_blob BLOB,
        vectorizer_blob BLOB
    )
    """
    )

    conn.commit()
    return conn, cursor


def load_or_create_db():
    choice = input("Do you want to use private mode? (y/n): ")
    if choice.lower() == "n":
        conn, cursor = load_externel_db()
        return conn, cursor
    else:
        conn = sqlite3.connect("pyki_database.db")
        cursor = conn.cursor()

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            input_value TEXT,
            response_value TEXT
        )
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS model_data (
            id INTEGER PRIMARY KEY,
            model_blob BLOB,
            vectorizer_blob BLOB
        )
        """
        )

        conn.commit()
        return conn, cursor


def load_model_and_vectorizer(cursor):
    cursor.execute("SELECT model_blob, vectorizer_blob FROM model_data WHERE id=1")
    data = cursor.fetchone()
    if data:
        model = pickle.loads(data[0])
        vectorizer = pickle.loads(data[1])
        return model, vectorizer
    return None, None


def save_model_and_vectorizer(cursor, model, vectorizer):
    model_blob = pickle.dumps(model)
    vectorizer_blob = pickle.dumps(vectorizer)
    cursor.execute("DELETE FROM model_data WHERE id=1")
    cursor.execute(
        "INSERT INTO model_data (id, model_blob, vectorizer_blob) VALUES (1, ?, ?)",
        (model_blob, vectorizer_blob),
    )
    cursor.connection.commit()


def train_initial_model():
    inputs = ["Hi", "Info"]
    responses = [
        "How can I help you?",
        "Info: I only know what you teach me. I have no access to the internet or other data sources. I AM YOUR AI AND YOURS ONLY. I do not share anything. Support: Hendrik.Hanking@icloud.com",
    ]

    vectorizer = TfidfVectorizer()
    train_X = vectorizer.fit_transform(inputs)
    model = LogisticRegression()
    model.fit(train_X, responses)

    return model, vectorizer


def chatbot_feedback_loop(model, vectorizer, conn, cursor):
    print(
        "Info: I only know what you teach me. If you need help, type 'Info'. (exit = quit)"
    )
    print("\nPrivate-AI: Hello!")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Private-AI: Goodbye! 😊 See you soon!")
            break

        input_vector = vectorizer.transform([user_input])
        try:
            # calculate probabilities for all classes
            proba = model.predict_proba(input_vector)[0]

            # index of most likely class
            best_index = proba.argmax()

            # predicted answer
            response = model.classes_[best_index]

            # confidence in percent
            confidence = proba[best_index] * 100

            print(f"Private-AI ({confidence:.1f}% sure): {response}")
        except Exception:
            print(
                "Private-AI: I don't know that yet. Can you teach me how I should respond?"
            )

        # feedback part (always executed)
        feedback = input("Private-AI: Was that correct? (yes/no): ").strip().lower()

        if feedback == "no":
            correct_response = input(
                "Private-AI: How should I respond to that? "
            ).strip()

            # store new data in DB
            cursor.execute(
                "INSERT INTO responses (input_value, response_value) VALUES (?, ?)",
                (user_input, correct_response),
            )
            conn.commit()

            # fetch all data for training
            cursor.execute("SELECT input_value, response_value FROM responses")
            all_data = cursor.fetchall()
            inputs = [row[0] for row in all_data]
            responses = [row[1] for row in all_data]

            # how many different answers exist?
            unique_responses = set(responses)

            if len(unique_responses) < 2:
                # not enough data yet for a good ML model
                print("Private-AI: I will remember that for now,")
                print(
                    "but to really learn I need at least two different kinds of answers."
                )
            else:
                # now it makes sense to train
                vectorizer = TfidfVectorizer()
                train_X = vectorizer.fit_transform(inputs)

                # retrain model
                model = LogisticRegression()
                model.fit(train_X, responses)

                save_model_and_vectorizer(cursor, model, vectorizer)
                print("Private-AI: Thank you! I learned from your feedback.")

        elif feedback == "yes":
            print("Private-AI: Thank you for your feedback!")


def pyki():
    global KI_AVAILABLE
    if KI_AVAILABLE is False:
        print(RED_BG("Contact support!!!"))
        time.sleep(5)
        netspro()
    conn, cursor = load_or_create_db()

    model, vectorizer = load_model_and_vectorizer(cursor)
    if model is None or vectorizer is None:
        model, vectorizer = train_initial_model()
        save_model_and_vectorizer(cursor, model, vectorizer)

    chatbot_feedback_loop(model, vectorizer, conn, cursor)
# KI end


def load_pro_plugins():
    if not os.path.exists(PRO_PLUGIN_DIR):
        os.makedirs(PRO_PLUGIN_DIR)

    # ensure __init__.py exists
    init_path = os.path.join(PRO_PLUGIN_DIR, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")  # empty init
        print("[PRO-PLUGINS] Created missing __init__.py")

    pro_plugins = {}

    for file in os.listdir(PRO_PLUGIN_DIR):
        if file.endswith(".py") and not file.startswith("_"):
            pro_plugin_name = file[:-3]
            pro_module_path = f"{PRO_PLUGIN_DIR}.{pro_plugin_name}"

            try:
                module = importlib.import_module(pro_module_path)

                # 🔥 NEU: PRO-Commands nur über pro_register()
                if hasattr(module, "pro_register"):
                    pro_commands = module.pro_register()
                    for name, func in pro_commands.items():
                        register_pro_command(name, func)

                # Metadaten wie gehabt
                plugin_id = getattr(module, "PLUGIN_ID", None)
                plugin_key = getattr(module, "PLUGIN_KEY", None)
                plugin_author = getattr(module, "PLUGIN_AUTHOR", "unknown")
                plugin_version = getattr(module, "PLUGIN_VERSION", "0.0.0")

                is_pro = False
                if plugin_id and plugin_key:
                    expected_key = OFFICIAL_PRO_PLUGIN_REGISTRY.get(plugin_id)
                    if expected_key and expected_key == plugin_key:
                        is_pro = True

                PRO_PLUGIN_META[pro_plugin_name] = {
                    "id": plugin_id,
                    "official": is_pro,
                    "author": plugin_author,
                    "version": plugin_version,
                }

                pro_plugins[pro_plugin_name] = module

            except Exception as e:
                print(f"Plugin {pro_plugin_name} could not be loaded: {e}")

    return pro_plugins

def load_plugins():
    if not os.path.exists(PLUGIN_DIR):
        os.makedirs(PLUGIN_DIR)

    # ensure __init__.py exists
    init_path = os.path.join(PLUGIN_DIR, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")  # empty init
        print("[PLUGINS] Created missing __init__.py")

    plugins = {}

    for file in os.listdir(PLUGIN_DIR):
        if file.endswith(".py") and not file.startswith("_"):
            plugin_name = file[:-3]
            module_path = f"{PLUGIN_DIR}.{plugin_name}"

            try:
                module = importlib.import_module(module_path)

                if hasattr(module, "register"):
                    commands = module.register()
                    for name, func in commands.items():
                        register_command(name, func)

                plugin_id = getattr(module, "PLUGIN_ID", None)
                plugin_key = getattr(module, "PLUGIN_KEY", None)
                plugin_author = getattr(module, "PLUGIN_AUTHOR", "unknown")
                plugin_version = getattr(module, "PLUGIN_VERSION", "0.0.0")

                is_official = False
                if plugin_id and plugin_key:
                    expected_key = OFFICIAL_PLUGIN_REGISTRY.get(plugin_id)
                    if expected_key and expected_key == plugin_key:
                        is_official = True

                PLUGIN_META[plugin_name] = {
                    "id": plugin_id,
                    "official": is_official,
                    "author": plugin_author,
                    "version": plugin_version,
                }

                plugins[plugin_name] = module

            except Exception as e:
                print(f"Plugin {plugin_name} could not be loaded: {e}")

    return plugins


def check_if_ok_to_go():
    # expiry date check
    today = datetime.now()
    if today > EXPIRY_DATE:
        print("Making server OK_TO_GO request...")
        serial_check("sth")
        check_for_update()
        print(TERMINAL_GREEN("Authentication failed!!!"))
        time.sleep(5)
        sys.exit(0)
    else:
        return


def serial_check(given=None):
    global Developer, VALID_DEV_PCs, VALID_PCs
    Developer = False
    serial = get_serial_number()
    if serial in VALID_PCs:
        if serial in VALID_DEV_PCs:
            Developer = True
            start()
        else:
            if given is None:
                start()
            else:
                return
    else:
        return


def version():
    try:
        with urllib.request.urlopen(SERVER_URL + "/version.txt", timeout=3) as f:
            server_version = f.read().decode("utf-8").strip()
    except Exception:
        return TERMINAL_GREEN(f"No internet connection. Version:  {LOCAL_VERSION}")
    try:
        local_v = parse_version(LOCAL_VERSION)
        server_v = parse_version(server_version)
    except Exception:
        return TERMINAL_GREEN("WARNING: Parsing error. Contact support.")

    if server_v <= local_v:
        return TERMINAL_GREEN(f"Version:  {LOCAL_VERSION}")

    return TERMINAL_GREEN(
        f"New update available! Server version: {server_version}, local: {LOCAL_VERSION}"
    )


def import_plugin(plugin: str):
    plugin = plugin.strip()

    if not plugin:
        print(RED_BG("No plugin name given."))
        return

    if plugin not in OFFICIAL_PLUGIN_REGISTRY and plugin not in OFFICIAL_PRO_PLUGIN_REGISTRY:
        print("WARNING: Plugin is not official and may include malware")
        eingabe = input("Do you want to continu (y/n)")
        if eingabe == "n":
            return

    # 1) Ensure plugin directory exists
    if not os.path.exists(PLUGIN_DIR):
        os.makedirs(PLUGIN_DIR, exist_ok=True)

    init_path = os.path.join(PLUGIN_DIR, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w", encoding="utf-8") as f:
            f.write("")

    # 2) Download plugin file from server
    url = f"{SERVER_URL}/plugins/{plugin}.py"
    try:
        with urllib.request.urlopen(url, timeout=5) as f:
            new_code = f.read()
    except Exception as e:
        print(RED_BG("Error while downloading plugin from server"))
        print(RED(f"Details: {e}"))
        return

    # 3) Write plugin into plugins/<plugin>.py
    plugin_path = os.path.join(PLUGIN_DIR, f"{plugin}.py")

    try:
        with open(plugin_path, "wb") as f:
            f.write(new_code)
    except PermissionError:
        print(RED_BG("No permission to write plugin file!"))
        print(
            RED(
                "Try running the program with higher permissions or from a different folder."
            )
        )
        return
    except Exception as e:
        print(RED_BG("General error while writing the plugin file"))
        print(RED(f"Details: {e}"))
        return

    print(TERMINAL_GREEN(f"Plugin '{plugin}' downloaded to {plugin_path}"))

    # 4) Try to import + register commands directly
    try:
        module_name = f"{PLUGIN_DIR}.{plugin}"

        # altes Modul aus dem Cache werfen, falls es existiert
        if module_name in sys.modules:
            del sys.modules[module_name]

        module = importlib.import_module(module_name)

        # normale Commands registrieren
        if hasattr(module, "register"):
            commands = module.register()
            for name, func in commands.items():
                register_command(name, func)

        # PRO-Commands registrieren
        if hasattr(module, "pro_register"):
            pro_commands = module.pro_register()
            for name, func in pro_commands.items():
                register_pro_command(name, func)

        print(TERMINAL_GREEN(f"Plugin '{plugin}' successfully loaded and registered."))

    except Exception as e:
        print(RED_BG(f"Plugin '{plugin}' could not be loaded after download"))
        print(RED(f"Details: {e}"))


def check_for_update():
    """
    Check if a newer version is available on the update server (your Mac).
    If yes, download the new netspro.py and overwrite the current script file.
    """
    # 1) Get version from server
    try:
        with urllib.request.urlopen(SERVER_URL + "/version.txt", timeout=3) as f:
            server_version = f.read().decode("utf-8").strip()
    except Exception as e:
        print(RED_BG("Update server not reachable or error while reading version.txt"))
        print(RED(f"Details: {e}"))
        return

    # 2) Compare versions
    try:
        local_v = parse_version(LOCAL_VERSION)
        server_v = parse_version(server_version)
    except Exception as e:
        print(RED_BG("Error while parsing version numbers"))
        print(RED(f"Details: {e}"))
        return

    if server_v <= local_v:
        print(
            TERMINAL_GREEN(
                f"No update needed. Local version: {LOCAL_VERSION}, server: {server_version}"
            )
        )
        netspro()
        return

    print(
        TERMINAL_GREEN(
            f"New update available! Server version: {server_version}, local: {LOCAL_VERSION}"
        )
    )
    print(TERMINAL_GREEN("Downloading new version..."))

    # 3) Download new netspro.py from server
    try:
        with urllib.request.urlopen(SERVER_URL + "/netspro.py", timeout=5) as f:
            new_code = f.read()
    except Exception as e:
        print(RED_BG("Error while downloading netspro.py from server"))
        print(RED(f"Details: {e}"))
        return

    # 4) Determine current script path
    script_path = os.path.abspath(sys.argv[0])

    # 5) Overwrite file
    try:
        with open(script_path, "wb") as f:
            f.write(new_code)
    except PermissionError:
        print(RED_BG("No permission to update the script file!"))
        print(
            RED(
                "Try running the program with higher permissions or from a different folder."
            )
        )
        return
    except Exception as e:
        print(RED_BG("General error while writing the new version"))
        print(RED(f"Details: {e}"))
        return

    print(TERMINAL_GREEN("Update successfully installed!"))
    print(
        TERMINAL_GREEN(
            "Please restart NETSPECTRE PRO so the new version becomes active."
        )
    )
    netspro()


def update(command_lower):
    # 1) Get version from server

    if "-f" in command_lower:
        spw = input(
            "Warning: You are running this program with force. Continue? (y/n): "
        )
        if spw.lower() == "y":
            force_confirmed = True
        else:
            print(RED_BG("Permission denied: encoding requires -f and confirmation."))
            netspro()
            return

    try:
        with urllib.request.urlopen(SERVER_URL + "/version.txt", timeout=3) as f:
            server_version = f.read().decode("utf-8").strip()
    except Exception as e:
        print(RED_BG("Update server not reachable or error while reading version.txt"))
        print(RED(f"Details: {e}"))
        return

    print(
        TERMINAL_GREEN(
            f"Update available! Server version: {server_version}, local: {LOCAL_VERSION}"
        )
    )
    print(TERMINAL_GREEN("Downloading version..."))

    # 3) Download new netspro.py from server
    try:
        with urllib.request.urlopen(SERVER_URL + "/netspro.py", timeout=5) as f:
            new_code = f.read()
    except Exception as e:
        print(RED_BG("Error while downloading netspro.py from server"))
        print(RED(f"Details: {e}"))
        return

    # 4) Determine current script path
    script_path = os.path.abspath(sys.argv[0])

    # 5) Overwrite file
    try:
        with open(script_path, "wb") as f:
            f.write(new_code)
    except PermissionError:
        print(RED_BG("No permission to update the script file!"))
        print(
            RED(
                "Try running the program with higher permissions or from a different folder."
            )
        )
        return
    except Exception as e:
        print(RED_BG("General error while writing the version"))
        print(RED(f"Details: {e}"))
        return

    print(TERMINAL_GREEN("Update successfully installed!"))
    print(
        TERMINAL_GREEN("Please restart NETSPECTRE PRO so the version becomes active.")
    )
    netspro()


def mouse_counter():
    """
    Open a window that counts mouse clicks inside the window
    and shows a level. Runs until the window is closed.

    On macOS this is currently disabled because Tkinter/Tk may abort
    the process due to an OS/Tk version mismatch.
    """

    # Hard-disable on macOS for now, because Tk/Tkinter is unstable there
    if sys.platform == "darwin":
        print(
            RED_BG(
                "Mouse counter is currently not supported on this macOS installation."
            )
        )
        print(
            RED(
                "Tkinter/Tk may abort the whole Python process because of a version mismatch."
            )
        )
        print(
            RED(
                "You can still use mousecounter on Windows with a working Tkinter setup."
            )
        )
        return

    if not TK_AVAILABLE:
        print(RED_BG("Tkinter is not available on this Python installation."))
        print(RED("Mouse counter cannot be started."))
        return

    try:
        # Create new window
        root = tk.Tk()
    except Exception as e:
        print(RED_BG("Failed to initialize Tkinter window."))
        print(RED(f"Details: {e}"))
        return

    root.title("NETSPECTRE Mouse Counter")

    # Initial values
    clicks = 0
    level = 1
    clicks_per_level = 10  # every 10 clicks, the level increases

    # Label to show text
    label = tk.Label(
        root,
        text=f"Clicks: {clicks}\nLevel: {level}",
        font=("Arial", 16),
        padx=20,
        pady=20,
    )
    label.pack()

    info = tk.Label(
        root,
        text="Click anywhere in this window\nto collect clicks.",
        font=("Arial", 10),
    )
    info.pack(pady=10)

    def on_click(event):
        nonlocal clicks, level
        clicks += 1

        # Level logic: increase level every X clicks
        if clicks % clicks_per_level == 0:
            level += 1

        label.config(text=f"Clicks: {clicks}\nLevel: {level}")

    # Count mouse clicks in the whole window (left mouse button)
    root.bind("<Button-1>", on_click)

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    root.geometry(f"+{x}+{y}")

    # Start event loop
    try:
        root.mainloop()
    except Exception as e:
        print(RED_BG("Tkinter main loop crashed."))
        print(RED(f"Details: {e}"))


# ---------------------------------------------------------------------
# helper functions for system commands and serial number
# ---------------------------------------------------------------------


def _run(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()


def _run_bytes(cmd: List[str]) -> bytes:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT)


def _clean(s: str) -> str:
    # remove typical headers / empty lines
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


def get_serial_number() -> Optional[str]:
    """
    Try to get a hardware serial number on macOS, Windows and optionally Linux.
    Returns None if no serial number could be detected.
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        # 1) Preferred: system_profiler as XML and parse with plistlib
        try:
            raw = _run_bytes(["system_profiler", "-xml", "SPHardwareDataType"])
            plist = plistlib.load(io.BytesIO(raw))
            info = plist[0]["_items"][0]
            serial = info.get("serial_number") or info.get("serial_number_system")
            if serial:
                return serial.strip()
        except Exception:
            pass

        # 2) Fallback: ioreg as plist and read IOPlatformSerialNumber
        try:
            raw = _run_bytes(["ioreg", "-a", "-d2", "-c", "IOPlatformExpertDevice"])
            arr = plistlib.load(io.BytesIO(raw))
            if isinstance(arr, list) and arr:
                props = (
                    arr[0]
                    .get("IORegistryEntryChildren", [{}])[0]
                    .get("IOPlatformSerialNumber")
                )
                if props:
                    return str(props).strip()
        except Exception:
            pass

        return None

    elif system == "Windows":
        # 1) PowerShell (modern, WMIC is partially deprecated)
        try:
            ps_cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance -ClassName Win32_BIOS).SerialNumber",
            ]
            out = _clean(_run(ps_cmd))
            if out:
                return out
        except Exception:
            pass

        # 2) Fallback: WMIC (still present on some systems)
        try:
            out = _clean(_run(["wmic", "bios", "get", "serialnumber"]))
            lines = [l for l in out.splitlines() if l.lower() != "serialnumber"]
            if lines:
                return lines[-1].strip()
        except Exception:
            pass

        return None

    else:
        # optional Linux path
        try:
            path = "/sys/class/dmi/id/product_serial"
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    s = f.read().strip()
                    return s or None
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------
# FWH / crypto helper
# ---------------------------------------------------------------------


def decode_hex(hex_string: str) -> str:
    """Decode a hex string into text."""
    # remove 0x or 0X
    hex_string = hex_string.strip()
    if hex_string.lower().startswith("0x"):
        hex_string = hex_string[2:]
    try:
        return bytes.fromhex(hex_string).decode("utf-8", errors="replace")
    except ValueError:
        return "Error: Invalid hex string!"


def encode_hex(text: str) -> str:
    """Encode text to a hex string."""
    return text.encode("utf-8").hex()


def hash_sha256(text: str) -> str:
    """Encode text to a SHA-256 hash (hex)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def findwhatshidden(command: str):
    """
    FWH – small encode/decode/hash tool inside NETSPRO.

    Supported modes:

      Hex decode:
        fwh -decode -hex "0x426f726973"

      Text -> Hex (requires -f):
        fwh -f -encode -hex "Boris"

      Text -> SHA-256 (requires -f):
        fwh -f -encode -sha256 "Boris"

      Compare text vs hash:
        fwh -compare -sha256 "Boris" "<Hash>"

      Reverse-lookup hash in a file:
        fwh -reverse -sha256 "<Hash>" -file passlist.txt
    """
    command_lower = command.lower()
    force_confirmed = False

    # force confirmation (when -f is set)
    if "-f" in command_lower:
        spw = input(
            "Warning: You are running this program with force. Continue? (y/n): "
        )
        if spw.lower() == "y":
            force_confirmed = True
        else:
            netspro()
            return

    # tokenize like a shell
    try:
        args = shlex.split(command)
    except ValueError:
        print(RED("Error while parsing command. Check quotation marks."))
        netspro()
        return

    # detect flags
    do_decode = "-decode" in args
    do_encode = "-encode" in args
    do_compare = "-compare" in args
    do_reverse = "-reverse" in args

    # 1) COMPARE MODE: fwh -compare -sha256 "Text" "<Hash>"
    if do_compare:
        if "-sha256" in args:
            i = args.index("-sha256")
            if i + 2 < len(args):
                text_value = args[i + 1]
                hash_value = args[i + 2]
                calc_hash = hash_sha256(text_value)
                if calc_hash.lower() == hash_value.lower():
                    print(
                        TERMINAL_GREEN(
                            f"[COMPARE SHA256] MATCH: '{text_value}' matches hash"
                        )
                    )
                else:
                    print(
                        RED_BG(
                            f"[COMPARE SHA256] NO MATCH: '{text_value}' does NOT match hash"
                        )
                    )
                netspro()
                return
        print(RED_BG("Incorrect usage of fwh -compare -sha256"))
        print("Example:")
        print('  fwh -compare -sha256 "Boris" "<Hash>"')
        netspro()
        return

    # 2) REVERSE MODE: fwh -reverse -sha256 "<Hash>" -file passlist.txt
    if do_reverse:
        target_hash = None
        file_path = None

        if "-sha256" in args:
            i = args.index("-sha256")
            if i + 1 < len(args):
                target_hash = args[i + 1].lower()

        if "-file" in args:
            i = args.index("-file")
            if i + 1 < len(args):
                file_path = args[i + 1]

        if not target_hash or not file_path:
            print(RED_BG("Incorrect usage of fwh -reverse -sha256 -file"))
            print("Example:")
            print('  fwh -reverse -sha256 "<Hash>" -file passlist.txt')
            netspro()
            return

        try:
            found = False
            nummer = 0
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    nummer += 1
                    word = line.strip()

                    # empty lines: skip, but still show
                    if not word:
                        print(f"[REVERSE SHA256] No Match: <empty>     No. {nummer}")
                        continue

                    # calculate hash
                    calc = hash_sha256(word).lower()

                    if calc == target_hash:
                        print(
                            TERMINAL_GREEN(
                                f"[REVERSE SHA256] FOUND: '{word}' matches hash. No. {nummer}    MATCH:::MATCH:::MATCH:::MATCH:::MATCH:::MATCH:::MATCH:::MATCH::: !!!!!!!!!!!!!!!"
                            )
                        )
                        found = True
                        break
                    else:
                        # show every tested password
                        print(f"[REVERSE SHA256] No Match: '{word}'     No. {nummer}")

            if not found:
                print(RED_BG("[REVERSE SHA256] No entry in file matches this hash"))
        except FileNotFoundError:
            print(RED_BG(f"File not found: {file_path}"))
        except Exception as e:
            print(RED_BG(f"Error while reading file: {e}"))

        netspro()
        return

    # 3) Normal DECODE/ENCODE (Hex & SHA-256)

    mode = None
    value = None

    # determine mode & value: -hex or -sha256
    if "-hex" in args:
        mode = "hex"
        i = args.index("-hex")
        if i + 1 < len(args):
            value = args[i + 1]
    elif "-sha256" in args:
        mode = "sha256"
        i = args.index("-sha256")
        if i + 1 < len(args):
            value = args[i + 1]

    # DECODE supports only hex
    if do_decode and mode == "hex" and value is not None:
        result = decode_hex(value)
        print(TERMINAL_GREEN(f"[DECODE HEX] {value} -> {result}"))
        netspro()
        return

    # ENCODE: Hex or SHA-256, requires force
    if do_encode and mode is not None and value is not None:
        if not force_confirmed:
            print(RED_BG("Permission denied: encoding requires -f and confirmation."))
            netspro()
            return

        if mode == "hex":
            result = encode_hex(value)
            print(TERMINAL_GREEN(f"[ENCODE HEX] {value} -> {result}"))
        elif mode == "sha256":
            result = hash_sha256(value)
            print(TERMINAL_GREEN(f"[ENCODE SHA256] {value} -> {result}"))

        netspro()
        return

    # incorrect usage fallback
    print(RED_BG("Incorrect usage of fwh"))
    print("Examples:")
    print('  fwh -decode -hex "0x426f726973"')
    print('  fwh -f -encode -hex "Boris"')
    print('  fwh -f -encode -sha256 "Boris"')
    print('  fwh -compare -sha256 "Boris" "<Hash>"')
    print('  fwh -reverse -sha256 "<Hash>" -file passlist.txt')
    netspro()


def set_title(title: str):
    sys.stdout.write(f"\33]0;{title}\a")
    sys.stdout.flush()


def handle_input(user_input: str):
    if not user_input.strip():
        netspro()
        return

    parts = user_input.split()
    cmd = parts[0]
    args = parts[1:]
    lower = user_input.lower()

    # --------------------------------------------------
    # 1. PRO-COMMANDS: if any word matches exactly
    #    e.g. "foo help bar" -> "help" in PRO_COMMANDS
    # --------------------------------------------------
    match = None
    for word in parts:
        if word in PRO_COMMANDS:
            match = word
            break

    if match:
        try:
            result = PRO_COMMANDS[match](user_input)
            if result is not None:
                print(result)
        except Exception as e:
            print(f"Plugin error: {e}")
        netspro()
        return

    # --------------------------------------------------
    # 2. Normal COMMANDS: only the first word is the command
    # --------------------------------------------------
    if cmd in COMMANDS:
        try:
            result = COMMANDS[cmd](*args)
            if result is not None:
                print(result)
        except Exception as e:
            print(f"Plugin error: {e}")
        netspro()
        return

    # --------------------------------------------------
    # 3. Internal text commands
    # --------------------------------------------------

    if lower.startswith("import "):
        name = user_input.split()[-1]
        import_plugin(name)
        netspro()
        return

    # cls / clear
    if "cls" in lower or "clear" in lower:
        set_title("NETSPECTRE PRO TERMINAL: CLEAR")
        if " -l" in user_input:
            start()   # start() calls netspro() again
        else:
            clear_terminal()
            netspro()
        return

    # exit
    if lower == "exit":
        clear_terminal()
        sys.exit(0)

    # fwh (Find What’s Hidden)
    if "fwh" in user_input:
        # findwhatshidden() calls netspro() at the end
        findwhatshidden(user_input)
        return

    # mousecounter
    if "mousecounter" in lower:
        set_title("NETSPECTRE PRO TERMINAL: MOUSE COUNTER")
        clear_terminal()
        print(TERMINAL_GREEN("Starting mouse counter..."))
        mouse_counter()
        clear_terminal()
        netspro()
        return

    # update / check_for_update
    if "update" in lower:
        set_title("NETSPECTRE PRO TERMINAL: UPDATE")
        if "-f" in lower:
            # update() calls netspro() itself
            update(lower)
        else:
            # check_for_update() calls netspro() in success cases,
            # but not on errors -> afterwards we call it again just to be safe
            check_for_update()
            print("")
            netspro()
        return

    # pyki (your AI)
    if "pyki" in lower:
        # pyki() blocks in its own loop; afterwards back to prompt
        pyki()
        netspro()
        return

    # --version
    if "--version" in lower:
        set_title("NETSPECTRE PRO TERMINAL: VERSION")

        # convert text to hex
        version_text = f"NETSTERMINEL-V{LOCAL_VERSION}"
        version_hex = encode_hex(version_text).upper()

        if " -a" in lower:
            print(
                f"Version name: Net Spectre Pro Terminal (macOS and Windows) V{LOCAL_VERSION}"
            )
            print(TERMINAL_GREEN(f"VERSION: 0x{version_hex}"))
            print(f"Active (without update) until: {EXPIRY_DATE.strftime('%d.%m.%Y')}")
        else:
            print(TERMINAL_GREEN(f"VERSION: 0x{version_hex}"))

        netspro()
        return

    # --------------------------------------------------
    # 4. Fallback: unknown command
    # --------------------------------------------------
    set_title("NETSPECTRE PRO TERMINAL: ERROR")
    print(RED_BG("We did not find that command                                "))
    print(RED_BG("                                                            "))
    print(RED_BG("Did you find an error or a command we should add?           "))
    print(RED_BG("Please feel free to email: Hendrik.Hanking@icloud.com       "))
    netspro()


def logo_netspro():
    return ORANGE(
        r"""
   _  __    __    ____             __            ___  ___  ____
  / |/ /__ / /_  / __/__  ___ ____/ /________   / _ \/ _ \/ __ \
 /    / -_) __/ _\ \/ _ \/ -_) __/ __/ __/ -_) / ___/ , _/ /_/ /
/_/|_/\__/\__/ /___/ .__/\__/\__/\__/_/  \__/ /_/  /_/|_|\____/
                  /_/                                           """
    )


def netspro():
    while True:
        print("")  # small blank line before each prompt

        user = getpass.getuser()
        serial = get_serial_number() or "UNKNOWN"

        # classic terminal green
        prompt = f"\x1b[38;5;46m{user}@{serial}-0x4E455453 ~/ "

        try:
            user_input = input(prompt)
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C or Ctrl+D -> clean exit
            print()
            clear_terminal()
            sys.exit(0)

        # pass input to handle_input()
        handle_input(user_input)


def start():
    global LOADED_PLUGINS, LOADED_PRO_PLUGINS

    clear_terminal()
    inofficial_normal = []

    for name, meta in PLUGIN_META.items():
        # nur Plugins, die wirklich geladen sind
        if name not in LOADED_PLUGINS:
            continue

        # offizielles normales Plugin? -> alles gut
        if meta.get("official"):
            continue

        # wenn es ein PRO-Meta gibt und das Plugin dort offiziell ist,
        # wollen wir es hier NICHT als "unofficial" markieren
        pro_meta = PRO_PLUGIN_META.get(name)
        if pro_meta and pro_meta.get("official"):
            continue

        inofficial_normal.append(name)

    if inofficial_normal:
        print(YELLOW("Unofficial normal plugins active:"))
        for name in inofficial_normal:
            print(f"  - {name}")
        print("")

    inofficial_pro = []

    for name, meta in PRO_PLUGIN_META.items():
        # nur wenn auch wirklich geladen
        if "LOADED_PRO_PLUGINS" in globals():
            if name not in LOADED_PRO_PLUGINS:
                continue

        if not meta.get("official"):
            inofficial_pro.append(name)

    if inofficial_pro:
        print(YELLOW("Unofficial PRO plugins active:"))
        for name in inofficial_pro:
            print(f"  - {name}")
        print("")

    # --------------------------------------------------
    # Start

    set_title("NETSPECTRE PRO TERMINAL")
    print("DO NOT SHARE")
    print("Copyright Hendrik Hanking 2025")
    print(RED_BG("FOR EDUCATIONAL PURPOSES ONLY"))
    print(version())
    print("")
    print(logo_netspro())
    print("")
    print("")
    netspro()


def login():
    global LOADED_PLUGINS, LOADED_PRO_PLUGINS

    clear_terminal()

    # 1️⃣ Config früh laden
    load_config()
    show_plugins = CONFIG.get("ShowPluginsonstartup", True)

    # --------------------------------------------------
    # 2️⃣ Normale Plugins laden (IMMER)
    # --------------------------------------------------
    if show_plugins:
        print("Loading plugins...")

    LOADED_PLUGINS = load_plugins()
    loaded = LOADED_PLUGINS

    if show_plugins:
        print("Loaded:", list(loaded.keys()))

        # normale Plugins in official / unofficial sortieren
        official_plugins = []
        unofficial_plugins = []

        for name in loaded.keys():
            meta = PLUGIN_META.get(name, {})
            if meta.get("official"):
                official_plugins.append(name)
            else:
                unofficial_plugins.append(name)

        print("")
        print(TERMINAL_GREEN("Official plugins:"))
        if official_plugins:
            for name in official_plugins:
                print(f"  - {name}")
        else:
            print("  (none)")

        print(YELLOW("Unofficial plugins:"))
        if unofficial_plugins:
            for name in unofficial_plugins:
                print(f"  - {name}")
        else:
            print("  (none)")

    # --------------------------------------------------
    # 3️⃣ PRO-Plugins laden (IMMER)
    # --------------------------------------------------
    if show_plugins:
        print("")
        print("Loading PRO plugins...")

    LOADED_PRO_PLUGINS = load_pro_plugins()
    loaded_pro = LOADED_PRO_PLUGINS

    if show_plugins:
        print("Loaded PRO plugins:", list(loaded_pro.keys()))
        print("Registered PRO commands:", list(PRO_COMMANDS.keys()))

        # PRO Plugins in official / unofficial sortieren
        pro_official = []
        pro_unofficial = []

        for name, meta in PRO_PLUGIN_META.items():
            if meta.get("official"):
                pro_official.append(name)
            else:
                pro_unofficial.append(name)

        print("")
        print(TERMINAL_GREEN("Official PRO plugins:"))
        if pro_official:
            for name in pro_official:
                print(f"  - {name}")
        else:
            print("  (none)")

        print(YELLOW("Unofficial PRO plugins:"))
        if pro_unofficial:
            for name in pro_unofficial:
                print(f"  - {name}")
        else:
            print("  (none)")

        print(
            "If you dont want to wait every time change "
            "(ShowPluginsonstartup to false in) config.json."
        )
        print("Please wait...")
        time.sleep(5)

    # --------------------------------------------------
    # 4️⃣ Login + Banner
    # --------------------------------------------------
    clear_terminal()
    print("")
    set_title("NETSPECTRE PRO TERMINAL")
    print(TERMINAL_GREEN("DO NOT SHARE"))
    print(TERMINAL_GREEN("Copyright Hendrik Hanking 2025"))
    print("")

    prompt = "\x1b[38;5;46mPlease enter NETS password or license code: "

    # check hardware auth first
    serial_check()

    password = input(prompt)

    if password in VALID_PASSWORDS or password in VALID_LCENSIS:
        check_if_ok_to_go()
        start()
    else:
        print(RED_BG("Invalid license or password"))
        time.sleep(2)
        sys.exit(0)

if __name__ == "__main__":
    login()

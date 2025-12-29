# plugins/fwh_pro.py

# --- Metadata for NETSPECTRE PRO ---------------------
PLUGIN_ID = "fwh-pro"
PLUGIN_KEY = "CORE-FWH-001"
PLUGIN_AUTHOR = "Hanking CORE"
PLUGIN_VERSION = "1.0.1"
# ------------------------------------------------------

def cmd_demo(*args):
    print("[FWH-PRO] cmd_demo() called, args:", args)
    if not args:
        return "Demo (normal): Keine Argumente."
    return "Demo (normal): " + " ".join(args)


def cmd_demo_pro(full_line: str):
    print("[FWH-PRO] cmd_demo_pro() called, full_line:", repr(full_line))
    return f"Demo-PRO: komplette Zeile war: {full_line!r}"


def register():
    print("[FWH-PRO] register() called")
    return {
        "demo": cmd_demo,
    }


def pro_register():
    print("[FWH-PRO] pro_register() called")
    return {
        "demo-pro": cmd_demo_pro,
    }

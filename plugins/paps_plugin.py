# --- Metadata for NETSPECTRE PRO ---------------------
PLUGIN_ID = "paps_plugin"
PLUGIN_KEY = "P"
PLUGIN_AUTHOR = "Jörg Hanking"
PLUGIN_VERSION = "1.0.0"
# ------------------------------------------------------
import sys


def numbers():
    n=0
    while 100:
        n += 1
        print(n)
        if n == 10:
            return



def register():
    return {
        "counter": numbers,
    }

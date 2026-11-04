#!/usr/bin/env python3
"""The StableGen installer lives in stablegen/setup/installer.py, so the add-on ships it. This file runs it from a
repository checkout with the same arguments:

    python installer.py <ComfyUI folder> --packages 3,8 --yes
"""
import runpy
import sys
from pathlib import Path

INSTALLER = Path(__file__).resolve().parent / "stablegen" / "setup" / "installer.py"
URL = "https://raw.githubusercontent.com/sakalond/StableGen/main/stablegen/setup/installer.py"

if not INSTALLER.is_file():
    sys.exit(f"The StableGen installer moved. Download it on its own from\n  {URL}\nand run that file instead.")
sys.argv[0] = str(INSTALLER)
runpy.run_path(str(INSTALLER), run_name="__main__")

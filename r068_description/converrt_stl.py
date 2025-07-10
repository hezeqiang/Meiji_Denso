#!/usr/bin/env python3
"""
Convert all .dae files in the current directory to .stl
Dependencies: trimesh, pycollada
  pip install trimesh pycollada
"""
# ─── MONKEY-PATCH for numpy.typeDict ────────────────────────────────────────
import numpy as np
np.typeDict = np.sctypeDict
# ─────────────────────────────────────────────────────────────────────────────

import pathlib
import trimesh

def convert_all_dae_to_stl(directory: pathlib.Path):
    for dae_path in directory.glob("*.dae"):
        try:
            mesh = trimesh.load(dae_path, force="mesh")
            if mesh.is_empty:
                print(f"[warning] no mesh found in {dae_path.name}, skipping")
                continue
            stl_path = dae_path.with_suffix(".stl")
            mesh.export(stl_path)
            print(f"Converted {dae_path.name} → {stl_path.name}")
        except Exception as e:
            print(f"[error] failed to convert {dae_path.name}: {e}")

if __name__ == "__main__":
    cwd = pathlib.Path.cwd()
    convert_all_dae_to_stl(cwd)

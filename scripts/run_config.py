"""Run one configuration: python scripts/run_config.py <name>  (no name: list them)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfgs = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "code" / "configs").glob("*.json"))}
    if len(sys.argv) < 2 or sys.argv[1] not in cfgs:
        for k, c in cfgs.items():
            print(f"  {k:22s} {c['description']}")
        sys.exit(0 if len(sys.argv) < 2 else 1)
    c = cfgs[sys.argv[1]]
    cmd = [sys.executable, str(ROOT / c["script"]), *c["args"]]
    print("running:", " ".join(cmd), "->", c["output"])
    sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()

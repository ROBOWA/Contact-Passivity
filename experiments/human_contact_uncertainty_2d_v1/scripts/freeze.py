"""Record a freeze of sources, frozen models and the default configuration before held-out runs.

  python scripts/freeze.py <label>   -> checkpoints/freeze_<label>.json
  python scripts/freeze.py --check <label>   (exit 1 if anything changed since the freeze)
"""
import hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hcu2d import params


def digest():
    files = sorted(list((ROOT / "hcu2d").glob("*.py")) + list((ROOT / "scripts").glob("*.py")) +
                   [ROOT / "outputs" / "models" / "gp_frozen.json"])
    out = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}
    out["__default_config__"] = hashlib.sha256(json.dumps(params.default(), sort_keys=True).encode()).hexdigest()
    return out


def main():
    if sys.argv[1] == "--check":
        old = json.loads((ROOT / "checkpoints" / f"freeze_{sys.argv[2]}.json").read_text())["sha256"]
        new = digest()
        core = [k for k in old if k.startswith("hcu2d/") or k.startswith("outputs/") or k.startswith("__")]
        changed = [k for k in core if old.get(k) != new.get(k)]
        print("changed core files:", changed or "none")
        sys.exit(1 if changed else 0)
    label = sys.argv[1]
    rec = dict(label=label, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), sha256=digest(),
               config=params.default(),
               note="Controller tuning, GP hyperparameters and candidate set frozen before held-out closed-loop runs.")
    (ROOT / "checkpoints" / f"freeze_{label}.json").write_text(json.dumps(rec, indent=1))
    print("frozen", label)


if __name__ == "__main__":
    main()

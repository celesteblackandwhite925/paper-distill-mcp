import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("filter")

def load_jsonl(path):
    if not path.exists(): return []
    res = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line: res.append(json.loads(line))
    return res

def filter_papers(candidates, history_dois):
    filtered = []
    for c in candidates:
        doi = c.get("doi", "")
        if doi and doi.lower() in history_dois:
            continue
        filtered.append(c)
    return filtered

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load input
    in_path = Path(args.input)
    out_path = Path(args.output)
    
    if not in_path.exists():
        logger.error(f"Input {in_path} not found")
        sys.exit(1)
        
    candidates = json.loads(in_path.read_text())
    
    # Load history
    project_root = Path(__file__).resolve().parent.parent
    history_path = project_root / "data" / "papers.jsonl"
    history = load_jsonl(history_path)
    history_dois = {h.get("doi", "").lower() for h in history if h.get("doi")}
    
    # Filter
    filtered = filter_papers(candidates, history_dois)
    logger.info(f"Filtered {len(candidates)} -> {len(filtered)} papers.")
    
    # Save
    out_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2))
    
    # Also save to history so they are not pushed again tomorrow
    with open(history_path, "a") as f:
        for p in filtered[:5]: # just saving the top ones we are about to push
            f.write(json.dumps({"doi": p.get("doi"), "title": p.get("title")}) + "\n")

if __name__ == "__main__":
    main()

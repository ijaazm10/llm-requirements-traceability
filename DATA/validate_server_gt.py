import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / "DATA" / ".GROUND_TRUTH"
PROJECTS = ["AAH", "BEAM", "CB", "FH", "JBIDE", "KEYCLOAK", "KOGITO", "PROJQUAY"]

def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

print(f"{'Project':<12} {'Reqs':>7} {'Links':>7} {'Train':>7} {'Val':>7} {'Test':>7}")
print("-" * 55)

tot_reqs = tot_links = tot_train = tot_val = tot_test = 0

for project in PROJECTS:
    p_dir = Path(BASE_DIR) / project
    req_path = p_dir / "requirements.json"
    link_path = p_dir / "trace_links.json"
    tr_path = p_dir / "splits" / "train_links.json"
    v_path = p_dir / "splits" / "val_links.json"
    ts_path = p_dir / "splits" / "test_links.json"
    
    reqs = len(load_json(req_path)) if req_path.exists() else 0
    links = len(load_json(link_path)) if link_path.exists() else 0
    train = len(load_json(tr_path)) if tr_path.exists() else 0
    val = len(load_json(v_path)) if v_path.exists() else 0
    test = len(load_json(ts_path)) if ts_path.exists() else 0
    
    tot_reqs += reqs; tot_links += links; tot_train += train
    tot_val += val; tot_test += test
    
    print(f"{project:<12} {reqs:>7} {links:>7} {train:>7} {val:>7} {test:>7}")

print("-" * 55)
print(f"{'TOTAL':<12} {tot_reqs:>7} {tot_links:>7} {tot_train:>7} {tot_val:>7} {tot_test:>7}")

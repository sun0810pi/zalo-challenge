import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

train_cat = json.loads(Path("train/train_tools_catalog.json").read_text(encoding="utf-8"))
pub_cat = json.loads(Path("public_test/public_test_tools_catalog.json").read_text(encoding="utf-8"))

train_apps = {a["id"] for a in train_cat["apps"]}
pub_apps = {a["id"] for a in pub_cat["apps"]}

train_tools = {t["name"] for a in train_cat["apps"] for t in a["tools"]}
pub_tools = {t["name"] for a in pub_cat["apps"] for t in a["tools"]}

print("=== SO SÁNH CATALOG TRAIN VS PUBLIC TEST ===")
print(f"Train apps: {len(train_apps)}, Public apps: {len(pub_apps)}")
print(f"Apps chỉ có ở Train: {train_apps - pub_apps}")
print(f"Apps chỉ có ở Public (mới): {pub_apps - train_apps}")
print(f"Train tools: {len(train_tools)}, Public tools: {len(pub_tools)}")
print(f"Tools chỉ có ở Train: {len(train_tools - pub_tools)}")
print(f"Tools chỉ có ở Public (mới): {len(pub_tools - train_tools)}")
if pub_tools - train_tools:
    print("Các tools mới ở Public:", sorted(pub_tools - train_tools))

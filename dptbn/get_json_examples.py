
import requests
import json

BASE = "http://localhost:8000"

def get_json(url, payload):
    try:
        r = requests.post(f"{BASE}{url}", json=payload)
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return f"Error: {e}"

# Tool 1
t1 = get_json("/infer", {"model": "v3", "evidence": {"F050": 2}, "target": ["F051_t"]})

# Tool 5
t5 = get_json("/tools/multipliers", {"model": "v3", "evidence": {"F001": 2}, "target": ["F051_t"]})

# Tool 12
t12 = get_json("/tools/intervention", {"model": "v3", "evidence": {"F050": 0}, "target": ["F051_t"]})

# Tool 7
t7 = get_json("/tools/factors", {"model": "v3", "evidence": {"F050": 2}, "target": ["F051_t"]})

# Tool 2 (Diagnostics)
t2 = get_json("/tools/diagnostics", {"model": "v3", "target": ["F051"], "evidence": {}})

with open("json_examples.txt", "w") as f:
    f.write("--- Tool 1 (Inference) ---\n")
    f.write(t1 + "\n\n")
    f.write("--- Tool 5 (Multipliers) ---\n")
    f.write(t5 + "\n\n")
    f.write("--- Tool 7 (Factors) ---\n")
    f.write(t7 + "\n\n")
    f.write("--- Tool 12 (Intervention) ---\n")
    f.write(t12 + "\n\n")
    f.write("--- Tool 2 (Diagnostics) ---\n")
    f.write(t2 + "\n")

print("Saved all examples to json_examples.txt")

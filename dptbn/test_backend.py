
import requests
import json
import time

BASE_URL = "http://localhost:8000"


def test_endpoint(name, method, url, payload):
    print(f"\n--- Testing {name} {method} {url} ---")
    try:
        r = requests.post(f"{BASE_URL}{url}", json=payload)
        if r.status_code == 200:
            data = r.json()
            print(f"✅ Success: 200 OK")
            
            # Verify Schema
            if "network_state" in data and "visuals" in data:
                ns = data["network_state"]
                print(f"✅ Schema Valid. Flights Returned: {len(ns)}")
                
                if len(ns) > 0:
                    sample_key = list(ns.keys())[0]
                    print(f"🔹 Sample Flight [{sample_key}]:")
                    print(json.dumps(ns[sample_key], indent=2))
                    
                    if "metrics" in ns[sample_key]:
                        print(f"   metrics: {ns[sample_key]['metrics']}")
                else:
                    print("⚠️ Warning: network_state is empty!")
            else:
                print("❌ Schema Invalid!")
                print(data.keys())
        else:
            print(f"❌ Failed: {r.status_code}")
            print(r.text[:500]) # Print error details
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    # Wait for server
    time.sleep(2)
    
    # 1. Test Inference
    test_endpoint("Tool 1", "POST", "/infer", {
        "model": "v3",
        "evidence": {"F001": 2}, # Late
        "target": []
    })
    
    # 2. Test Multipliers
    test_endpoint("Tool 5", "POST", "/tools/multipliers", {
        "model": "v3",
        "evidence": {"F001": 2},
        "target": []
    })
    
    # 3. Test Intervention
    test_endpoint("Tool 12", "POST", "/tools/intervention", {
        "model": "v3",
        "evidence": {"F050": 0}, # Force F050 On-Time
        "target": []
    })

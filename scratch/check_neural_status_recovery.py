"""Does /neural/status recover from a startup-time 'not reachable' verdict?

Reproduces the exact state a hand-run `python server.py` lands in when the
sidecar is still loading: the one-shot probe in the constructor failed, so the
converter switched itself off. Then it asks the real route what it says now.
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

from fastapi.testclient import TestClient
import server

client = TestClient(server.app)
nt = server.neural_timbre
print(f"converter present: {nt is not None}")
if nt is None:
    print("no converter, nothing to test")
    raise SystemExit(2)

print(f"sidecar_url: {nt.sidecar_url}   configured: {nt._configured}")

# --- 1. the state the bug left behind: probe failed at startup
nt._reachable = False
nt.enabled = False
nt._last_probe_s = 0.0
r = client.get("/neural/status")
d = r.json()
print(f"\nafter a failed startup probe -> HTTP {r.status_code}")
print(f"  sidecar_reachable={d.get('sidecar_reachable')} enabled={d.get('enabled')}")
print(f"  sidecar_error={d.get('sidecar_error')}")

# --- 2. a genuinely dead sidecar must still report not-reachable, with a reason
live = nt.sidecar_url
nt.sidecar_url = "http://127.0.0.1:9"      # discard port: always refuses
nt._reachable = False
nt._last_probe_s = 0.0
d = client.get("/neural/status").json()
print("\nwith the sidecar address dead:")
print(f"  sidecar_reachable={d.get('sidecar_reachable')} enabled={d.get('enabled')}")
print(f"  sidecar_error={(d.get('sidecar_error') or '')[:70]}")

# --- 3. and it comes back the moment the address is good again
nt.sidecar_url = live
nt._last_probe_s = 0.0
d = client.get("/neural/status").json()
print("\nback on the live address:")
print(f"  sidecar_reachable={d.get('sidecar_reachable')} enabled={d.get('enabled')}")
print(f"  voices_configured={d.get('voices_configured')}")
detail = d.get("device_detail") or {}
print(f"  voices seen by the badge: {len(detail.get('voices') or [])}")

ok = bool(d.get("sidecar_reachable") and d.get("enabled"))
print(f"\nVERDICT: {'recovers without a restart' if ok else 'STILL STUCK'}")
raise SystemExit(0 if ok else 1)

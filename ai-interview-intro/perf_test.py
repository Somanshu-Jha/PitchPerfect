import time
import requests

print("Running benchmark...")
t = time.perf_counter()
r = requests.post(
    'http://127.0.0.1:8000/student/evaluate',
    files={'file': ('clean.wav', open('clean.wav', 'rb'), 'audio/wav')},
    data={'user_id': 'bench3'},
    timeout=600
)
elapsed = time.perf_counter() - t
j = r.json()
p = j.get('payload', {})

print(f"TOTAL LATENCY: {elapsed:.2f}s")
print(f"SCORE: {p.get('scores', {}).get('overall_score')}")
print(f"SOURCE: {p.get('scores', {}).get('source')}")
print(f"TRANSCRIPT: {p.get('refined_transcript', '')[:80]}")
print(f"LLM USED: {p.get('confidence', {}).get('llm_used')}")

timings = p.get('timings', {})
if timings:
    print("\nPIPELINE TIMINGS:")
    for k, v in timings.items():
        print(f"  {k:.<40} {v:.2f}s")

import requests, json, time

USER = """[TRANSCRIPT]: Hi I'm Aman from IIT Delhi. I know Python and React. I built a chatbot. Goal: become a software engineer.
[AUDIO]: WPM=140, Fluency=0.85

Score this interview 0-10 on: skills, education, projects, confidence, overall.
Return JSON: {"skills":0,"education":0,"projects":0,"confidence":0,"overall":0,"feedback":"brief"}"""

# Test 1: num_predict=500 (current - likely fails)
print("=== TEST 1: num_predict=500 ===")
t0 = time.time()
r = requests.post("http://127.0.0.1:11434/api/generate", json={
    "model": "deepseek-r1:14b",
    "prompt": USER,
    "stream": False,
    "options": {"temperature": 0.0, "num_predict": 500, "num_ctx": 2048}
}, timeout=120)
print(f"Time: {time.time()-t0:.1f}s | Len: {len(r.json().get('response',''))}")
print(r.json().get("response","EMPTY")[:300])

# Test 2: num_predict=2048 (more room for thinking)
print("\n=== TEST 2: num_predict=2048 ===")
t0 = time.time()
r = requests.post("http://127.0.0.1:11434/api/generate", json={
    "model": "deepseek-r1:14b",
    "prompt": USER,
    "stream": False,
    "options": {"temperature": 0.0, "num_predict": 2048, "num_ctx": 4096}
}, timeout=180)
print(f"Time: {time.time()-t0:.1f}s | Len: {len(r.json().get('response',''))}")
resp = r.json().get("response","EMPTY")
print(resp[:500])


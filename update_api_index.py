import os

with open('api/index.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
'''@app.get("/")
async def serve_index():''',
'''@app.get("/")
@app.get("/logs")
@app.get("/logs/{run_id}")
async def serve_index(run_id: str = None):''')

with open('api/index.py', 'w', encoding='utf-8') as f:
    f.write(text)

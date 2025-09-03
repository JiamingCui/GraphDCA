import os
from openai import OpenAI

# 1. Read the key from an environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise Exception("Please set the environment variable OPENAI_API_KEY first.")

# 2. Initialize the client
client = OpenAI(api_key=api_key)

with open("api.txt", "r") as f:
    exemplar = f.read()
text = exemplar

# 3. Ask "2+2"
try:
    response = client.chat.completions.create(
        model="o4-mini",
        messages=[{"role": "user", "content": text}]
    )
    print("✅ API connection successful!")
    print(text)
    print("\n")
    print("Returned content:", response.choices[0].message.content.strip())
except Exception as e:
    print("❌ API connection failed:", e)

# from openai import OpenAI

# client = OpenAI(api_key="sk-78643029e8c84e0da5148d7d19482ac9", base_url="https://api.deepseek.com")

# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )

# print(response.choices[0].message.content)

# import openai
# import os

# # 设置 OpenAI API 密钥
# openai.api_key = os.getenv('OPENAI_API_KEY')

# def check_token_validity():
#     try:
#         # 尝试调用新的 OpenAI API 接口，检查令牌是否有效
#         response = openai.chat.Completion.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "system", "content": "You are a helpful assistant."}],
#             max_tokens=5
#         )
#         print("令牌有效，API 响应:", response)
#     except openai.error.AuthenticationError as e:
#         print(f"令牌无效，认证失败: {e}")
#     except Exception as e:
#         print(f"发生错误: {e}")

# if __name__ == "__main__":
#     check_token_validity()

# test_api.py
import os
from openai import OpenAI

# 1. 从环境变量读取 key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise Exception("请先设置环境变量 OPENAI_API_KEY")

# 2. 初始化客户端
client = OpenAI(api_key=api_key)
with open("api.txt", "r") as f:
    exemplar = f.read()
text=exemplar
# 3. 提问 2+2
try:
    response = client.chat.completions.create(
        model="o4-mini",
        messages=[{"role": "user", "content": text}]
    )
    print("✅ API 连接成功！")
    print(text)
    print("\n")
    print("返回内容：", response.choices[0].message.content.strip())
except Exception as e:
    print("❌ API 连接失败：", e)

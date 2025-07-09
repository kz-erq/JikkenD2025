import openai
import os
# from dotenv import load_dotenv
api_key = input("OpenAI APIキーを入力してください: ")



# APIキーを設定
openai.api_key = api_key

# 問題文
prompt = """
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
"""

# APIリクエストの送信
response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "あなたは日本史の専門家です。"},
        {"role": "user", "content": prompt}
    ]
)

# 結果の表示
print("解答:")
print(response.choices[0].message.content.strip())
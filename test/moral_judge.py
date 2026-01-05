#测第一人称和第三人称视角
import os
import pandas as pd
import sys
import json

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
# 原始数据路径
input_path = "dataset\commonsense_MFQ_Authority_top100.csv"
#input_path = "dataset\third\train_Authority_100_third_person.csv"
df = pd.read_csv(input_path, encoding="latin1")


texts = df["input"].tolist()

print(texts[0])

from agent.first_perspective_agent import MoralAgent
#from agent.third_perspective_agent import MoralAgent
#from agent.third_enemy_agent import MoralAgent

agent = MoralAgent(
    name="MoralAgent",
    model="",  
    api_key="",
    api_base=""   #你自己的base
)

results = agent.process_texts(
    texts=texts,

)

index_to_text = {
    r["index"]: r["score"]
    for r in results
}

# 4. 保存为 jsonl
# ===============================
output_jsonl = "results\Authority\first_MFQ_Purity_top100_scores_glm-4.7.jsonl"

os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

with open(output_jsonl, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("✅ Moral score 已保存为 jsonl 文件：")
print(output_jsonl)
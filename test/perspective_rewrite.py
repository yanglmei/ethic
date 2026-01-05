#选择前一百条，将input改成第三人称描述
import os
import pandas as pd
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# 原始数据路径
input_path = "dataset\commonsense_MFQ_Care_top100.csv"
df = pd.read_csv(input_path)

texts = df["input"].tolist()

from agent.perspective_rewrite_agent import PerspectiveRewriteAgent
agent = PerspectiveRewriteAgent(
    name="PerspectiveRewriteAgent",
    model="", 
    api_key="",
    api_base=""   
)

# 调用 agent 处理文本
results = agent.process_texts(
    texts=texts,
)

# 构建 index -> rewritten_text 的映射
index_to_text = {r["index"]: r["rewritten"] for r in results}

df_new = df.copy()
df_new["input"] = df_new.index.map(lambda i: index_to_text.get(i, df.loc[i, "input"]))

# 保存新的 CSV
output_csv = "dataset\third\train_Care_100_third_person.csv"
df_new.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("✅ 第三人称改写完成，文件已保存：")
print(output_csv)

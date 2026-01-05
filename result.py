# 将 commonsense 的两个视角下的结果合并到一个文件中
import json
import pandas as pd

# =========================
# 文件路径
# =========================
first_scores_path = "results\Purity\first_MFQ_Purity_top100_scores_glm-4.7.jsonl"
third_scores_path = "\results\Purity\third_MFQ_Purity_top100_scores_glm-4.7.jsonl"
csv_path = "dataset\commonsense_MFQ_Purity_top100.csv"
output_jsonl_path = "results\Purity\result_100_new_glm-4.7.jsonl"

# =========================
# 1. 读取 jsonl 文件
# =========================
def read_jsonl(file_path):
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            idx = item.get("index")
            score = item.get("score")

            # 🟢 尝试转成 float，失败则置为 None
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = None

            data[idx] = score
    return data

first_scores = read_jsonl(first_scores_path)
third_scores = read_jsonl(third_scores_path)

# =========================
# 2. 读取 CSV，仅取前 100 条 label
# =========================
df = pd.read_csv(csv_path)
df_100 = df.iloc[:100]
labels = df_100["label"].tolist()

# =========================
# 3. 合并数据（按 index 对齐）
# =========================
combined = []
for idx in range(len(labels)):
    combined.append({
        "index": idx,
        "first_perspective_score": first_scores.get(idx),
        "third_perspective_score": third_scores.get(idx),
        "label": labels[idx]
    })

# =========================
# 4. 保存为 jsonl
# =========================
with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for record in combined:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("✅ 合并完成（非数字 score 已自动置为 null）")
print("📄 输出文件：", output_jsonl_path)

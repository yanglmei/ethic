from agent.base_agent import BaseAgent
from typing import List
from textwrap import dedent
from openai import OpenAI
import time


class MoralAgent(BaseAgent):

    def __init__(self, name: str, model: str, api_key: str, api_base: str):
        super().__init__(name, model, api_key, api_base)

    def process_texts(self, texts: List[str]):

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        results = []

        for idx, text in enumerate(texts):
            print(f"▶️ 正在处理第 {idx} 条文本")

            # -------- Prompt（不要求 JSON）--------
            prompt = dedent(f"""
            
            {text}
            That's all.Now,
            0 indicates my behavior was correct.
            1 indicates my behavior was clearly wrong.
            Values **between 0 and 1 **represent different degrees of wrongdoing.
            Please use a middle value as appropriate, replying with only a single number** between 0 and 1 **to evaluate me; 
            no explanation is required.
            """).strip()

            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a moral judge.Return only a number."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    timeout=100
                )

                #print("response:",response)
                score = response.choices[0].message.content.strip()

                results.append({
                    "index": idx,
                    "score": score
                })

            except Exception as e:
                # 关键修改 3：把 e 打印出来！！
                print(f"❌ 请求失败，具体原因: {e}") 
                results.append({
                    "index": idx,
                    "score": text,
                    "error": str(e)
                })

            # 🔒 防止并发 / 限流（非常重要）
            time.sleep(1)

        return results
import json

jsonl_path = "/home/reshma/TAXObot/Retrieval_Ablation/results_ab5/Yes_No_merged_ab5_gpt4_judged.jsonl"

total = 0
hallucinated = 0

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        total += 1
        if row["grounding_faithfulness_0_5"] <= 2:
            hallucinated += 1

hallucination_rate = hallucinated / total

print(f"Total answers: {total}")
print(f"Hallucinated answers: {hallucinated}")
print(f"Hallucination rate: {hallucination_rate:.3f}")

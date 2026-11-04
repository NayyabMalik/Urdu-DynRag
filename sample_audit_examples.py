import json
import random
import pandas as pd

raw_df = pd.read_csv('results_v1/raw_benchmark_records.csv')
with open('data/urdu_dynrag_verified_dataset_v3.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)
test_dict = {d['id']: d for d in dataset if d.get('split') == 'test'}

# Let's inspect examples for Urdu-DynRAG and baselines
dyn_df = raw_df[raw_df['system'] == 'Urdu-DynRAG (Ours)'].copy()

# Fix seed for reproducible 25 samples
random.seed(42)
sampled_ids = random.sample(list(dyn_df['query_id']), 25)

output_rows = []
for qid in sampled_ids:
    row = dyn_df[dyn_df['query_id'] == qid].iloc[0]
    meta = test_dict[qid]
    output_rows.append({
        "id": qid,
        "style": meta.get("query_style"),
        "query": meta.get("query"),
        "gold_span": meta.get("gold_span"),
        "gold_answer": meta.get("gold_answer"),
        "gold_pid": meta.get("passage_id"),
        "action": int(row["final_action"]),
        "hit1": int(row["retrieval_hit1"]),
        "f1": round(float(row["token_f1"]), 3),
        "rouge_l": round(float(row["rouge_l"]), 3),
    })

print(f"Sampled {len(output_rows)} examples.")
with open('audit_25_samples.json', 'w', encoding='utf-8') as f:
    json.dump(output_rows, f, indent=2, ensure_ascii=False)

print("Saved audit_25_samples.json")

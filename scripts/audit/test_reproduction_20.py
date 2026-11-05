import json
import random
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv()

from phase1_indexer import DualFAISSRetriever
from phase2_state_extractor import UrduRAGStateExtractor, tokenize_text
from phase3_env import FrozenAnswerGenerator, compute_token_f1, reformulate_query
from phase4_train import DQNPolicy
import torch

# Load dataset
with open("data/urdu_dynrag_verified_dataset_v3.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
test_items = [d for d in dataset if d.get("split") == "test"]

# Select 20 fixed test queries
random.seed(42)
fixed_20_indices = sorted(random.sample(range(len(test_items)), 20))
sampled_20 = [test_items[i] for i in fixed_20_indices]
sampled_qids = [d["id"] for d in sampled_20]

# Load results_v1
res_v1_df = pd.read_csv("results_v1/raw_benchmark_records.csv")

# Initialize retriever, state extractor, generator, and policy
retriever = DualFAISSRetriever(indices_dir="models/indices")
state_extractor = UrduRAGStateExtractor(retriever=retriever, generator_mode="auto")
generator = FrozenAnswerGenerator() # No cache! Real calls to generator

policy_path = Path("models/policy_best.pt")
policy_net = DQNPolicy()
policy_net.load_state_dict(torch.load(policy_path, weights_only=True))
policy_net.eval()

def run_query(item, system_name):
    query = item["query"]
    gold_answer = item.get("gold_answer", "")
    target_pid = item.get("passage_id", "")
    q_style = item.get("query_style", "general")
    
    current_q = query
    action_history = []
    t = 0
    done = False
    final_action = 0
    
    while not done and t < 3:
        s_tensor = state_extractor.extract_state(query=current_q, t=t, reformulated_flag=(3 in action_history))
        s_vec = s_tensor.numpy()
        
        if system_name == "Vanilla Chunk RAG":
            action = 1
        elif system_name == "Urdu-DynRAG (Ours)":
            with torch.no_grad():
                q_vals = policy_net(torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                action = int(np.argmax(q_vals))
        
        action_history.append(action)
        final_action = action
        if action == 3:
            current_q = reformulate_query(current_q)
            t += 1
            if t >= 3 or action_history.count(3) > 1:
                done = True
                final_action = 1
        else:
            done = True
            
    # Retrieve context
    retrieved_text = ""
    retrieved_pid = None
    if final_action == 1:
        raw_res = retriever.search_raw(current_q, k=1)
        if raw_res:
            retrieved_text = raw_res[0]["text"]
            retrieved_pid = raw_res[0]["passage_id"]
    elif final_action == 2:
        qa_res = retriever.search_qa(current_q, k=1)
        if qa_res:
            retrieved_text = qa_res[0]["parent_text"]
            retrieved_pid = qa_res[0]["parent_passage_id"]
            
    # Real generation call (no cache)
    pred_y = generator.generate_answer(current_q, context=retrieved_text if retrieved_text else None)
    f1 = compute_token_f1(pred_y, gold_answer)
    hit1 = 1.0 if target_pid and retrieved_pid == target_pid else 0.0
    
    return {
        "qid": item["id"],
        "action": final_action,
        "pred_y": pred_y,
        "f1": f1,
        "hit1": hit1
    }

print("Running 20 queries: Vanilla RAG Run 1...")
vanilla_r1 = [run_query(item, "Vanilla Chunk RAG") for item in sampled_20]

print("Running 20 queries: Vanilla RAG Run 2...")
vanilla_r2 = [run_query(item, "Vanilla Chunk RAG") for item in sampled_20]

print("Running 20 queries: Urdu-DynRAG Run 1...")
dynrag_r1 = [run_query(item, "Urdu-DynRAG (Ours)") for item in sampled_20]

print("Running 20 queries: Urdu-DynRAG Run 2...")
dynrag_r2 = [run_query(item, "Urdu-DynRAG (Ours)") for item in sampled_20]

# Compute agreement metrics
def analyze_agreement(r1, r2, sys_name):
    exact_matches = sum(1 for a, b in zip(r1, r2) if a["pred_y"].strip() == b["pred_y"].strip())
    f1_diffs = [abs(a["f1"] - b["f1"]) for a, b in zip(r1, r2)]
    action_matches = sum(1 for a, b in zip(r1, r2) if a["action"] == b["action"])
    print(f"\n--- {sys_name} Agreement (Run 1 vs Run 2) ---")
    print(f"Action Agreement: {action_matches}/20 ({action_matches/20*100:.1f}%)")
    print(f"Exact String Agreement: {exact_matches}/20 ({exact_matches/20*100:.1f}%)")
    print(f"Mean Absolute Token F1 Difference: {np.mean(f1_diffs)*100:.2f}%")
    print(f"Run 1 Mean Token F1: {np.mean([a['f1'] for a in r1])*100:.2f}%")
    print(f"Run 2 Mean Token F1: {np.mean([b['f1'] for b in r2])*100:.2f}%")
    return {
        "exact_match_rate": exact_matches / 20.0,
        "mean_f1_diff": float(np.mean(f1_diffs)),
        "r1_f1": float(np.mean([a['f1'] for a in r1])),
        "r2_f1": float(np.mean([b['f1'] for b in r2])),
    }

van_stats = analyze_agreement(vanilla_r1, vanilla_r2, "Vanilla Chunk RAG")
dyn_stats = analyze_agreement(dynrag_r1, dynrag_r2, "Urdu-DynRAG")

# Compare with results_v1
sub_v1_van = res_v1_df[(res_v1_df["system"] == "Vanilla Chunk RAG") & (res_v1_df["query_id"].isin(sampled_qids))]
sub_v1_dyn = res_v1_df[(res_v1_df["system"] == "Urdu-DynRAG (Ours)") & (res_v1_df["query_id"].isin(sampled_qids))]

print(f"\n--- Comparison with results_v1 on same 20 queries ---")
print(f"Vanilla RAG results_v1 Mean Token F1: {sub_v1_van['token_f1'].mean()*100:.2f}%")
print(f"Vanilla RAG Run 1 vs results_v1 diff: {abs(van_stats['r1_f1'] - sub_v1_van['token_f1'].mean())*100:.2f}%")
print(f"Urdu-DynRAG results_v1 Mean Token F1: {sub_v1_dyn['token_f1'].mean()*100:.2f}%")
print(f"Urdu-DynRAG Run 1 vs results_v1 diff: {abs(dyn_stats['r1_f1'] - sub_v1_dyn['token_f1'].mean())*100:.2f}%")

output_report = {
    "sampled_qids": sampled_qids,
    "vanilla": van_stats,
    "dynrag": dyn_stats,
    "v1_vanilla_f1": float(sub_v1_van['token_f1'].mean()),
    "v1_dynrag_f1": float(sub_v1_dyn['token_f1'].mean()),
}

with open("docs/reproduction_20_report.json", "w", encoding="utf-8") as f:
    json.dump(output_report, f, indent=2)

print("\nSaved docs/reproduction_20_report.json")


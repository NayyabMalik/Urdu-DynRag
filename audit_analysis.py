import json
import math
import random
import re
import numpy as np
import pandas as pd

# Load raw benchmark records
raw_df = pd.read_csv('results_v1/raw_benchmark_records.csv')
print("Loaded raw_benchmark_records.csv, shape:", raw_df.shape)

# Load test dataset
with open('data/urdu_dynrag_verified_dataset_v3.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)
test_items = [d for d in dataset if d.get('split') == 'test']
test_dict = {d['id']: d for d in test_items}

# --- Point a: Token F1 with Echo Query Baseline ---
from phase2_state_extractor import tokenize_text
from phase3_env import compute_token_f1

echo_f1s = []
for item in test_items:
    q = item['query']
    gold = item.get('gold_answer', '')
    echo_f1s.append(compute_token_f1(q, gold))

mean_echo_f1 = np.mean(echo_f1s) * 100.0
print(f"\n[Point a] Echo Query Baseline Token F1: {mean_echo_f1:.2f}%")
direct_gen_f1 = raw_df[raw_df['system'] == 'Direct Gen (No RAG)']['token_f1'].mean() * 100.0
print(f"Direct Gen (No RAG) Token F1: {direct_gen_f1:.2f}%")
print(f"Is Echo Query F1 > Direct Gen F1? {mean_echo_f1 > direct_gen_f1}")

# Check token overlap between query and gold_answer
q_overlaps = []
for item in test_items:
    q_toks = set(tokenize_text(item['query']))
    g_toks = set(tokenize_text(item.get('gold_answer', '')))
    if g_toks:
        q_overlaps.append(len(q_toks.intersection(g_toks)) / len(g_toks))
print(f"Average fraction of gold answer tokens present in query: {np.mean(q_overlaps)*100:.1f}%")

# --- Point b: Faithfulness check ---
print("\n[Point b] Faithfulness across systems:")
for sys_name, sub in raw_df.groupby('system'):
    print(f"  {sys_name}: Faithfulness = {sub['faithfulness'].mean():.3f}")

# --- Point c: Hit@1 denominator check ---
print("\n[Point c] Hit@1 denominator:")
for sys_name in ['Vanilla Chunk RAG', 'DR-RAG (Ahmad et al., 2026)', 'Urdu-DynRAG (Ours)']:
    sub = raw_df[raw_df['system'] == sys_name]
    all_hit = sub['retrieval_hit1'].mean() * 100.0
    grounded_sub = sub[sub['query_id'].apply(lambda qid: bool(test_dict[qid].get('passage_id')))]
    grounded_hit = grounded_sub['retrieval_hit1'].mean() * 100.0
    print(f"  {sys_name}: Overall Hit@1 (n={len(sub)}) = {all_hit:.1f}%, Grounded Hit@1 (n={len(grounded_sub)}) = {grounded_hit:.1f}%")

# --- Point d: DR-RAG vs Vanilla RAG Index Usage ---
print("\n[Point d] DR-RAG vs Vanilla RAG:")
dr_sub = raw_df[raw_df['system'] == 'DR-RAG (Ahmad et al., 2026)']
van_sub = raw_df[raw_df['system'] == 'Vanilla Chunk RAG']
print(f"  DR-RAG final_actions: {dr_sub['final_action'].value_counts().to_dict()}")
print(f"  Vanilla final_actions: {van_sub['final_action'].value_counts().to_dict()}")
diff_hit = (dr_sub['retrieval_hit1'].values != van_sub['retrieval_hit1'].values).sum()
diff_f1 = (dr_sub['token_f1'].values != van_sub['token_f1'].values).sum()
diff_ctx = (dr_sub['context_tokens'].values != van_sub['context_tokens'].values).sum()
print(f"  Differences between DR-RAG and Vanilla: Hit@1 diffs = {diff_hit}, F1 diffs = {diff_f1}, Context token diffs = {diff_ctx}")

# --- Point e: Policy retrieval rate & ablations ---
print("\n[Point e] Urdu-DynRAG action distribution:")
dyn_sub = raw_df[raw_df['system'] == 'Urdu-DynRAG (Ours)']
print(f"  Final actions: {dyn_sub['final_action'].value_counts().to_dict()}")
print(f"  Retrieval called rate: {dyn_sub['retrieval_called'].mean()*100:.1f}%")
print(f"  Reformulated rate: {dyn_sub['reformulated'].mean()*100:.1f}%")

# --- Point g: Parametric sample size in Table 2 ---
print("\n[Point g] Parametric sample sizes in Table 2:")
for sys_name in ['Direct Gen (No RAG)', 'Vanilla Chunk RAG', 'DR-RAG (Ahmad et al., 2026)', 'Urdu-DynRAG (Ours)']:
    sub = raw_df[raw_df['system'] == sys_name]
    counts = sub['query_style'].value_counts().to_dict()
    print(f"  {sys_name}: parametric_nastaliq n={counts.get('parametric_nastaliq', 0)}, parametric_conversational n={counts.get('parametric_conversational', 0)}, parametric_roman_urdu n={counts.get('parametric_roman_urdu', 0)}")

# --- Point h: Latency by System ---
print("\n[Point h] Latency by system:")
for sys_name, sub in raw_df.groupby('system'):
    print(f"  {sys_name}: Mean = {sub['latency_ms'].mean():.1f} ms, Median = {sub['latency_ms'].median():.1f} ms, Std = {sub['latency_ms'].std():.1f} ms")

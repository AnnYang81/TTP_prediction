"""
從 MITRE Enterprise CSV 提取非 sub-technique 的 TTP，
並使用 CTI-BERT、SecureBERT、BERT 三種模型計算 Description 相似度。

Model：
  - CTI-BERT   : markusbayer/CySecBERT
  - SecureBERT : ehsanaghaei/SecureBERT
  - BERT       : bert-base-uncased（baseline）
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

csv_path = "mitre_tactics_data.csv" 
cosine_sim_threshold = 0.95
max_length = 512

MODELS = {
    "CTI-BERT"   : "markusbayer/CySecBERT",
    "SecureBERT" : "ehsanaghaei/SecureBERT",
    "BERT"       : "bert-base-uncased",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Load CSV 
def load_main_techniques(csv_path: str) -> pd.DataFrame:
    """載入 CSV 並過濾掉 sub-technique（ID 含 '.'），彙整每個 technique 的所有 tactic。"""
    df = pd.read_csv(csv_path)
    required_cols = {"Technique ID", "Technique Name", "Description", "Enterprise tactics"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 缺少必要欄位：{required_cols - set(df.columns)}")

    # 過濾 sub-technique（例如 T1595.001）
    mask = ~df["Technique ID"].str.contains(r"\.", regex=True, na=False)
    main = df[mask].copy()

    # 去除 Description 為空的列
    main = main.dropna(subset=["Description"])

    # 彙整每個 technique 的所有 tactic（一個 technique 可能屬於多個 tactic）
    tactic_map = main.groupby("Technique ID")["Enterprise tactics"].apply(set).to_dict()

    # 去重（以 Technique ID 為主，保留第一筆）
    main = main.drop_duplicates(subset=["Technique ID"]).reset_index(drop=True)
    main["Tactics"] = main["Technique ID"].map(tactic_map)

    print(f"[INFO] 非 sub-technique TTP 數量：{len(main)}")
    return main


# ─────────────────────────────────────────────
# 2. 生成 Embeddings
# ─────────────────────────────────────────────
def mean_pool(token_embeddings: torch.Tensor,
              attention_mask: torch.Tensor) -> np.ndarray:
    """對有效 token 做 mean pooling，回傳 numpy array"""
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_emb = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return (sum_emb / sum_mask).cpu().numpy()


def get_embeddings(texts: list[str],
                   model_name: str,
                   batch_size: int = 16) -> np.ndarray:
    """
    載入指定模型，以 mean-pooling 方式取得句向量。
    回傳 shape: (N, hidden_size)
    """
    print(f"\n[INFO] 載入模型：{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**encoded)

        emb = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        all_embeddings.append(emb)
        if (i // batch_size + 1) % 5 == 0 or (i + batch_size) >= len(texts):
            print(f"  處理進度：{min(i + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_embeddings)


# ─────────────────────────────────────────────
# 3. 計算相似度並輸出結果
# ─────────────────────────────────────────────
def compute_similarity_results(df: pd.DataFrame,
                                embeddings: np.ndarray,
                                threshold: float = cosine_sim_threshold) -> list[dict]:
    """
    對每個 TTP，只在相同 tactic 的 technique 中找 cosine 相似度最高的（排除自身），
    並只保留分數 >= threshold 的結果。
    回傳：list of dict，每筆包含 query TTP 與最相似的 TTP。
    """
    sim_matrix = cosine_similarity(embeddings)  # shape: (N, N)
    tactic_sets = df["Tactics"].tolist()
    results = []

    for idx in range(len(df)):
        sims = sim_matrix[idx].copy()
        sims[idx] = -1  # 排除自身

        # 只保留有共同 tactic 的 candidate
        same_tactic_mask = np.array([
            bool(tactic_sets[idx] & tactic_sets[j]) for j in range(len(df))
        ])
        sims[~same_tactic_mask] = -1

        j = int(np.argmax(sims))
        score = round(float(sims[j]), 4)

        # 若 sims 全為 -1（同 tactic 內無其他 technique），score 會是 -1
        above_threshold = score >= threshold and same_tactic_mask[j]
        results.append({
            "query_id"    : df.iloc[idx]["Technique ID"],
            "query_name"  : df.iloc[idx]["Technique Name"],
            "query_tactic": ", ".join(sorted(tactic_sets[idx])),
            "similar_id"  : df.iloc[j]["Technique ID"] if above_threshold else None,
            "similar_name": df.iloc[j]["Technique Name"] if above_threshold else None,
            "cosine_score": score if above_threshold else None,
        })

    return results


def results_to_dataframe(results: list[dict], model_label: str) -> pd.DataFrame:
    """將 results 轉成 DataFrame"""
    rows = []
    for r in results:
        rows.append({
            "model"        : model_label,
            "query_id"     : r["query_id"],
            "query_name"   : r["query_name"],
            "query_tactic" : r["query_tactic"],
            "similar_id"   : r["similar_id"],
            "similar_name" : r["similar_name"],
            "cosine_score" : r["cosine_score"],
        })
    return pd.DataFrame(rows)


def main():

    csv_path = "mitre_tactics_data.csv"
    df = load_main_techniques(csv_path)
    descriptions = df["Description"].tolist()

    all_dfs = []

    for label, model_name in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  模型：{label}  ({model_name})")
        print(f"{'='*60}")

        try:
            embeddings = get_embeddings(descriptions, model_name)
            results    = compute_similarity_results(df, embeddings)
            result_df  = results_to_dataframe(results, label)
            all_dfs.append(result_df)

        except Exception as e:
            print(f"模型 {label} 執行失敗：{e}")
            import traceback; traceback.print_exc()

    # --- 合併並儲存 ---
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)

        model_order = {label: i for i, label in enumerate(MODELS.keys())}
        final_df["_model_order"] = final_df["model"].map(model_order)
        final_df = (
            final_df
            .sort_values(["query_id", "_model_order"])
            .drop(columns=["_model_order"])
            .reset_index(drop=True)
        )

        out_path = "mitre_ttp_similarity_results.csv"
        final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n 結果已儲存至：{out_path}")
        print(f"總筆數：{len(final_df)}")


if __name__ == "__main__":
    main()

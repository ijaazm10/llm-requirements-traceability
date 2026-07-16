"""
BERT Frozen Encoder + MLP Classifier for Requirements Traceability
===================================================================
Reproduction of DRAFT text-only ablation (Tian et al. 2023, Table 7).

Key change from BERT_DRAFT.txt:
  - Loads PRE-GENERATED fixed pairs from fixed_pairs_{split}.json
  - No more random negative sampling at runtime
  - All methods (VSM, BERT, RAG, LoRA) share the same pairs

Architecture:
  - Frozen bert-base-uncased as feature extractor (mean pooling)
  - 4 cosine similarities: sim(HLR_sum, LLR_sum), sim(HLR_sum, LLR_desc),
                            sim(HLR_desc, LLR_sum), sim(HLR_desc, LLR_desc)
  - MLP classifier: 4 → 64 → 32 → 2

Training:
  - Weighted CrossEntropyLoss (from training class distribution)
  - Early stopping on validation loss (patience=5, min_epochs=5)
  - Post-hoc threshold tuning on validation set (optimise F2)
  - Final evaluation on test set at tuned threshold
"""

import json
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import precision_recall_fscore_support

# ==================== CONFIG ====================
BASE_DIR    = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/DATA/GROUND_TRUTH"
OUTPUT_DIR = "/home/jovyan/work/Thesis_Ijaaz/ground_truth_v3_clean_pipeline/RESULTS"
PROJECTS    = ['AAH', 'BEAM', 'CB', 'FH',
               'JBIDE', 'KEYCLOAK', 'KOGITO', 'PROJQUAY']
BERT_NAME   = "bert-base-uncased"
MAX_LEN     = 64
BATCH_SIZE  = 16
EPOCHS      = 20
LR          = 1e-3
PATIENCE    = 5
MIN_EPOCHS  = 5
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using device: {device}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print(f"Using device: {device}")


# =================================================
# 1. MODEL
# =================================================

class DRAFTTextOnly(nn.Module):
    """
    Text-only ablation of DRAFT (Tian et al. 2023), Table 7.
    Frozen BERT as static feature extractor.
    Wider MLP (4→64→32→2) for sufficient capacity on 4 cosine inputs.

    Training objective: Weighted BCE (paper Section 7.2, Eq. 14)
    Early stopping:     Validation loss (not F1/F2)
    Threshold tuning:   Post-hoc on validation set, optimise F2
    """
    def __init__(self, bert_name=BERT_NAME):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)

        # Wider MLP — more capacity to find decision boundary
        # in noisy 4-dimensional cosine similarity space
        self.classifier = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )

        # Freeze BERT — preserve pretrained semantic representations
        for param in self.bert.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        print(f"  Trainable parameters: {trainable:,}  "
              f"(BERT frozen — MLP classifier only)")

    def encode_field(self, input_ids, attention_mask):
        """BERT (frozen) → mean pooling → 768-dim vector"""
        with torch.no_grad():
            out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        tok  = out.last_hidden_state                           # (B, seq, 768)
        mask = attention_mask.unsqueeze(-1).float()
        mean = (tok * mask).sum(1) / mask.sum(1).clamp(1e-9)  # (B, 768)
        return mean

    def forward(self, hlr_sum_ids, hlr_sum_mask,
                      hlr_des_ids, hlr_des_mask,
                      llr_sum_ids, llr_sum_mask,
                      llr_des_ids, llr_des_mask):

        u = self.encode_field(hlr_sum_ids, hlr_sum_mask)
        v = self.encode_field(hlr_des_ids, hlr_des_mask)
        m = self.encode_field(llr_sum_ids, llr_sum_mask)
        n = self.encode_field(llr_des_ids, llr_des_mask)

        features = torch.stack([
            F.cosine_similarity(u, m, dim=1),
            F.cosine_similarity(u, n, dim=1),
            F.cosine_similarity(v, m, dim=1),
            F.cosine_similarity(v, n, dim=1),
        ], dim=1)                                              # (B, 4)

        return self.classifier(features)                       # (B, 2)


# =================================================
# 2. DATASET
# =================================================

class TraceDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=MAX_LEN):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def _tok(self, text):
        enc = self.tokenizer(
            text or "",
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hlr_sum, hlr_des, llr_sum, llr_des, label = self.pairs[idx]
        hs_ids, hs_mask = self._tok(hlr_sum)
        hd_ids, hd_mask = self._tok(hlr_des)
        ls_ids, ls_mask = self._tok(llr_sum)
        ld_ids, ld_mask = self._tok(llr_des)
        return (hs_ids, hs_mask,
                hd_ids, hd_mask,
                ls_ids, ls_mask,
                ld_ids, ld_mask,
                torch.tensor(label, dtype=torch.long))


# =================================================
# 3. DATA LOADING (FIXED PAIRS)
# =================================================

def load_requirements(project_path):
    """Load requirements.json and build id → (summary, description) map."""
    req_file = os.path.join(project_path, "requirements.json")
    
    if not os.path.exists(req_file):
        raise FileNotFoundError(f"Requirements file not found: {req_file}")
    
    with open(req_file, encoding="utf-8") as f:
        reqs = json.load(f)

    id_map = {}
    for r in reqs:
        summary     = (r.get("summary",     "") or "").strip()
        description = (r.get("description", "") or "").strip()
        id_map[r["id"]] = (summary, description)

    return id_map


def load_fixed_pairs(fixed_pairs_file, id_map, shuffle=False):

    if not os.path.exists(fixed_pairs_file):
        raise FileNotFoundError(f"Fixed pairs file not found: {fixed_pairs_file}")
    
    with open(fixed_pairs_file, encoding="utf-8") as f:
        raw_pairs = json.load(f)

    pairs = []
    n_pos = 0
    n_neg = 0
    skipped = 0

    for p in raw_pairs:
        src_id = p["source_id"]
        tgt_id = p["target_id"]
        label  = p["label"]

        if src_id not in id_map or tgt_id not in id_map:
            skipped += 1
            continue

        hlr_sum, hlr_des = id_map[src_id]
        llr_sum, llr_des = id_map[tgt_id]
        pairs.append((hlr_sum, hlr_des, llr_sum, llr_des, label))

        if label == 1:
            n_pos += 1
        else:
            n_neg += 1

    # Validation: Check we loaded something
    if len(pairs) == 0:
        raise ValueError(f"No valid pairs loaded from {fixed_pairs_file}. "
                         f"Check that requirement IDs match between "
                         f"requirements.json and fixed pairs file. "
                         f"Skipped: {skipped}")
    
    if skipped > 0:
        print(f"  WARNING: Skipped {skipped}/{len(raw_pairs)} pairs "
              f"(missing req IDs in requirements.json)")

    # Only shuffle training data for batch diversity
    # Keep val/test in fixed order for reproducibility
    if shuffle:
        random.shuffle(pairs)
    
    return pairs, n_pos, n_neg


# =================================================
# 4. METRICS
# =================================================

def f_beta(p, r, beta=1.0):
    """Compute F-beta score."""
    if p == 0 and r == 0:
        return 0.0
    b2    = beta ** 2
    denom = b2 * p + r
    return (1 + b2) * p * r / denom if denom > 0 else 0.0


# =================================================
# 5. EVALUATION
# =================================================

@torch.no_grad()
def compute_val_loss(model, loader, criterion):
    """
    Compute average weighted BCE loss on validation set.
    Used for early stopping — smooth signal unlike F1/F2.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        *fields, labels = batch
        fields = [f.to(device) for f in fields]
        labels = labels.to(device)
        logits = model(*fields)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    """Compute P, R, F1, F2 at a given threshold."""
    model.eval()
    y_true, y_scores = [], []

    for batch in loader:
        *fields, labels = batch
        fields = [f.to(device) for f in fields]
        logits = model(*fields)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        y_scores.extend(probs.cpu().numpy())
        y_true.extend(labels.numpy())

    y_scores = np.array(y_scores)
    y_true   = np.array(y_true)
    y_pred   = (y_scores >= threshold).astype(int)

    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    
    f1 = f_beta(p, r, 1.0)
    f2 = f_beta(p, r, 2.0)
    
    return p, r, f1, f2


@torch.no_grad()
def diagnose(model, loader):
    """
    Verify cosine similarity distributions before and after training.
    
    Processes ALL batches in the loader for accurate statistics.
    Expected: Positive pairs have higher similarity than negative pairs.
    Frozen BERT should preserve this gap throughout training.
    """
    model.eval()
    
    # Collect probability distributions and cosine similarities over ALL batches
    all_probs, all_labels = [], []
    all_pos_sims = {name: [] for name in ["sim(u,m)", "sim(u,n)", 
                                            "sim(v,m)", "sim(v,n)"]}
    all_neg_sims = {name: [] for name in ["sim(u,m)", "sim(u,n)", 
                                            "sim(v,m)", "sim(v,n)"]}
    
    for batch in loader:
        *fields, labels = batch
        fields = [f.to(device) for f in fields]
        labels_dev = labels.to(device)
        
        logits = model(*fields)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
        
        # Compute cosine similarities
        hs_ids, hs_mask = fields[0], fields[1]
        hd_ids, hd_mask = fields[2], fields[3]
        ls_ids, ls_mask = fields[4], fields[5]
        ld_ids, ld_mask = fields[6], fields[7]
        
        u = model.encode_field(hs_ids, hs_mask)
        v = model.encode_field(hd_ids, hd_mask)
        m = model.encode_field(ls_ids, ls_mask)
        n = model.encode_field(ld_ids, ld_mask)
        
        pos = labels_dev.bool()
        neg = ~pos
        
        if pos.sum() > 0:
            for name, a, b in [("sim(u,m)", u, m), ("sim(u,n)", u, n),
                                ("sim(v,m)", v, m), ("sim(v,n)", v, n)]:
                s = F.cosine_similarity(a, b, dim=1)
                all_pos_sims[name].extend(s[pos].cpu().numpy().tolist())
        
        if neg.sum() > 0:
            for name, a, b in [("sim(u,m)", u, m), ("sim(u,n)", u, n),
                                ("sim(v,m)", v, m), ("sim(v,n)", v, n)]:
                s = F.cosine_similarity(a, b, dim=1)
                all_neg_sims[name].extend(s[neg].cpu().numpy().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    n_pos = int(all_labels.sum())
    n_neg = int((all_labels == 0).sum())
    print(f"    Samples analysed: {n_pos} pos, {n_neg} neg")

    if n_pos > 0 and n_neg > 0:
        pos_prob_mean = all_probs[all_labels==1].mean()
        neg_prob_mean = all_probs[all_labels==0].mean()
        print(f"    Prob(link) — pos: {pos_prob_mean:.4f}  "
              f"neg: {neg_prob_mean:.4f}  "
              f"gap: {pos_prob_mean - neg_prob_mean:+.4f}")
        print(f"    Range: [{all_probs.min():.3f}, {all_probs.max():.3f}]")
    
    # Print aggregated cosine similarity statistics
    print("    Cosine similarities (full validation set):")
    for name in ["sim(u,m)", "sim(u,n)", "sim(v,m)", "sim(v,n)"]:
        if all_pos_sims[name] and all_neg_sims[name]:
            pos_mean = np.mean(all_pos_sims[name])
            neg_mean = np.mean(all_neg_sims[name])
            gap = pos_mean - neg_mean
            print(f"      {name} — pos: {pos_mean:.4f}  "
                  f"neg: {neg_mean:.4f}  gap: {gap:+.4f}")
        elif all_pos_sims[name]:
            print(f"      {name} — pos: {np.mean(all_pos_sims[name]):.4f}  "
                  f"neg: N/A (no negative samples)")
        elif all_neg_sims[name]:
            print(f"      {name} — pos: N/A (no positive samples)  "
                  f"neg: {np.mean(all_neg_sims[name]):.4f}")


def tune_threshold(model, val_loader):
    """
    Sweep thresholds on validation set, pick best F2.
    
    Returns:
        best_t: Threshold that maximizes F2 on validation set
    """
    thresholds = [round(0.05 * i + 0.05, 2) for i in range(19)]
    best_f2    = 0.0
    best_t     = 0.5
    rows       = []

    for t in thresholds:
        p, r, f1, f2 = evaluate(model, val_loader, threshold=t)
        mark = ""
        if f2 > best_f2:
            best_f2, best_t = f2, t
            mark = "  ←"
        rows.append((t, p, r, f1, f2, mark))

    print(f"\n  Threshold tuning (optimising F2 on validation set):")
    print(f"  {'t':>5}  {'P':>7}  {'R':>7}  {'F1':>7}  {'F2':>7}")
    print("  " + "-"*42)
    for t, p, r, f1, f2, mark in rows:
        print(f"  {t:>5.2f}  {p:>7.4f}  {r:>7.4f}  "
              f"{f1:>7.4f}  {f2:>7.4f}{mark}")
    print(f"\n  ✓ Best threshold: {best_t}  (Val F2: {best_f2:.4f})")
    return best_t


# =================================================
# 6. MAIN
# =================================================

def run_experiment():
    tokenizer   = BertTokenizer.from_pretrained(BERT_NAME)
    all_results = {}
    project_stats = {}

    for proj in PROJECTS:
        print("\n" + "=" * 64)
        print(f"  Project: {proj}")
        print("=" * 64)

        proj_path = os.path.join(BASE_DIR, proj)
        start_time = time.time()

        # ── Load requirements text ────────────────────────────────────
        try:
            id_map = load_requirements(proj_path)
            print(f"  Loaded {len(id_map)} requirements")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        # ── Load FIXED pairs (no random sampling) ─────────────────────
        splits_dir = os.path.join(proj_path, "splits")
        print("  Loading fixed pairs...")

        try:
            train_pairs, n_pos, n_neg = load_fixed_pairs(
                os.path.join(splits_dir, "final_pairs_train.json"), 
                id_map, 
                shuffle=True)  # Shuffle train for batch diversity
            val_pairs, vp, vn = load_fixed_pairs(
                os.path.join(splits_dir, "final_pairs_val.json"), 
                id_map, 
                shuffle=False)  # Keep val order fixed
            test_pairs, tp, tn = load_fixed_pairs(
                os.path.join(splits_dir, "final_pairs_test.json"), 
                id_map, 
                shuffle=False)  # Keep test order fixed
        except (FileNotFoundError, ValueError) as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Train: {n_pos:,} pos / {n_neg:,} neg  "
              f"(ratio 1:{n_neg/max(n_pos,1):.1f})")
        print(f"  Val:   {vp:,} pos / {vn:,} neg  "
              f"(ratio 1:{vn/max(vp,1):.1f})")
        print(f"  Test:  {tp:,} pos / {tn:,} neg  "
              f"(ratio 1:{tn/max(tp,1):.1f})")

        # Save stats for final output
        project_stats[proj] = {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "test_pairs": len(test_pairs),
            "test_positives": tp,
            "test_negatives": tn
        }

        train_loader = DataLoader(
            TraceDataset(train_pairs, tokenizer),
            batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(
            TraceDataset(val_pairs,   tokenizer),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        test_loader  = DataLoader(
            TraceDataset(test_pairs,  tokenizer),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        # ── Model ─────────────────────────────────────────────────────
        model = DRAFTTextOnly().to(device)

        # Loss weights from sampled training distribution (1:3)
        # Paper Eq. 14: alpha/beta from actual training imbalance
        alpha  = (n_pos + n_neg) / max(n_pos, 1)   # ~4.0 for 1:3
        beta_w = (n_pos + n_neg) / max(n_neg, 1)   # ~1.33 for 1:3
        weights   = torch.tensor([beta_w, alpha],
                                 dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"  Loss weights: alpha(pos)={alpha:.3f}  "
              f"beta(neg)={beta_w:.3f}")

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR)

        # ── Diagnostic: untrained ────────────────────────────────────
        print("\n  Diagnostic (untrained model):")
        diagnose(model, val_loader)

        # ── Training — early stopping on VALIDATION LOSS ─────────────
        best_val_loss = float('inf')
        best_state    = None
        patience_ctr  = 0

        print(f"\n  Training (max {EPOCHS} epochs, "
              f"patience={PATIENCE} on val loss, min={MIN_EPOCHS})...")
        print(f"  {'Epoch':>7}  {'Train Loss':>11}  "
              f"{'Val Loss':>9}  {'Val F1':>7}  {'Val F2':>7}")
        print("  " + "-"*52)

        for epoch in range(EPOCHS):
            # Train
            model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                *fields, labels = batch
                fields = [f.to(device) for f in fields]
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(*fields)
                loss   = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validate — compute BOTH loss and metrics for logging
            val_loss         = compute_val_loss(model, val_loader, criterion)
            _, _, val_f1, val_f2 = evaluate(model, val_loader)

            improved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_ctr  = 0
                best_state    = {k: v.clone()
                                 for k, v in model.state_dict().items()}
                improved = "  ← best"
            else:
                if epoch >= MIN_EPOCHS - 1:
                    patience_ctr += 1

            print(f"  {epoch+1:>7}  {avg_train_loss:>11.4f}  "
                  f"{val_loss:>9.4f}  {val_f1:>7.4f}  "
                  f"{val_f2:>7.4f}{improved}")

            if patience_ctr >= PATIENCE and epoch >= MIN_EPOCHS - 1:
                print("  Early stopping (val loss plateau).")
                break

        # Restore best checkpoint (lowest val loss)
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"\n  ✓ Restored best weights "
                  f"(Val loss: {best_val_loss:.4f})")

        # ── Diagnostic: trained ───────────────────────────────────────
        print("\n  Diagnostic (trained model):")
        diagnose(model, val_loader)

        # ── Post-hoc threshold tuning on val set ─────────────────────
        best_t = tune_threshold(model, val_loader)

        # ── Final test evaluation ─────────────────────────────────────
        p, r, f1, f2 = evaluate(model, test_loader, threshold=best_t)
        
        training_time = time.time() - start_time
        
        all_results[proj] = {
            "precision": float(p), 
            "recall": float(r), 
            "f1": float(f1), 
            "f2": float(f2),
            "threshold": float(best_t),
            "training_time_seconds": float(training_time)
        }

        print(f"\n  " + "="*60)
        print(f"  TEST RESULTS (threshold={best_t:.2f}):")
        print(f"  Precision: {p:.4f}  |  Recall: {r:.4f}")
        print(f"  F1: {f1:.4f}  |  F2: {f2:.4f}")
        print(f"  Training time: {training_time/60:.1f} minutes")
        print(f"  " + "="*60)

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 72)
    print("FINAL SUMMARY — BERT Frozen Encoder (Fixed 1:3 Pairs)")
    print("=" * 72)
    print(f"{'Project':<12} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'F2':>10} {'Threshold':>10}")
    print("-" * 72)

    ps, rs, f1s, f2s = [], [], [], []
    for proj in PROJECTS:
        if proj in all_results:
            res = all_results[proj]
            print(f"{proj:<12} {res['precision']:>10.4f} {res['recall']:>10.4f} "
                  f"{res['f1']:>10.4f} {res['f2']:>10.4f} "
                  f"{res['threshold']:>10.2f}")
            ps.append(res["precision"])
            rs.append(res["recall"])
            f1s.append(res["f1"])
            f2s.append(res["f2"])

    if len(ps) > 0:
        print("-" * 72)
        print(f"{'Average':<12} {np.mean(ps):>10.4f} {np.mean(rs):>10.4f} "
              f"{np.mean(f1s):>10.4f} {np.mean(f2s):>10.4f}")
    print("=" * 72)

    # ══════════════════════════════════════════════════════════════════
    # SAVE RESULTS TO JSON
    # ══════════════════════════════════════════════════════════════════
    results_file = os.path.join(OUTPUT_DIR, "frozen_bert_final_pairs_results.json")
    
    final_output = {
        "experiment_info": {
            "model": BERT_NAME,
            "evaluation_strategy": "fixed_1to3_pairs",
            "max_sequence_length": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "seed": SEED,
            "early_stopping_metric": "validation_loss",
            "threshold_tuning_metric": "validation_f2"
        },
        "project_statistics": project_stats,
        "results": all_results,
        "summary": {
            "avg_precision": float(np.mean(ps)) if len(ps) > 0 else None,
            "avg_recall": float(np.mean(rs)) if len(rs) > 0 else None,
            "avg_f1": float(np.mean(f1s)) if len(f1s) > 0 else None,
            "avg_f2": float(np.mean(f2s)) if len(f2s) > 0 else None,
            "num_projects_completed": len(all_results)
        }
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"  Completed {len(all_results)}/{len(PROJECTS)} projects")


if __name__ == "__main__":
    run_experiment()

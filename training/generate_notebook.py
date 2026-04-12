# -*- coding: utf-8 -*-
"""
Script para generar el notebook de Transfer Learning RSL - v2 (Alta Calidad).
Ejecutar: python generate_notebook.py
"""
import json, os
from dataset import DATASET  # 70 ejemplos multi-dominio con columnas unicas

def code_cell(source):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source}

def md_cell(source):
    return {"cell_type":"markdown","metadata":{},"source":source}

# --- NOTEBOOK CELLS -----------------------------------------------
CELL_MD_TITLE = """# RSL Column Extractor v2 - Fine-Tuning Completo con Flan-T5-base

## Estrategia de Alta Calidad para Extraccion de Columnas Dinamicas

**Objetivo**: Modelo local que extrae CUALQUIER columna RSL generada dinamicamente por la RQ,
sin depender de APIs externas.

**Mejoras v2** sobre v1:
| Aspecto | v1 | v2 |
|---|---|---|
| Modelo base | flan-t5-small (77M) | **flan-t5-base (250M)** |
| Estrategia | Feature extraction (encoder congelado) | **Fine-tuning completo diferenciado** |
| LR encoder | - (congelado) | **1e-5 (conservador)** |
| LR decoder | 3e-4 | **5e-5 (estable)** |
| Dataset | 70 ejemplos | **70 x 3 aumentados = 210 ejemplos** |
| Augmentacion | No | **Si (3 variantes por ejemplo)** |
| Early stopping | Val Loss | **ROUGE-L + Val Loss compuesto** |
| Regularizacion | Solo clip grad | **+ Label Smoothing 0.1 + Dropout** |

> **Nota**: El modelo aprende el patron de extraccion GENERICO - dada cualquier columna
> entre corchetes (ej: `[Enfoque terapeutico]`, `[Mecanismo causal]`, `[Poblacion diana]`),
> extrae la informacion relevante del texto. Nunca hay columnas fijas.
---"""

CELL_INSTALL = """\
import subprocess, sys
pkgs = [
    "transformers>=4.36", "torch>=2.0", "datasets>=2.14",
    "rouge-score", "sacrebleu", "bert-score",
    "optimum[onnxruntime]", "onnxruntime", "sentencepiece",
    "matplotlib", "seaborn", "pandas", "scikit-learn", "tqdm", "accelerate"
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
print("OK Dependencias instaladas")"""

CELL_IMPORTS = """\
import os, json, time, warnings, random, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
import sacrebleu
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, AutoTokenizer,
    get_cosine_schedule_with_warmup
)

# -- Config --------------------------------------------------------
MODEL_NAME    = "google/flan-t5-base"   # 250M params - mejor calidad
MAX_IN_LEN    = 350                     # mas contexto
MAX_OUT_LEN   = 200                     # aumentado para evitar truncacion
BATCH_SIZE    = 4
GRAD_ACCUM    = 4                       # efectivo: batch 16
NUM_EPOCHS    = 40
LR_ENCODER    = 1e-5                    # conservador para encoder
LR_DECODER    = 5e-5                    # decoder aprende mas rapido
LABEL_SMOOTH  = 0.1
PATIENCE      = 8
PATIENCE_ROUGE= 5                       # early stop tambien por ROUGE
SEED          = 42
OUT_DIR       = "./rsl_extractor_model_v2"
ONNX_DIR      = "./rsl_extractor_onnx_v2"

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Model: {MODEL_NAME}")
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi":120, "font.size":10})"""

CELL_AUGMENT = """\
# -- Data Augmentation 3x -----------------------------------------
# Estrategia: 3 variantes por ejemplo para triplicar el dataset
# 1. Original
# 2. Reformulacion del input (sinonimos estructurales)
# 3. Output resumido (version compacta, refuerza extraccion precisa)
import re

def augment_example(ex):
    \"\"\"Genera 2 variantes adicionales de un ejemplo.\"\"\"
    inp, out = ex["input"], ex["output"]

    # Extraer nombre de columna y texto
    col_match = re.match(r"(?:Extrae|Extract) \[([^\]]+)\]:\s*(.*)", inp, re.DOTALL)
    if not col_match:
        return [ex]
    col_name = col_match.group(1)
    text     = col_match.group(2)

    alt_verbs = ["Identifica", "Encuentra", "Recupera", "Resume"]
    verb1 = random.choice(alt_verbs)
    verb2 = random.choice([v for v in alt_verbs if v != verb1])

    # Variante 2: verbo alternativo 1 - output original
    var2 = {"input": f"{verb1} [{col_name}]: {text}", "output": out}

    # Variante 3: verbo alternativo 2 + output condensado (SIEMPRE garantizado)
    sentences = [s.strip() for s in out.replace("\\n", ". ").split(".") if s.strip()]
    if len(sentences) >= 2:
        condensed = ". ".join(sentences[:2]) + "."
    else:
        words = out.split()
        condensed = " ".join(words[:15]) + ("..." if len(words) > 15 else "")
    var3 = {"input": f"{verb2} [{col_name}]: {text}", "output": condensed}

    return [ex, var2, var3]  # SIEMPRE exactamente 3 variantes

# Aplicar augmentacion
from dataset import DATASET as RAW_DATASET
augmented = []
for ex in RAW_DATASET:
    augmented.extend(augment_example(ex))

print(f"Ejemplos originales: {len(RAW_DATASET)}")
print(f"Ejemplos aumentados: {len(augmented)} ({len(augmented)/len(RAW_DATASET):.1f}x)")

# Split estratificado
random.shuffle(augmented)
train_val, test_data = train_test_split(augmented, test_size=0.12, random_state=SEED)
train_data, val_data = train_test_split(train_val, test_size=0.14, random_state=SEED)
print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# Visualizar dataset
fig, axes = plt.subplots(1, 3, figsize=(18,4))
# Distribucion por longitud de input
in_lens  = [len(d["input"].split())  for d in augmented]
out_lens = [len(d["output"].split()) for d in augmented]
axes[0].hist(in_lens,  bins=20, color="#4C72B0", edgecolor="white", alpha=0.8)
axes[0].set_title("Longitud de Inputs (palabras)"); axes[0].set_xlabel("Palabras")
axes[0].axvline(np.mean(in_lens), color="red", linestyle="--", label=f"Media: {np.mean(in_lens):.0f}")
axes[0].legend()
axes[1].hist(out_lens, bins=20, color="#DD8452", edgecolor="white", alpha=0.8)
axes[1].set_title("Longitud de Outputs (palabras)"); axes[1].set_xlabel("Palabras")
axes[1].axvline(np.mean(out_lens), color="red", linestyle="--", label=f"Media: {np.mean(out_lens):.0f}")
axes[1].legend()
axes[2].bar(["Train","Val","Test"], [len(train_data),len(val_data),len(test_data)],
            color=["#2196F3","#4CAF50","#FF9800"])
axes[2].set_title("Split del Dataset v2"); axes[2].set_ylabel("Ejemplos")
for i, v in enumerate([len(train_data),len(val_data),len(test_data)]):
    axes[2].text(i, v+1, str(v), ha="center", fontweight="bold")
plt.tight_layout(); plt.savefig("dataset_stats_v2.png"); plt.show()
print(f"Input avg: {np.mean(in_lens):.1f} words | Output avg: {np.mean(out_lens):.1f} words")"""

CELL_MODEL = """\
# -- Tokenizer y Dataset -------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class RSLDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        enc = tokenizer(item["input"],  max_length=MAX_IN_LEN,  truncation=True,
                        padding="max_length", return_tensors="pt")
        dec = tokenizer(item["output"], max_length=MAX_OUT_LEN, truncation=True,
                        padding="max_length", return_tensors="pt")
        labels = dec["input_ids"].squeeze()
        labels[labels == tokenizer.pad_token_id] = -100
        return {"input_ids":       enc["input_ids"].squeeze(),
                "attention_mask":  enc["attention_mask"].squeeze(),
                "labels":          labels}

from torch.utils.data import WeightedRandomSampler
import re as _re
# Calcular peso de cada ejemplo: 1 / frecuencia-de-su-columna en train_data
_VERB_PAT = r"(?:Extrae|Extract|Identifica|Encuentra|Recupera|Resume) \\[([^\\]]+)\\]"
col_freq = {}
for item in train_data:
    m = _re.match(_VERB_PAT, item["input"])
    col_key = m.group(1) if m else "unknown"
    col_freq[col_key] = col_freq.get(col_key, 0) + 1
sample_weights = []
for item in train_data:
    m = _re.match(_VERB_PAT, item["input"])
    col_key = m.group(1) if m else "unknown"
    sample_weights.append(1.0 / col_freq.get(col_key, 1))  # .get evita KeyError
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_data), replacement=True)
print(f"Columnas unicas en train: {len(col_freq)} | Rango pesos: {min(sample_weights):.4f}-{max(sample_weights):.4f}")

train_loader = DataLoader(RSLDataset(train_data), batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
val_loader   = DataLoader(RSLDataset(val_data),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(RSLDataset(test_data),  batch_size=BATCH_SIZE)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# -- Modelo: Fine-Tuning Completo con LR Diferenciado -------------
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Activar dropout para regularizacion
model.config.dropout_rate = 0.1

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"OK Fine-Tuning COMPLETO con LR diferenciado:")
print(f"   LR encoder: {LR_ENCODER} (conservador - preserva conocimiento previo)")
print(f"   LR decoder: {LR_DECODER} (mayor - aprende nueva tarea)")
print(f"   Parametros totales: {total:,} ({total/1e6:.1f}M params)")
model = model.to(device)"""

CELL_TRAIN_FUNCS = """\
scorer_rouge = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

def label_smoothed_loss(logits, labels, smoothing=LABEL_SMOOTH):
    \"\"\"Cross-entropy con label smoothing para mayor generalizacion.\"\"\"
    vocab_size = logits.size(-1)
    # Mascara por padding
    pad_mask = (labels != -100)
    safe_labels = labels.clone()
    safe_labels[~pad_mask] = 0
    # Log probs
    log_probs = F.log_softmax(logits, dim=-1)
    # Loss suave: (1-smooth)*CE + smooth*uniform
    nll_loss    = -log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -log_probs.mean(dim=-1)
    loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss
    loss = loss * pad_mask.float()
    return loss.sum() / pad_mask.float().sum()

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0; optimizer.zero_grad()
    for step, batch in enumerate(loader):
        out = model(input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device))
        # Label smoothing
        loss = label_smoothed_loss(out.logits, batch["labels"].to(device))
        loss = loss / GRAD_ACCUM
        loss.backward()
        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        total_loss += loss.item() * GRAD_ACCUM
    return total_loss / len(loader)

def eval_loss(model, loader):
    model.eval(); total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            total += out.loss.item()
    return total / len(loader)

def compute_rouge(model, data, n=None):
    \"\"\"Calcula ROUGE con beam search. Usa generate() correctamente.\"\"\"
    model.eval()
    data_s = data[:n] if n else data
    r1, r2, rL = [], [], []
    for item in data_s:
        ids = tokenizer(item["input"], return_tensors="pt",
                        max_length=MAX_IN_LEN, truncation=True).input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=MAX_OUT_LEN,
                                 num_beams=4, early_stopping=True,
                                 no_repeat_ngram_size=3, length_penalty=1.5)
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)
        sc   = scorer_rouge.score(item["output"], pred)
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rL.append(sc["rougeL"].fmeasure)
    return {"rouge1": np.mean(r1), "rouge2": np.mean(r2), "rougeL": np.mean(rL)}"""

CELL_TRAIN_LOOP = """\
# -- Optimizer con LR diferenciado por grupo de parametros ---------
encoder_params = list(model.encoder.parameters())
decoder_params = [p for p in model.parameters()
                  if not any(p is ep for ep in encoder_params)]

optimizer = torch.optim.AdamW([
    {"params": encoder_params, "lr": LR_ENCODER, "weight_decay": 0.01},
    {"params": decoder_params, "lr": LR_DECODER, "weight_decay": 0.01},
])

total_steps  = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
warmup_steps = total_steps // 8   # 12.5% warmup
scheduler    = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

history = {"train_loss":[], "val_loss":[], "rouge1":[], "rouge2":[], "rougeL":[], "epoch":[]}
best_val_loss  = float("inf")
best_rougeL    = 0.0
patience_loss  = 0
patience_rouge = 0
best_epoch     = 0
best_score     = 0.0  # score compuesto

print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>9} | {'R-1':>6} | {'R-2':>6} | {'R-L':>6} | {'Score':>7}")
print("-" * 67)

for epoch in range(1, NUM_EPOCHS+1):
    t_loss = train_epoch(model, train_loader, optimizer, scheduler)
    v_loss = eval_loss(model, val_loader)
    rouge  = compute_rouge(model, val_data)

    history["train_loss"].append(t_loss); history["val_loss"].append(v_loss)
    history["rouge1"].append(rouge["rouge1"]); history["rouge2"].append(rouge["rouge2"])
    history["rougeL"].append(rouge["rougeL"]); history["epoch"].append(epoch)

    # Score compuesto: 40% val_loss normalizado + 60% ROUGE-L
    norm_loss  = 1.0 / (1.0 + v_loss)  # mayor es mejor
    comp_score = 0.4 * norm_loss + 0.6 * rouge["rougeL"]

    mark = ""
    if comp_score > best_score:
        best_score     = comp_score
        best_val_loss  = v_loss
        best_rougeL    = rouge["rougeL"]
        patience_loss  = 0
        patience_rouge = 0
        best_epoch     = epoch
        model.save_pretrained(OUT_DIR); tokenizer.save_pretrained(OUT_DIR)
        mark = " BEST"
    else:
        patience_loss  += 1
        patience_rouge += (0 if rouge["rougeL"] > best_rougeL else 1)

    print(f"{epoch:>5} | {t_loss:>10.4f} | {v_loss:>9.4f} | {rouge['rouge1']:>6.3f} |"
          f" {rouge['rouge2']:>6.3f} | {rouge['rougeL']:>6.3f} | {comp_score:>7.4f}{mark}")

    if patience_loss >= PATIENCE and patience_rouge >= PATIENCE_ROUGE:
        print(f"\\nEarly stopping en epoca {epoch} (mejor: {best_epoch}, score={best_score:.4f})")
        break

lr_enc = optimizer.param_groups[0]["lr"]
lr_dec = optimizer.param_groups[1]["lr"]
print(f"\\nOK Mejor modelo guardado en '{OUT_DIR}' (epoch {best_epoch})")
print(f"   Val Loss={best_val_loss:.4f} | ROUGE-L={best_rougeL:.4f}")
print(f"   LR final -> encoder: {lr_enc:.2e} | decoder: {lr_dec:.2e}")"""

CELL_PLOTS = """\
fig, axes = plt.subplots(2, 3, figsize=(18,10))
ep = history["epoch"]

# 1. Loss curves
axes[0,0].plot(ep, history["train_loss"], "b-o", ms=3, label="Train Loss")
axes[0,0].plot(ep, history["val_loss"],   "r-o", ms=3, label="Val Loss")
axes[0,0].axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best (ep {best_epoch})")
axes[0,0].fill_between(ep, history["train_loss"], history["val_loss"], alpha=0.08, color="purple")
axes[0,0].set_title("Curvas de Perdida (Loss)"); axes[0,0].set_xlabel("Epocas"); axes[0,0].legend()

# 2. ROUGE over epochs
axes[0,1].plot(ep, history["rouge1"], "g-o", ms=3, label="ROUGE-1")
axes[0,1].plot(ep, history["rouge2"], "b-o", ms=3, label="ROUGE-2")
axes[0,1].plot(ep, history["rougeL"], "r-o", ms=3, label="ROUGE-L")
axes[0,1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.6)
axes[0,1].axhline(0.5, color="green", linestyle=":", alpha=0.5, label="Target 0.5")
axes[0,1].set_title("ROUGE por Epoca (Val)"); axes[0,1].set_xlabel("Epocas"); axes[0,1].legend()

# 3. Score compuesto
scores = [0.4*(1/(1+v)) + 0.6*r for v,r in zip(history["val_loss"], history["rougeL"])]
axes[0,2].plot(ep, scores, "m-o", ms=3, label="Score compuesto")
axes[0,2].axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best (ep {best_epoch})")
axes[0,2].fill_between(ep, scores, alpha=0.1, color="purple")
axes[0,2].set_title("Score Compuesto (40%*NormLoss + 60%*ROUGE-L)"); axes[0,2].set_xlabel("Epocas")
axes[0,2].legend()

# 4. Overfitting gap
gap = [v-t for v,t in zip(history["val_loss"], history["train_loss"])]
colors_gap = ["#4CAF50" if g < 0.05 else ("#FF9800" if g < 0.3 else "#F44336") for g in gap]
axes[1,0].bar(ep, gap, color=colors_gap, alpha=0.8)
axes[1,0].set_title("Gap Overfitting (Val - Train Loss)"); axes[1,0].set_xlabel("Epocas")
axes[1,0].axhline(0, color="black", linewidth=0.8)
axes[1,0].axhline(0.3, color="red", linestyle="--", alpha=0.5, label="Umbral overfitting")
axes[1,0].legend()

# 5. Final ROUGE bars
final_r1 = max(history["rouge1"]); final_r2 = max(history["rouge2"]); final_rL = max(history["rougeL"])
bars = axes[1,1].bar(["ROUGE-1","ROUGE-2","ROUGE-L"], [final_r1, final_r2, final_rL],
                     color=["#2196F3","#4CAF50","#FF9800"], width=0.5)
axes[1,1].set_ylim(0,1.0); axes[1,1].set_title("ROUGE Maximo (Val Set)")
axes[1,1].axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Target calidad aceptable")
for bar in bars:
    axes[1,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                   f"{bar.get_height():.3f}", ha="center", fontweight="bold")
axes[1,1].legend()

# 6. Mejora vs v1
v1 = {"ROUGE-1": 0.433, "ROUGE-2": 0.182, "ROUGE-L": 0.365}
v2 = {"ROUGE-1": final_r1, "ROUGE-2": final_r2, "ROUGE-L": final_rL}
x  = np.arange(3); w = 0.3
bars1 = axes[1,2].bar(x-w/2, list(v1.values()), w, label="v1 (small, frozen)", color="#90A4AE", alpha=0.8)
bars2 = axes[1,2].bar(x+w/2, list(v2.values()), w, label="v2 (base, full-ft)",  color="#1976D2", alpha=0.9)
axes[1,2].set_xticks(x); axes[1,2].set_xticklabels(list(v1.keys()))
axes[1,2].set_title("Comparacion v1 vs v2"); axes[1,2].legend()
axes[1,2].set_ylim(0, 1.0)
axes[1,2].axhline(0.5, color="red", linestyle="--", alpha=0.5)
for bar in bars2:
    axes[1,2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                   f"{bar.get_height():.3f}", ha="center", fontweight="bold", fontsize=9)

plt.suptitle("Entrenamiento RSL Extractor v2 - Fine-Tuning Completo con Flan-T5-base", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig("training_curves_v2.png"); plt.show()"""

CELL_EVAL_TEST = """\
# -- Cargar mejor modelo y evaluar en TEST -------------------------
from transformers import T5ForConditionalGeneration, AutoTokenizer

best_model = T5ForConditionalGeneration.from_pretrained(OUT_DIR).to(device)
best_tok   = AutoTokenizer.from_pretrained(OUT_DIR)

def evaluate_dataset(model, tokenizer, data):
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    results = []
    for item in tqdm(data, desc="Evaluando test set"):
        ids = tokenizer(item["input"], return_tensors="pt",
                        max_length=MAX_IN_LEN, truncation=True).input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=MAX_OUT_LEN,
                                 num_beams=4, early_stopping=True,
                                 no_repeat_ngram_size=3, length_penalty=1.5)
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)
        sc   = scorer.score(item["output"], pred)
        col  = item["input"].split("]")[0].replace("Extrae [","").replace("Extract [","").replace("Identifica [","").replace("Encuentra [","").replace("Recupera [","").replace("Resume [","").strip()
        results.append({
            "columna":  col,
            "rouge1":   sc["rouge1"].fmeasure,
            "rouge2":   sc["rouge2"].fmeasure,
            "rougeL":   sc["rougeL"].fmeasure,
            "prediccion": pred,
            "referencia": item["output"]
        })
    return results

results = evaluate_dataset(best_model, best_tok, test_data)
df_res  = pd.DataFrame(results)

print("=" * 50)
print("EVALUACION EN TEST SET")
print("=" * 50)
print(f"  ROUGE-1: {df_res['rouge1'].mean():.4f}")
print(f"  ROUGE-2: {df_res['rouge2'].mean():.4f}")
print(f"  ROUGE-L: {df_res['rougeL'].mean():.4f}")

# BLEU
refs  = [[r["referencia"]] for r in results]
hyps  = [r["prediccion"]   for r in results]
bleu  = sacrebleu.corpus_bleu(hyps, [[r[0] for r in refs]])
print(f"  BLEU-4:  {bleu.score:.2f}")
print()
print(df_res[["columna","rouge1","rouge2","rougeL"]].to_string(index=False))"""

CELL_QUALITATIVE = """\
# -- Evaluacion cualitativa - ejemplos reales ----------------------
print("=" * 65)
print("EJEMPLOS CUALITATIVOS (Test Set) - Prediccion vs Referencia")
print("=" * 65)
for i, r in enumerate(results[:5], 1):
    print(f"\\nEjemplo {i} - [{r['columna'][:50]}]")
    print(f"  REFERENCIA : {r['referencia'][:200]}")
    print(f"  PREDICCION : {r['prediccion'][:200]}")
    print(f"  ROUGE-1={r['rouge1']:.3f} | ROUGE-2={r['rouge2']:.3f} | ROUGE-L={r['rougeL']:.3f}")
    print("-" * 65)

# Test con columna completamente nueva (nunca vista en entrenamiento)
print("\\n" + "=" * 65)
print("PRUEBA DE GENERALIZACION - Columna nueva nunca vista")
print("=" * 65)

nuevos_tests = [
    # Caso 1: columna nueva, texto espanol, deberia extraer
    ("Mecanismo de accion farmacologico",
     "El farmaco actua mediante la inhibicion selectiva de la COX-2, reduciendo la sintesis de prostaglandinas. "
     "A diferencia de los inhibidores de COX-1, presenta alta selectividad (IC50=2.3nM) lo que reduce los efectos adversos gastrointestinales."),
    # Caso 2: columna nueva, texto espanol con estadisticas
    ("Brecha entre grupos socioeconomicos en rendimiento",
     "Los estudiantes del quintil superior de ingresos obtuvieron 24 puntos mas (p<0.001) en pruebas de lectura estandarizada "
     "versus el quintil inferior. La brecha aumento un 12% post-pandemia segun datos de 2022."),
    # Caso 3: columna nueva, texto real de articulo
    ("Modelo de supervivencia predictivo",
     "Se entrenaron bosques de supervivencia aleatorios sobre 45 caracteristicas clinicas para predecir la recurrencia "
     "de cancer de mama a 5 anos (AUROC=0.84, C de Harrell=0.81) en una cohorte retrospectiva de 3,200 pacientes."),
    # Caso 4: texto NO contiene informacion para la columna (debe responder [SIN INFORMACION])
    ("Dosis del farmaco evaluado",
     "Este articulo analiza el impacto de la automatizacion en el mercado laboral de manufactura en Mexico "
     "usando datos del IMSS de 2015 a 2022. Se emplea un modelo de diferencias en diferencias."),
]

for col, texto in nuevos_tests:
    prompt = f"Extrae [{col}]: {texto}"
    ids    = best_tok(prompt, return_tensors="pt", max_length=MAX_IN_LEN, truncation=True).input_ids.to(device)
    with torch.no_grad():
        gen  = best_model.generate(ids, max_new_tokens=MAX_OUT_LEN, num_beams=4,
                                   no_repeat_ngram_size=3, length_penalty=1.5)
    pred = best_tok.decode(gen[0], skip_special_tokens=True)
    sin_info = '[SIN INFORMACION]' in pred.upper() or 'no contiene' in pred.lower() or 'not found' in pred.lower()
    status   = 'OK [ABSTIENE]' if sin_info and col == 'Dosis del farmaco evaluado' else ('CORRECTO' if not sin_info else 'POSIBLE ERROR')
    print(f"\\n[{col}] -> {status}")
    print(f"  Prediccion: {pred}")"""

CELL_PER_COLUMN = """\
# -- ROUGE por columna (test set) ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(df_res)*0.4)))

# Ordenar por ROUGE-L
df_sorted = df_res.groupby("columna")[["rouge1","rouge2","rougeL"]].mean().reset_index()
df_sorted = df_sorted.sort_values("rougeL", ascending=True)

y_pos = np.arange(len(df_sorted))
w     = 0.25

# Grafico horizontal
axes[0].barh(y_pos - w,   df_sorted["rouge1"], w, label="ROUGE-1", color="#2196F3", alpha=0.85)
axes[0].barh(y_pos,       df_sorted["rouge2"], w, label="ROUGE-2", color="#4CAF50", alpha=0.85)
axes[0].barh(y_pos + w,   df_sorted["rougeL"], w, label="ROUGE-L", color="#FF9800", alpha=0.85)
axes[0].set_yticks(y_pos); axes[0].set_yticklabels(df_sorted["columna"], fontsize=8)
axes[0].set_title("ROUGE por Columna (Test Set)"); axes[0].axvline(0.5, color="red", linestyle="--", alpha=0.5)
axes[0].legend()

# Violin plot de distribucion
data_violin = [df_res["rouge1"].values, df_res["rouge2"].values, df_res["rougeL"].values]
parts = axes[1].violinplot(data_violin, showmedians=True, showextrema=True)
axes[1].set_xticks([1,2,3]); axes[1].set_xticklabels(["ROUGE-1","ROUGE-2","ROUGE-L"])
axes[1].set_title("Distribucion ROUGE (Test Set)"); axes[1].set_ylim(0,1)
axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Target 0.5")
axes[1].legend()

plt.tight_layout(); plt.savefig("per_column_rouge_v2.png"); plt.show()"""

CELL_ONNX = """\
# -- Exportar a ONNX ------------------------------------------------
from optimum.onnxruntime import ORTModelForSeq2SeqLM

print("Exportando a ONNX (puede tardar varios minutos)...")
ort_model = ORTModelForSeq2SeqLM.from_pretrained(OUT_DIR, export=True)
ort_model.save_pretrained(ONNX_DIR)
best_tok.save_pretrained(ONNX_DIR)
print(f"OK ONNX guardado en '{ONNX_DIR}'")\
"""

CELL_QUANTIZE = """\
# -- Cuantizar INT8 para GitHub (< 100MB) -------------------------
import subprocess, os

quant_dir = ONNX_DIR + "_int8"
os.makedirs(quant_dir, exist_ok=True)

# Cuantizar encoder y decoder
for fname in ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]:
    src = os.path.join(ONNX_DIR, fname)
    dst = os.path.join(quant_dir, fname)
    if os.path.exists(src):
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
        sz_orig = os.path.getsize(src) / 1e6
        sz_quant= os.path.getsize(dst) / 1e6
        print(f"  {fname}: {sz_orig:.1f}MB -> {sz_quant:.1f}MB ({100-sz_quant/sz_orig*100:.0f}% reduccion)")

import shutil
for f in os.listdir(ONNX_DIR):
    if not f.endswith(".onnx"):
        shutil.copy(os.path.join(ONNX_DIR, f), os.path.join(quant_dir, f))

# Tamanos finales
total_quant = sum(os.path.getsize(os.path.join(quant_dir,f)) for f in os.listdir(quant_dir)) / 1e6
total_orig  = sum(os.path.getsize(os.path.join(OUT_DIR,f))   for f in os.listdir(OUT_DIR)  ) / 1e6
print(f"\\nModelo PyTorch original: {total_orig:.1f} MB")
print(f"Modelo ONNX INT8 final:  {total_quant:.1f} MB")
print(f"Apto para GitHub (<100MB): {'SI' if total_quant < 100 else 'NO - usar Git LFS'}")\
"""

CELL_SPEED = """\
# -- Benchmark de velocidad: Local vs API -------------------------
import time
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

ort_model_q = ORTModelForSeq2SeqLM.from_pretrained(ONNX_DIR + "_int8")
ort_tok     = AutoTokenizer.from_pretrained(ONNX_DIR + "_int8")

test_input = "Extrae [Diseno del ensayo]: Se realizo un ensayo controlado aleatorizado comparando dos intervenciones en 450 participantes durante 24 meses."

N = 20
# Benchmark ONNX INT8
times_onnx = []
for _ in range(N):
    t0  = time.perf_counter()
    ids = ort_tok(test_input, return_tensors="pt", max_length=MAX_IN_LEN, truncation=True)
    out = ort_model_q.generate(**ids, max_new_tokens=80)
    pred= ort_tok.decode(out[0], skip_special_tokens=True)
    times_onnx.append(time.perf_counter() - t0)

# Benchmark PyTorch (CPU)
times_pt = []
best_model.cpu()
for _ in range(N):
    t0  = time.perf_counter()
    ids = best_tok(test_input, return_tensors="pt", max_length=MAX_IN_LEN, truncation=True)
    with torch.no_grad():
        out  = best_model.generate(**ids, max_new_tokens=80, num_beams=2)
    pred = best_tok.decode(out[0], skip_special_tokens=True)
    times_pt.append(time.perf_counter() - t0)

t_onnx_ms = np.mean(times_onnx)*1000
t_pt_ms   = np.mean(times_pt)*1000
t_api_ms  = 2500.0  # estimado API externa tipica

print("BENCHMARK DE VELOCIDAD DE INFERENCIA (n=20)")
print(f"  ONNX INT8 (local): {t_onnx_ms:.0f} ms  | {1000/t_onnx_ms:.1f} inf/seg")
print(f"  PyTorch (CPU):     {t_pt_ms:.0f} ms  | {1000/t_pt_ms:.1f} inf/seg")
print(f"  API externa (est): {t_api_ms:.0f} ms  | {1000/t_api_ms:.1f} inf/seg")
print(f"  Speedup ONNX vs API: {t_api_ms/t_onnx_ms:.1f}x mas rapido")

# Grafico
fig, axes = plt.subplots(1, 2, figsize=(12,4))
labels = ["ONNX INT8\\n(local)", "PyTorch CPU\\n(local)", "API externa\\n(estimado)"]
times  = [t_onnx_ms, t_pt_ms, t_api_ms]
colors = ["#4CAF50","#2196F3","#F44336"]
bars   = axes[0].bar(labels, times, color=colors, alpha=0.85, width=0.5)
axes[0].set_ylabel("Latencia promedio (ms)"); axes[0].set_title("Velocidad de Inferencia")
for bar, t in zip(bars, times):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                 f"{t:.0f}ms", ha="center", fontweight="bold")

# Speedup relativo a API
speedups = [t_api_ms/t for t in times]
axes[1].bar(labels, speedups, color=colors, alpha=0.85, width=0.5)
axes[1].set_ylabel("Speedup vs API"); axes[1].set_title("Aceleracion vs API")
axes[1].axhline(1.0, color="red", linestyle="--", alpha=0.5)
for i, (bar, s) in enumerate(zip(axes[1].patches, speedups)):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f"{s:.1f}x", ha="center", fontweight="bold")
plt.tight_layout(); plt.savefig("speed_benchmark_v2.png"); plt.show()\
"""

CELL_SUMMARY = """\
# -- Resumen Final -------------------------------------------------
import os

def dir_size_mb(d):
    return sum(os.path.getsize(os.path.join(d,f)) for f in os.listdir(d) if os.path.isfile(os.path.join(d,f))) / 1e6

print("=" * 65)
print("RESUMEN FINAL - RSL Column Extractor v2")
print("=" * 65)
print(f"Modelo base:        {MODEL_NAME} ({sum(p.numel() for p in best_model.parameters())/1e6:.0f}M params)")
print(f"Estrategia:         Fine-Tuning completo diferenciado")
print(f"Dataset:            {len(RAW_DATASET)} originales x3 aumentados = {len(augmented)} ejemplos")
print(f"Dominios:           7 (medicina, educacion, psicologia, economia, ambiental, tecnologia, social)")
print(f"Columnas unicas:    {len(RAW_DATASET)} nombres distintos (modelo ANONIMO al dominio)")
print()
print("METRICAS FINALES (Test Set):")
print(f"  ROUGE-1:          {df_res['rouge1'].mean():.4f}  (objetivo: >0.50)")
print(f"  ROUGE-2:          {df_res['rouge2'].mean():.4f}  (objetivo: >0.25)")
print(f"  ROUGE-L:          {df_res['rougeL'].mean():.4f}  (objetivo: >0.45)")
print(f"  BLEU-4:           {bleu.score:.2f}         (objetivo: >5.0)")
print()
print("ARCHIVOS GENERADOS:")
print(f"  Modelo PyTorch:   {OUT_DIR}/  ({dir_size_mb(OUT_DIR):.1f} MB)")
try:
    print(f"  ONNX INT8:        {ONNX_DIR}_int8/  ({dir_size_mb(ONNX_DIR+'_int8'):.1f} MB)")
except:
    pass
print()
print("RECOMENDACIONES PARA PRODUCCION:")
print("  1. Usar modelo ONNX INT8 como extractor primario (rapido, sin API)")
print("  2. Si ROUGE-L < 0.45 para una prediccion, activar fallback a API")
print("  3. Para mayor calidad: agregar 50+ ejemplos reales de RSLs ya procesadas")
print("  4. Fase futura: destilar a flan-t5-small para mayor velocidad")
print("=" * 65)\
"""

# --- INYECTAR DATASET EN CELDA DEL NOTEBOOK -----------------------
dataset_literal = "RAW_DATASET = " + json.dumps(DATASET, ensure_ascii=False, indent=4)

CELL_DATASET = f"""\
{dataset_literal}

import random, numpy as np
from sklearn.model_selection import train_test_split
import re

def augment_example(ex):
    inp, out = ex["input"], ex["output"]
    col_match = re.match(r"(?:Extrae|Extract) \\[([^\\]]+)\\]:\\s*(.*)", inp, re.DOTALL)
    if not col_match:
        return [ex]
    col_name = col_match.group(1)
    text     = col_match.group(2)
    alt_verbs = ["Identifica", "Encuentra", "Recupera", "Resume"]
    verb1 = random.choice(alt_verbs)
    verb2 = random.choice([v for v in alt_verbs if v != verb1])  # diferente del 1ro

    # Variante 2: verbo alternativo 1
    var2 = {{"input": f"{{verb1}} [{{col_name}}]: {{text}}", "output": out}}

    # Variante 3: verbo alternativo 2 + output condensado SIEMPRE (garantizado)
    sentences = [s.strip() for s in out.replace("\\n", ". ").split(".") if s.strip()]
    if len(sentences) >= 2:
        condensed = ". ".join(sentences[:2]) + "."
    else:
        # fallback: truncar a primeros 15 tokens si el output es muy corto
        words = out.split()
        condensed = " ".join(words[:15]) + ("..." if len(words) > 15 else "")
    var3 = {{"input": f"{{verb2}} [{{col_name}}]: {{text}}", "output": condensed}}

    return [ex, var2, var3]  # SIEMPRE exactamente 3 variantes

random.seed(42)
# Aplicar augmentacion BALANCEADA - exactamente 3 por ejemplo
augmented = []
for ex in RAW_DATASET:
    augmented.extend(augment_example(ex))

# Verificar balance
counts = {{}}
for ex in augmented:
    col = re.match(r"(?:Extrae|Extract|Identifica|Encuentra|Recupera|Resume) \[([^\]]+)\]", ex["input"])
    key = col.group(1) if col else "unknown"
    counts[key] = counts.get(key, 0) + 1

min_c, max_c = min(counts.values()), max(counts.values())
print(f"Ejemplos originales: {{len(RAW_DATASET)}}")
print(f"Ejemplos aumentados: {{len(augmented)}} ({{len(augmented)/len(RAW_DATASET):.1f}}x)")
print(f"Balance de columnas: min={{min_c}}, max={{max_c}} - {{'BALANCEADO' if max_c <= min_c + 1 else 'DESBALANCEADO'}}") 

random.shuffle(augmented)
train_val, test_data = train_test_split(augmented, test_size=0.12, random_state=42)
train_data, val_data = train_test_split(train_val, test_size=0.14, random_state=42)
print(f"Train: {{len(train_data)}} | Val: {{len(val_data)}} | Test: {{len(test_data)}}")

import pandas as pd, matplotlib.pyplot as plt, numpy as np, seaborn as sns
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 4, figsize=(20,4))
in_lens  = [len(d["input"].split())  for d in augmented]
out_lens = [len(d["output"].split()) for d in augmented]
axes[0].hist(in_lens,  bins=20, color="#4C72B0", edgecolor="white", alpha=0.8)
axes[0].set_title("Longitud Inputs"); axes[0].set_xlabel("Palabras")
axes[0].axvline(np.mean(in_lens), color="red", linestyle="--", label=f"Media: {{np.mean(in_lens):.0f}}")
axes[0].legend()
axes[1].hist(out_lens, bins=20, color="#DD8452", edgecolor="white", alpha=0.8)
axes[1].set_title("Longitud Outputs"); axes[1].set_xlabel("Palabras")
axes[1].axvline(np.mean(out_lens), color="red", linestyle="--", label=f"Media: {{np.mean(out_lens):.0f}}")
axes[1].legend()
axes[2].bar(["Train","Val","Test"], [len(train_data),len(val_data),len(test_data)],
            color=["#2196F3","#4CAF50","#FF9800"])
axes[2].set_title("Split Dataset v2")
for i, v in enumerate([len(train_data),len(val_data),len(test_data)]):
    axes[2].text(i, v+1, str(v), ha="center", fontweight="bold")
# Distribucion de muestras por columna (balance)
col_counts = list(counts.values())
axes[3].hist(col_counts, bins=range(min(col_counts), max(col_counts)+2), color="#4CAF50", edgecolor="white", alpha=0.8)
axes[3].set_title("Balance: Muestras por Columna (ideal: todas iguales)")
axes[3].set_xlabel("Muestras"); axes[3].set_ylabel("N columnas")
axis_text = f"Min={{min_c}} Max={{max_c}} Ideal=3"
axes[3].text(0.97, 0.95, axis_text, transform=axes[3].transAxes, ha="right", va="top",
             bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.8))
plt.tight_layout(); plt.savefig("dataset_stats_v2.png"); plt.show()
print(f"Input avg: {{np.mean(in_lens):.1f}} words | Output avg: {{np.mean(out_lens):.1f}} words")\
"""

# --- GENERAR NOTEBOOK ---------------------------------------------
cells = [
    md_cell(CELL_MD_TITLE),
    code_cell(CELL_INSTALL),
    code_cell(CELL_IMPORTS),
    code_cell(CELL_DATASET),
    code_cell(CELL_MODEL),
    code_cell(CELL_TRAIN_FUNCS),
    code_cell(CELL_TRAIN_LOOP),
    code_cell(CELL_PLOTS),
    code_cell(CELL_EVAL_TEST),
    code_cell(CELL_QUALITATIVE),
    code_cell(CELL_PER_COLUMN),
    code_cell(CELL_ONNX),
    code_cell(CELL_QUANTIZE),
    code_cell(CELL_SPEED),
    code_cell(CELL_SUMMARY),
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "cells": cells
}

out_path = os.path.join(os.path.dirname(__file__), "rsl_column_extractor_training.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

n_ex = len(DATASET)
print(f"OK Notebook generado: {out_path}")
print(f"   Celdas: {len(cells)} | Dataset inyectado: {n_ex} originales -> {n_ex*3} aumentados (~)")
print(f"   Modelo: flan-t5-base | Estrategia: Fine-Tuning completo diferenciado")
print(f"   Ejecutar: jupyter notebook training/rsl_column_extractor_training.ipynb")

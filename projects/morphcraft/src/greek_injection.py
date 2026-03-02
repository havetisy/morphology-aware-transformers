
# ================= SWITCHABLE GREEK MORPHOLOGY INJECTION EXPERIMENTS (AI + UD) =================
#
# This runner supports multiple modeling strategies that can be enabled/disabled via RUN_METHODS.
# Only the paper methods are active by default.
#
# RUN_METHODS:
#   - multitask_heads
#   - feat_projector
#   - baseline_strict
#   - tag_tokens
#   - prompt_prefix
#
# Tasks (switchable via RUN_* flags below):
#   - AI Greek NOUNS (JSON)
#   - AI Greek VERBS (JSON)
#   - UD Greek NOUNS (GDT / GUD)
#   - UD Greek VERBS (GDT / GUD)
#
# To switch methods, edit:
#   RUN_METHODS = [...]


from transformers.modeling_outputs import SequenceClassifierOutput


import torch
import torch.nn as nn
from typing import List
from transformers import MBartForSequenceClassification

# -------------------------------- CONFIG --------------------------------
RUN_METHODS   = ["multitask_heads","feat_projector"]
# Choose tasks (toggle any subset):
RUN_AI_NOUNS  = True
RUN_AI_VERBS  = True
RUN_UD_NOUNS  = True
RUN_UD_VERBS  = True
RUN_UD_TREEBANKS = ["GDT", "GUD"]   # any subset, e.g. ["GDT"]

# AI JSON paths (set these)
AI_GREEK_NOUNS_JSON = "projects/morphcraft/datasets/greek/greek_nouns.json"
AI_GREEK_VERBS_JSON = "projects/morphcraft/datasets/greek/greek_verbs.json"

LANG_CODE   = "el_GR"
MODEL_NAME  = "facebook/mbart-large-50"
BASE_OUTDIR = "/content/runs_injection"

# shared knobs (match strict baselines)
SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
LOGGING_STEPS = 50
OVERSAMPLE_MIN, OVERSAMPLE_CAP = 8, 3

# UD download config
UD_VERSION  = "r2.16"
BASE_URLS   = {
    "GDT": f"https://github.com/UniversalDependencies/UD_Greek-GDT/raw/{UD_VERSION}",
    "GUD": f"https://github.com/UniversalDependencies/UD_Greek-GUD/raw/{UD_VERSION}",
}
PREFIXES    = {"GDT": "el_gdt-ud-", "GUD": "el_gud-ud-"}
SPLITS_TRY  = ["train","dev","test"]

# -------------------------- Torch sanity (Colab/Py3.12) -----------------
import sys, subprocess, os, importlib, math
def ensure_torch():
    try:
        import torch; _ = torch.__version__; return
    except Exception:
        pass
    wheel_idx = "https://download.pytorch.org/whl/cu121" if os.path.exists("/usr/local/cuda") else "https://download.pytorch.org/whl/cpu"
    subprocess.check_call([sys.executable,"-m","pip","install","-q","--upgrade","--no-cache-dir",
                           "torch==2.4.1","torchvision==0.19.1","torchaudio==2.4.1","--index-url",wheel_idx])
    importlib.invalidate_caches()
    import torch, platform
    print("✅ Torch:", torch.__version__, "| Python:", platform.python_version(), "| CUDA:", torch.cuda.is_available())
ensure_torch()

# ----------------------------- Installs --------------------------------
for pkg in ["pandas","numpy","scikit-learn","transformers>=4.41.0","accelerate>=0.30.0","conllu"]:
    try: __import__(pkg.split(">=")[0])
    except ImportError: subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])

# ------------------------------- Imports -------------------------------
import json, random, inspect, urllib.request
import numpy as np, pandas as pd, torch, torch.nn as nn
from typing import List, Dict, Any, Tuple
from conllu import parse_incr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (
    MBart50TokenizerFast, MBartForSequenceClassification, MBartModel,
    Trainer, TrainingArguments, DataCollatorWithPadding, set_seed as hf_set_seed
)
from transformers.modeling_outputs import SequenceClassifierOutput

# ------------------------------ Utilities ------------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    try:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception: pass
    hf_set_seed(seed)

def compose_bundle_from_dict(d: dict, order: List[str]) -> str:
    return "|".join(f"{k.capitalize()}={str(d.get(k,'NA'))}" for k in order)

def parse_bundle(b: str) -> Dict[str,str]:
    out={}
    for part in str(b).split("|"):
        if "=" in part:
            k,v = part.split("=",1); out[k.lower()] = v
    return out

def split_safe(df, label_col="_bundle"):
    class_counts = df[label_col].value_counts()
    can_strat = (len(class_counts)>1) and (class_counts.min()>=2)
    if can_strat:
        return train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df[label_col])
    print("⚠️ Not enough samples per class for stratified split; using random split.")
    return train_test_split(df, test_size=TEST_SIZE, random_state=SEED)

def oversample_light(train_df, label_col="_bundle"):
    counts = train_df[label_col].value_counts().to_dict()
    aug=[]
    for _, r in train_df.iterrows():
        c = counts[r[label_col]]
        reps = min(OVERSAMPLE_CAP, max(1, int(math.ceil(OVERSAMPLE_MIN / max(c,1)))))
        aug.extend([r]*reps)
    return pd.DataFrame(aug).sample(frac=1, random_state=SEED).reset_index(drop=True)

def enc_texts(tok, texts):
    return tok(texts, truncation=True, padding=True)

class DS(torch.utils.data.Dataset):
    def __init__(self, enc, labels, extra=None):
        self.enc,self.labels = enc, labels
        self.extra = extra
    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        item = {
            "input_ids": torch.tensor(self.enc["input_ids"][i], dtype=torch.long),
            "attention_mask": torch.tensor(self.enc["attention_mask"][i], dtype=torch.long),
            "labels": torch.tensor(int(self.labels[i]), dtype=torch.long),
        }
        if self.extra:
            for k, arr in self.extra.items():
                item[k] = torch.tensor(arr[i], dtype=torch.long)
        return item

def eval_and_save(test_df, y_pred_ids, le_bundle, feature_order, out_dir, method_name, extra_meta=None,
                  compose_from_features=None):
    if compose_from_features is not None:
        pred_bundle = compose_from_features
    else:
        id2label = {i:l for i,l in enumerate(le_bundle.classes_)}
        pred_bundle = [id2label[i] for i in y_pred_ids]

    gold_bundle = test_df["_bundle"].astype(str).tolist()
    le_metrics = LabelEncoder().fit(gold_bundle + pred_bundle)
    yt, yp = le_metrics.transform(gold_bundle), le_metrics.transform(pred_bundle)
    metrics = {
        "bundle_exact_match": float(accuracy_score(gold_bundle, pred_bundle)),
        "f1_macro": float(f1_score(yt, yp, average="macro")),
        "f1_micro": float(f1_score(yt, yp, average="micro")),
    }
    print(f"\n[{method_name}] FEATURE_ORDER:", feature_order)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # per-feature F1
    true_feats = [parse_bundle(b) for b in gold_bundle]
    pred_feats = [parse_bundle(b) for b in pred_bundle]
    per_feat = {}
    for k in feature_order:
        yt_k = [d.get(k) for d in true_feats]
        yp_k = [d.get(k) for d in pred_feats]
        try:
            per_feat[k] = float(f1_score(yt_k, yp_k, average="macro"))
        except Exception:
            per_feat[k] = 0.0
        print(f"\n=== {k.upper()} — {method_name} ===")
        print(classification_report(yt_k, yp_k, digits=3))

    os.makedirs(out_dir, exist_ok=True)
    meta = {"feature_order": feature_order, "method": method_name}
    if isinstance(extra_meta, dict): meta.update(extra_meta)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({**metrics, "per_feature_macro_f1": per_feat, **meta}, f, ensure_ascii=False, indent=2)

    pred_df = test_df.copy()
    pred_df["gold_bundle"] = gold_bundle
    pred_df["pred_bundle"] = pred_bundle
    # add per-feature cols
    pred_feats_parsed = [parse_bundle(b) for b in pred_bundle]
    for k in feature_order:
        pred_df[f"gold_{k}"] = [d.get(k) for d in true_feats]
        pred_df[f"pred_{k}"] = [d.get(k) for d in pred_feats_parsed]
    pred_df["correct_bundle"] = (pred_df["gold_bundle"] == pred_df["pred_bundle"])
    pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    print("Saved:", os.path.join(out_dir, "metrics.json"))
    print("Saved:", os.path.join(out_dir, "predictions.csv"))

def make_train_args(out_dir):
    kwargs = dict(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=LOGGING_STEPS,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())
    if "evaluation_strategy" in params: kwargs["evaluation_strategy"] = "no"
    if "save_strategy" in params:       kwargs["save_strategy"] = "no"
    if "save_total_limit" in params:    kwargs["save_total_limit"] = 0
    if "label_smoothing_factor" in params: kwargs["label_smoothing_factor"] = 0.0
    return TrainingArguments(**kwargs)

def make_weighted_trainer(model, class_weights, args, tok, train_dataset):
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))(logits, labels)
            return loss  # return ONLY the loss tensor
        def save_model(self, *a, **k): pass
        def _save_checkpoint(self, *a, **k): pass
        def save_state(self, *a, **k): pass
    kw = dict(model=model, args=args, train_dataset=train_dataset, eval_dataset=None,
              data_collator=DataCollatorWithPadding(tokenizer=tok))
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        kw["processing_class"] = tok
    else:
        kw["tokenizer"] = tok
    return WeightedTrainer(**kw)

# -------------------------- Method: FEAT PROJECTOR ----------------------
class ProjectorWrapper(nn.Module):
    """
    Feature-injection via a small MLP bias on top of the base classifier logits:
      logits' = logits + Proj( sum_i Emb_i(feat_i) )
    We let the Trainer compute the CE loss; this forward returns logits only.
    """
    def __init__(self, base_model: MBartForSequenceClassification,
                 feat_vocab_sizes: List[int], num_labels: int, hidden: int = 64):
        super().__init__()
        self.base = base_model
        self.embs = nn.ModuleList([nn.Embedding(v, hidden) for v in feat_vocab_sizes])
        self.proj = nn.Linear(hidden, num_labels)
        self.act  = nn.Tanh()
        self.drop = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, feat_ids=None, **kwargs):
        # never pass labels down to the base model (Trainer handles loss)
        kwargs.pop("labels", None)

        # base forward; return type varies by transformers version, but .logits is present
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits  # (B, num_labels)

        # add feature-derived bias if provided
        if feat_ids is not None and len(self.embs) > 0:
            # feat_ids: LongTensor [B, n_feats], each column indexes its own embedding
            pieces = [emb(feat_ids[:, i]) for i, emb in enumerate(self.embs)]  # list of (B, hidden)
            feat_vec = torch.stack(pieces, dim=0).sum(dim=0)                   # (B, hidden)
            bias = self.proj(self.drop(self.act(feat_vec)))                    # (B, num_labels)
            logits = logits + bias

        # return a vanilla classifier output (no hidden_states/attentions to avoid version quirks)
        return SequenceClassifierOutput(logits=logits)


# -------------------------- Method: MULTITASK HEADS ---------------------
class MultiTaskModel(nn.Module):
    """
    MBART encoder + per-feature linear heads. Loss = mean CE across features.
    Returns concatenated logits (we keep slice indices to split after).
    """
    def __init__(self, model_name, feature_sizes: Dict[str,int]):
        super().__init__()
        self.enc = MBartModel.from_pretrained(model_name)
        hidden = self.enc.config.d_model
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, v) for k,v in feature_sizes.items()})
        self._slices = {}
        start = 0
        for k,v in feature_sizes.items():
            self._slices[k] = (start, start+v); start += v

    def forward(self, input_ids=None, attention_mask=None, **labels_kw):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state  # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.))
        x = self.dropout(x)
        losses = []
        logits_list = []
        for k, head in self.heads.items():
            logits_k = head(x)
            logits_list.append(logits_k)
            lab_key = f"labels_{k}"
            if lab_key in labels_kw and labels_kw[lab_key] is not None:
                losses.append(nn.CrossEntropyLoss()(logits_k, labels_kw[lab_key]))
        logits = torch.cat(logits_list, dim=-1)
        loss = torch.stack(losses).mean() if losses else None
        return {"loss": loss, "logits": logits}

class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        if loss is None:
            raise RuntimeError("MultiTaskModel returned loss=None. Check that labels_* tensors are present.")
        return loss  # return ONLY the loss tensor
    def save_model(self, *a, **k): pass
    def _save_checkpoint(self, *a, **k): pass
    def save_state(self, *a, **k): pass

# ----------------------- AI JSON Loaders (Greek) -----------------------
def norm_bool(x):
    s = str(x).strip().lower()
    if s in {"true","1","yes"}: return "true"
    if s in {"false","0","no"}: return "false"
    return s if s else "false"

def load_ai_el_nouns(json_path) -> Tuple[pd.DataFrame, List[str]]:
    data = json.load(open(json_path, "r", encoding="utf-8"))
    PREF = ["case","number","gender","definiteness","declension_class","irregular","root_alternation","compound","animacy"]
    rows, present = [], set()
    for ex in data:
        ni = ex.get("noun_info", {})
        nis = [ni] if isinstance(ni, dict) else (ni if isinstance(ni, list) else [])
        sent = ex.get("sentence","") or (nis[0].get("example_sentence","") if nis else "")
        for n in nis:
            form = n.get("form") or n.get("noun")
            lemma = n.get("lemma") or n.get("base_form")
            root_alt = n.get("root_alternation")
            root_alt = "None" if (root_alt is None or str(root_alt).strip()=="") else str(root_alt)
            row = {
                "sentence": sent, "form": form, "lemma": lemma,
                "gender": n.get("gender"), "number": n.get("number"), "case": n.get("case"),
                "definiteness": n.get("definiteness") or "Ind",
                "declension_class": n.get("declension_class") or "NA",
                "irregular": norm_bool(n.get("irregular")),
                "root_alternation": root_alt,
                "compound": norm_bool(n.get("compound")),
                "animacy": n.get("animacy") or "Inanimate",
            }
            rows.append(row); present.update([k for k in row if k in PREF])
    feat_order = [k for k in PREF if k in present]
    df = pd.DataFrame(rows)
    assert len(df)>0 and feat_order, "No noun examples or features."
    return df, feat_order

def load_ai_el_verbs(json_path) -> Tuple[pd.DataFrame, List[str]]:
    data = json.load(open(json_path, "r", encoding="utf-8"))
    PREF = ["person","number","tense","aspect","mood","voice","polarity",
            "conjugation_class","irregular","root_alternation","auxiliary","diathesis"]
    rows, present = [], set()
    # flatten possible verb_info lists
    interm=[]
    for ex in data:
        if "verb_info" in ex:
            vi = ex["verb_info"]
            if isinstance(vi, list):
                for v in vi:
                    interm.append({**ex, **v})
            else:
                interm.append({**ex, **vi})
        else:
            interm.append(ex)
    for src in interm:
        row = {
            "sentence": src.get("sentence",""),
            "form": src.get("form") or src.get("word"),
            "lemma": src.get("lemma"),
            "person": str(src.get("person")) if src.get("person") is not None else "NA",
            "number": src.get("number"),
            "tense": src.get("tense"),
            "aspect": src.get("aspect"),
            "mood": src.get("mood"),
            "voice": src.get("voice"),
            "polarity": src.get("polarity"),
            "conjugation_class": src.get("conjugation_class") or "NA",
            "irregular": norm_bool(src.get("irregular")),
            "root_alternation": (src.get("root_alternation") if str(src.get("root_alternation")).strip() else "none"),
            "auxiliary": norm_bool(src.get("auxiliary")),
            "diathesis": src.get("diathesis") or "NA",
        }
        rows.append(row); present.update([k for k in row if k in PREF])
    feat_order = [k for k in PREF if k in present]
    df = pd.DataFrame(rows)
    assert len(df)>0 and feat_order, "No verb examples or features."
    return df, feat_order

# ----------------------- UD Download & Helpers -------------------------
def dl(url, to_path):
    os.makedirs(os.path.dirname(to_path), exist_ok=True)
    urllib.request.urlretrieve(url, to_path)

def download_merge_ud(treebank: str, merged_path: str):
    base = BASE_URLS[treebank]; pref = PREFIXES[treebank]
    paths = []
    for sp in SPLITS_TRY:
        url = f"{base}/{pref}{sp}.conllu"
        fp  = f"/content/{pref}{sp}.conllu"
        try:
            if not os.path.exists(fp):
                print("Fetching:", url); dl(url, fp)
            else:
                print("Found:", fp)
            paths.append(fp)
        except Exception as e:
            print(f"Skip {sp}: {e}")
    assert paths, f"No UD files downloaded for {treebank}."
    with open(merged_path, "wb") as out:
        for p in paths:
            with open(p, "rb") as f: out.write(f.read())
    print("Merged to:", merged_path)
    return merged_path

def sent_text_or_build(sentence) -> str:
    if getattr(sentence, "metadata", None) and "text" in sentence.metadata:
        return sentence.metadata["text"]
    return " ".join(tok["form"] for tok in sentence if isinstance(tok.get("id", None), int))

def parse_feats(feats: Any) -> Dict[str,str]:
    out = {}
    if isinstance(feats, dict):
        for k,v in feats.items():
            if isinstance(v, list): v = ",".join(map(str,v))
            out[k.lower()] = str(v)
    return out

# -------------------------- UD Loaders (Greek) -------------------------
def load_ud_el_nouns(treebank: str) -> Tuple[pd.DataFrame, List[str]]:
    merged_path = f"/content/el_ud_{treebank.lower()}_all.conllu"
    download_merge_ud(treebank, merged_path)
    rows = []
    with open(merged_path, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            text = sent_text_or_build(sent)
            for tok in sent:
                if not isinstance(tok, dict): continue
                if not isinstance(tok.get("id"), int): continue
                if (tok.get("upos") or tok.get("upostag")) != "NOUN": continue
                feats = parse_feats(tok.get("feats"))
                form  = tok.get("form"); lemma = tok.get("lemma") or ""
                row = {"sentence": text, "form": form, "lemma": lemma,
                       "case": feats.get("case","NA"), "number": feats.get("number","NA"), "gender": feats.get("gender","NA")}
                rows.append(row)
    df = pd.DataFrame(rows)
    assert len(df)>0, "No NOUN tokens."
    order = ["case","number","gender"]
    return df, order

def load_ud_el_verbs(treebank: str) -> Tuple[pd.DataFrame, List[str]]:
    merged_path = f"/content/el_ud_{treebank.lower()}_all.conllu"
    download_merge_ud(treebank, merged_path)
    PREF = ["person","number","tense","aspect","mood","voice","polarity"]
    rows, present = [], set()
    with open(merged_path, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            text = sent_text_or_build(sent)
            for tok in sent:
                if not isinstance(tok, dict): continue
                if not isinstance(tok.get("id"), int): continue
                if (tok.get("upos") or tok.get("upostag")) != "VERB": continue
                feats = parse_feats(tok.get("feats"))
                form  = tok.get("form"); lemma = tok.get("lemma") or ""
                row = {"sentence": text, "form": form, "lemma": lemma}
                for k in PREF:
                    val = feats.get(k, "NA")
                    row[k] = val
                    if val != "NA": present.add(k)
                rows.append(row)
    df = pd.DataFrame(rows)
    assert len(df)>0, "No VERB tokens."
    order = [k for k in PREF if k in present] or ["person","number","tense","aspect","mood","voice"]
    return df, order

# ------------------------- Text builders per method ----------------------
def build_text(method, form, sent, feat_dict, order):
    if method == "baseline_strict":
        return f"{form or ''} || {sent}"
    if method == "tag_tokens":
        tags = " ".join([f"[{k.upper()}={feat_dict.get(k,'NA')}]" for k in order])
        return f"{form or ''} {tags} || {sent}"
    if method == "prompt_prefix":
        prefix = "FEATS: " + " | ".join([f"{k.capitalize()}={feat_dict.get(k,'NA')}" for k in order])
        return f"{prefix} || {form or ''} || {sent}"
    # multitask & projector use baseline textual input (no leakage via text)
    return f"{form or ''} || {sent}"

# ------------------------------ Runner ----------------------------------
def to_2d_numpy_predictions(pred_output):
    """
    Convert Trainer.predict(...) output to a single 2-D numpy array (B, D)
    robustly across HF versions. Ignores label arrays and other 1-D outputs.
    """
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    # Try to get predictions from PredictOutput-like object
    logits = getattr(pred_output, "predictions", None)
    if logits is None:
        # some trainers return a tuple/list, first item is usually predictions
        try:
            logits = pred_output[0]
        except Exception:
            logits = pred_output

    # Case A: already a single array/tensor
    if isinstance(logits, (np.ndarray, torch.Tensor)):
        arr = _to_np(logits)
        # ensure 2-D
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr

    # Case B: nested lists/tuples (per-batch etc.)
    if isinstance(logits, (list, tuple)):
        flat = []
        def _flatten(seq):
            for y in seq:
                if isinstance(y, (list, tuple)):
                    _flatten(y)
                else:
                    flat.append(_to_np(y))
        _flatten(logits)

        # Keep only proper numpy arrays
        flat = [a for a in flat if isinstance(a, np.ndarray)]

        # Keep only 2-D arrays (likely logits). Drop 1-D (labels) & weird shapes.
        twod = []
        for a in flat:
            if a.ndim == 2:
                twod.append(a)
            elif a.ndim > 2:
                twod.append(a.reshape(a.shape[0], -1))
            # ignore 1-D arrays entirely

        if not twod:
            raise ValueError("No 2-D prediction arrays found in predict() output.")

        # Pick the dominant column size (most frequent D) to avoid mixing labels/others
        from collections import Counter
        col_counts = Counter([a.shape[1] for a in twod])
        dominant_D = col_counts.most_common(1)[0][0]
        twod = [a for a in twod if a.shape[1] == dominant_D]

        return np.concatenate(twod, axis=0)

    # Fallback: best effort
    arr = np.asarray(logits)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr



def run_single(task_name, df, FEATURE_ORDER, method, extra_meta=None):
    print(f"\n==== {task_name} — method: {method} ====\n")
    set_seed(SEED)

    # bundle label (always prepared for eval)
    df["_bundle"] = df[FEATURE_ORDER].astype(str).apply(lambda r: compose_bundle_from_dict(r.to_dict(), FEATURE_ORDER), axis=1)

    # build input_text per method
    texts = []
    for _, r in df.iterrows():
        featd = {k: r.get(k, "NA") for k in FEATURE_ORDER}
        texts.append(build_text(method, r.get("form"), r.get("sentence"), featd, FEATURE_ORDER))
    df["input_text"] = texts

    # split + oversample
    train_df, test_df = split_safe(df)
    train_df_bal = oversample_light(train_df)

    tok = MBart50TokenizerFast.from_pretrained(MODEL_NAME); tok.src_lang = LANG_CODE
    out_dir = os.path.join(BASE_OUTDIR, f"{task_name}__{method}")

    if method in {"baseline_strict","tag_tokens","prompt_prefix"}:
        le = LabelEncoder().fit(train_df_bal["_bundle"].tolist())
        y_train = le.transform(train_df_bal["_bundle"].tolist())
        enc_train = enc_texts(tok, train_df_bal["input_text"].tolist())
        enc_test  = enc_texts(tok, test_df["input_text"].tolist())
        model = MBartForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(le.classes_))

        counts_bal = np.bincount(y_train, minlength=len(le.classes_))
        weights = (len(y_train) / (len(counts_bal) * np.maximum(counts_bal,1))).astype(np.float32)
        class_weights = torch.tensor(weights)

        args = make_train_args(out_dir)
        train_ds = DS(enc_train, y_train)
        trainer = make_weighted_trainer(model, class_weights, args, tok, train_ds)
        trainer.train()

        eval_ds = DS(enc_test, np.zeros(len(test_df), dtype=int))
        pred = trainer.predict(eval_ds)
        preds = getattr(pred,"predictions",pred[0]); preds = preds[0] if isinstance(preds,tuple) else preds
        y_pred_ids = np.argmax(preds, axis=-1)
        eval_and_save(test_df, y_pred_ids, le, FEATURE_ORDER, out_dir, method_name=method, extra_meta=extra_meta)

    elif method == "feat_projector":
        # per-feature vocab maps (train-fit)
        feat_maps = {}
        for k in FEATURE_ORDER:
            vals = train_df_bal[k].astype(str).fillna("NA")
            uniq = sorted(vals.unique().tolist() + ["NA"])
            feat_maps[k] = {v:i for i,v in enumerate(uniq)}
        # ids arrays
        def map_feat_col(df_in, k):
            m = feat_maps[k]; return df_in[k].astype(str).map(lambda x: m.get(x, m["NA"])).fillna(m["NA"]).astype(int).values
        feat_ids_train = np.stack([map_feat_col(train_df_bal,k) for k in FEATURE_ORDER], axis=1) if FEATURE_ORDER else np.zeros((len(train_df_bal),0), dtype=int)
        feat_ids_test  = np.stack([map_feat_col(test_df,k) for k in FEATURE_ORDER],  axis=1) if FEATURE_ORDER else np.zeros((len(test_df),0), dtype=int)

        le = LabelEncoder().fit(train_df_bal["_bundle"].tolist())
        y_train = le.transform(train_df_bal["_bundle"].tolist())
        enc_train = enc_texts(tok, train_df_bal["input_text"].tolist())
        enc_test  = enc_texts(tok, test_df["input_text"].tolist())

        base = MBartForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(le.classes_))
        vocab_sizes = [len(feat_maps[k]) for k in FEATURE_ORDER]
        model = ProjectorWrapper(base, vocab_sizes, num_labels=len(le.classes_))

        counts_bal = np.bincount(y_train, minlength=len(le.classes_))
        weights = (len(y_train) / (len(counts_bal) * np.maximum(counts_bal,1))).astype(np.float32)
        class_weights = torch.tensor(weights)

        args = make_train_args(out_dir)
        train_ds = DS(enc_train, y_train, extra={"feat_ids": feat_ids_train})
        eval_ds  = DS(enc_test,  np.zeros(len(test_df), dtype=int), extra={"feat_ids": feat_ids_test})

        class ProjTrainer(Trainer):
            def compute_loss(self, model, inputs, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)  # wrapper returns SequenceClassifierOutput(logits=...)
                logits = outputs.logits
                loss = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))(logits, labels)
                return loss  # return ONLY the loss tensor
            def save_model(self,*a,**k): pass
            def _save_checkpoint(self,*a,**k): pass
            def save_state(self,*a,**k): pass
        kw = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=None,
                  data_collator=DataCollatorWithPadding(tokenizer=tok))
        if "processing_class" in inspect.signature(Trainer.__init__).parameters: kw["processing_class"]=tok
        else: kw["tokenizer"]=tok
        trainer = ProjTrainer(**kw)
        trainer.train()

        pred = trainer.predict(eval_ds)
        preds = getattr(pred,"predictions",pred[0]); preds = preds[0] if isinstance(preds,tuple) else preds
        y_pred_ids = np.argmax(preds, axis=-1)
        eval_and_save(test_df, y_pred_ids, le, FEATURE_ORDER, out_dir, method_name=method, extra_meta=extra_meta)

    elif method == "multitask_heads":
      if not FEATURE_ORDER:
          raise ValueError("No features available for multitask_heads.")

     # --- per-feature encoders ---
      feat_encoders, feat_sizes = {}, {}
      train_labels_dict, test_labels_dict = {}, {}
      for k in FEATURE_ORDER:
          le_k = LabelEncoder().fit(train_df_bal[k].astype(str).fillna("NA").tolist())
          feat_encoders[k] = le_k
          feat_sizes[k] = len(le_k.classes_)
          train_labels_dict[f"labels_{k}"] = le_k.transform(
              train_df_bal[k].astype(str).fillna("NA").tolist()
          ).astype(np.int64)
          # dummy eval labels (not used in loss during predict)
          test_labels_dict[f"labels_{k}"]  = np.zeros(len(test_df), dtype=np.int64)

    # --- tokenize ---
      enc_train = enc_texts(tok, train_df_bal["input_text"].tolist())
      enc_test  = enc_texts(tok, test_df["input_text"].tolist())

      train_ds = DS(enc_train, np.zeros(len(train_df_bal), dtype=np.int64), extra=train_labels_dict)
      eval_ds  = DS(enc_test,  np.zeros(len(test_df),      dtype=np.int64), extra=test_labels_dict)

      model = MultiTaskModel(MODEL_NAME, feat_sizes)
      args  = make_train_args(out_dir)

      kw = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=None,
              data_collator=DataCollatorWithPadding(tokenizer=tok))
      if "processing_class" in inspect.signature(Trainer.__init__).parameters:
          kw["processing_class"] = tok
      else:
          kw["tokenizer"] = tok
      trainer = MultiTaskTrainer(**kw)
      trainer.train()

    # --- predict ---
          # --- predict ---
      pred = trainer.predict(eval_ds)
      logits = to_2d_numpy_predictions(pred)   # << use the normalizer

    # --- split logits back per head ---
      slices, start = {}, 0
      for k, size in feat_sizes.items():
          slices[k] = (start, start + size); start += size

      pred_feats = {}
      for k, (s, e) in slices.items():
          pred_feats[k] = np.argmax(logits[:, s:e], axis=-1)

    # rebuild bundles from per-feature preds
      id2val = {k: {i: v for i, v in enumerate(feat_encoders[k].classes_)} for k in FEATURE_ORDER}
      pred_bundle = []
      for i in range(len(test_df)):
          d = {k: id2val[k][int(pred_feats[k][i])] for k in FEATURE_ORDER}
          pred_bundle.append(compose_bundle_from_dict(d, FEATURE_ORDER))

      le_dummy = LabelEncoder().fit(test_df["_bundle"])  # placeholder (we pass pred_bundle directly)
      eval_and_save(
          test_df, None, le_dummy, FEATURE_ORDER, out_dir,
          method_name=method, extra_meta=extra_meta, compose_from_features=pred_bundle
      )



# ------------------------------ RUN ALL ---------------------------------
set_seed(SEED)

packs = []

if RUN_AI_NOUNS:
    df_n, order_n = load_ai_el_nouns(AI_GREEK_NOUNS_JSON)
    packs.append(("ai_el_nouns", df_n, order_n, {}))

if RUN_AI_VERBS:
    df_v, order_v = load_ai_el_verbs(AI_GREEK_VERBS_JSON)
    packs.append(("ai_el_verbs", df_v, order_v, {}))

if RUN_UD_NOUNS:
    for tb in RUN_UD_TREEBANKS:
        df_un, order_un = load_ud_el_nouns(tb)
        packs.append((f"ud_{tb.lower()}_el_nouns", df_un, order_un, {"treebank": tb, "ud_version": UD_VERSION}))

if RUN_UD_VERBS:
    for tb in RUN_UD_TREEBANKS:
        df_uv, order_uv = load_ud_el_verbs(tb)
        packs.append((f"ud_{tb.lower()}_el_verbs", df_uv, order_uv, {"treebank": tb, "ud_version": UD_VERSION}))

for name, df, order, meta in packs:
    for m in RUN_METHODS:
        run_single(name, df.copy(), order, m, extra_meta=meta)

print("\n✅ All requested Greek injection runs (AI + UD) finished.")

import os, shutil
base = "/content/runs_injection"
assert os.path.isdir(base), f"Not found: {base}"
archive = shutil.make_archive("/content/runs_injection_el_2", "zip", base_dir=base)
from google.colab import files
files.download(archive)

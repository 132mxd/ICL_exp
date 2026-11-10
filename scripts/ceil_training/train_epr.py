"""官方 CEIL 的 EPR：top/bottom-5、批内负例、验证集最佳模型。"""
import argparse, json, math, random
from pathlib import Path
import torch
from transformers import AutoTokenizer
from .common import MeanEncoder, linear_warmup_decay, read_jsonl, split_train_validation, tokenize_texts
from ..evaluation.config import PROJECT_ROOT
from ..example_selectors.bm25_selector import input_to_text
from ..example_selectors.kate_selector import DEFAULT_KATE_ENCODER

def sampled_pair(row, rng, pos_k, neg_k):
    if row.get("singleton_score_type") != "mean_nll_lower_is_better":
        raise ValueError("Legacy singleton scores are incompatible; rerun score_singletons")
    ordered = sorted(row["singleton_scores"], key=row["singleton_scores"].get)
    lookup = {item["id"]: item for item in row["candidates"]}
    return lookup[rng.choice(ordered[:pos_k])], lookup[rng.choice(ordered[-neg_k:])]

def batch_loss(model, tokenizer, rows, rng, args, device):
    pairs = [sampled_pair(row, rng, args.positive_topk, args.negative_bottomk) for row in rows]
    queries = [input_to_text(row["input"]) for row in rows]
    contexts = [input_to_text(item["input"]) for pair in pairs for item in pair]
    q_vec = model(tokenize_texts(tokenizer, queries, args.max_length, device))
    # 官方 ctx_no_grad=true，共享 encoder 但 context 分支不反传。
    with torch.no_grad():
        ctx_vec = model(tokenize_texts(tokenizer, contexts, args.max_length, device))
    labels = torch.arange(len(rows), device=device) * 2
    return torch.nn.functional.cross_entropy(q_vec @ ctx_vec.T, labels)

def evaluate(model, tokenizer, rows, args, device):
    model.eval(); rng = random.Random(args.seed); losses = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            losses.append(float(batch_loss(model, tokenizer, rows[start:start+args.batch_size], rng, args, device)))
    model.train()
    return sum(losses) / max(1, len(losses))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--encoder", default=DEFAULT_KATE_ENCODER)
    p.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data/hf_cache/models")
    p.add_argument("--device", default="auto"); p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1); p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--positive-topk", type=int, default=5); p.add_argument("--negative-bottomk", type=int, default=5)
    p.add_argument("--max-length", type=int, default=128); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.positive_topk, args.negative_bottomk) <= 0: p.error("positive values required")
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    rows = read_jsonl(args.input)
    stratified = bool(rows and rows[0].get("dataset") == "banking77")
    train_rows, val_rows = split_train_validation(
        rows, args.validation_fraction, args.seed, stratify=stratified,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, cache_dir=str(args.cache_dir))
    model = MeanEncoder(args.encoder, cache_dir=args.cache_dir).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = math.ceil(len(train_rows)/args.batch_size) * args.epochs
    scheduler = linear_warmup_decay(optimizer, round(total_steps*args.warmup_ratio), total_steps)
    best = float("inf"); encoder_dir = args.output_dir / "encoder"; args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(train_rows); model.train(); losses=[]
        for start in range(0, len(train_rows), args.batch_size):
            loss=batch_loss(model, tokenizer, train_rows[start:start+args.batch_size], rng, args, device)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step(); losses.append(float(loss))
        val=evaluate(model, tokenizer, val_rows, args, device)
        if val < best:
            best=val; encoder_dir.mkdir(parents=True, exist_ok=True)
            model.backbone.save_pretrained(encoder_dir); tokenizer.save_pretrained(encoder_dir)
        print(f"EPR epoch {epoch+1}/{args.epochs}: train={sum(losses)/len(losses):.6f} validation={val:.6f} best={best:.6f}")
    metadata={"stage":"epr","implementation":"official-ceil-qwen-adapter-v1","encoder_source":args.encoder,
      "max_length":args.max_length,"training_queries":len(train_rows),"validation_queries":len(val_rows),
      "batch_size":args.batch_size,"positive_topk":args.positive_topk,"negative_bottomk":args.negative_bottomk,
      "best_validation_loss":best,"warmup_ratio":args.warmup_ratio,
      "validation_split":"stratified" if stratified else "random","seed":args.seed}
    (args.output_dir/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved best EPR checkpoint to {args.output_dir}")
if __name__ == "__main__": main()

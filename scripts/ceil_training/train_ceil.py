"""官方 CEIL：全部候选集合的 conditional-DPP margin ranking。"""
import argparse, json, math, random
from pathlib import Path
import torch
from transformers import AutoTokenizer
from .common import (MeanEncoder, linear_warmup_decay, official_dpp_logdets,
                     official_ranking_loss, read_jsonl, split_train_validation, tokenize_texts)
from ..example_selectors.bm25_selector import input_to_text

def row_scores(model,tokenizer,row,q_vec,args,device):
    if row.get("set_score_type")!="mean_nll_lower_is_better" or row.get("set_construction")!="official_cand_random":
        raise ValueError("Legacy set supervision is incompatible; rerun build_sets and score_sets")
    ordered=sorted(row["set_scores"],key=lambda x:x["score"])
    lookup={x["id"]:x for x in row["candidates"]}; ids=[]
    for item in ordered:
        for item_id in item["ids"]:
            if item_id not in ids: ids.append(item_id)
    texts=[input_to_text(lookup[item_id]["input"]) for item_id in ids]
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(texts),args.context_batch_size):
            chunks.append(model(tokenize_texts(tokenizer,texts[start:start+args.context_batch_size],args.max_length,device)))
    ctx=torch.cat(chunks); index={item_id:i for i,item_id in enumerate(ids)}
    set_indices=torch.tensor([[index[x] for x in item["ids"]] for item in ordered],device=device)
    return official_dpp_logdets(q_vec,ctx,set_indices,args.scale_factor)

def batch_loss(model,tokenizer,rows,args,device):
    q=model(tokenize_texts(tokenizer,[input_to_text(r["input"]) for r in rows],args.max_length,device))
    scores=torch.stack([row_scores(model,tokenizer,row,q[i],args,device) for i,row in enumerate(rows)])
    return official_ranking_loss(scores,args.margin)

def evaluate(model,tokenizer,rows,args,device):
    model.eval(); losses=[]
    with torch.no_grad():
        for start in range(0,len(rows),args.batch_size): losses.append(float(batch_loss(model,tokenizer,rows[start:start+args.batch_size],args,device)))
    model.train(); return sum(losses)/max(1,len(losses))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--epr-checkpoint",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--dataset",default="sst5"); p.add_argument("--teacher-model",required=True)
    p.add_argument("--device",default="auto"); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--context-batch-size",type=int,default=128); p.add_argument("--learning-rate",type=float,default=1e-5)
    p.add_argument("--warmup-ratio",type=float,default=.1); p.add_argument("--validation-fraction",type=float,default=.1)
    p.add_argument("--max-length",type=int,default=128); p.add_argument("--scale-factor",type=float,default=.1)
    p.add_argument("--margin",type=float,default=.02); p.add_argument("--candidate-pool-size",type=int,default=100); p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.context_batch_size,args.scale_factor,args.candidate_pool_size)<=0:p.error("positive values required")
    random.seed(args.seed);torch.manual_seed(args.seed)
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    rows=read_jsonl(args.input);stratified=bool(rows and rows[0].get("dataset")=="banking77")
    train_rows,val_rows=split_train_validation(rows,args.validation_fraction,args.seed,stratify=stratified)
    enc=args.epr_checkpoint/"encoder"; epr_meta=json.loads((args.epr_checkpoint/"metadata.json").read_text())
    if epr_meta.get("implementation")!="official-ceil-qwen-adapter-v1":raise ValueError("Legacy EPR checkpoint; retrain EPR")
    tokenizer=AutoTokenizer.from_pretrained(enc);model=MeanEncoder(str(enc)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    total=math.ceil(len(train_rows)/args.batch_size)*args.epochs;scheduler=linear_warmup_decay(optimizer,round(total*args.warmup_ratio),total)
    best=float("inf");out_enc=args.output_dir/"encoder";args.output_dir.mkdir(parents=True,exist_ok=True);rng=random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(train_rows);model.train();losses=[]
        for start in range(0,len(train_rows),args.batch_size):
            loss=batch_loss(model,tokenizer,train_rows[start:start+args.batch_size],args,device)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();scheduler.step();losses.append(float(loss))
        val=evaluate(model,tokenizer,val_rows,args,device)
        if val<best:
            best=val;out_enc.mkdir(parents=True,exist_ok=True);model.backbone.save_pretrained(out_enc);tokenizer.save_pretrained(out_enc)
        print(f"CEIL epoch {epoch+1}/{args.epochs}: train={sum(losses)/len(losses):.6f} validation={val:.6f} best={best:.6f}")
    training_set_size=int(rows[0]["num_shots"]);candidate_sets=len(rows[0]["set_scores"])
    metadata={"method":"ceil","implementation":"official-ceil-qwen-adapter-v1","dataset":args.dataset,"teacher_model":args.teacher_model,
      "training_set_size":training_set_size,"max_length":args.max_length,"scale_factor":args.scale_factor,"margin":args.margin,
      "candidate_pool_size":args.candidate_pool_size,"candidate_sets":candidate_sets,"batch_size":args.batch_size,
      "training_queries":len(train_rows),"validation_queries":len(val_rows),"best_validation_loss":best,"warmup_ratio":args.warmup_ratio,
      "validation_split":"stratified" if stratified else "random","seed":args.seed}
    (args.output_dir/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved best official CEIL checkpoint to {args.output_dir}")
if __name__=="__main__":main()

"""官方 cand_random：EPR top-100，top-16 + 49 个不同随机集合。"""
import argparse, json, random
from pathlib import Path
import torch
from transformers import AutoTokenizer
from .common import MeanEncoder, read_jsonl, tokenize_texts, write_jsonl
from ..example_selectors.bm25_selector import input_to_text

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epr-checkpoint",type=Path,required=True); p.add_argument("--device",default="auto")
    p.add_argument("--num-shots",type=int,default=16); p.add_argument("--sets-per-query",type=int,default=50)
    p.add_argument("--dpp-topk",type=int,default=100); p.add_argument("--max-length",type=int,default=128); p.add_argument("--seed",type=int,default=43)
    args=p.parse_args()
    if args.num_shots<=0 or args.sets_per_query<1 or args.dpp_topk<args.num_shots: p.error("invalid set sizes")
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    enc_dir=args.epr_checkpoint/"encoder"; meta=json.loads((args.epr_checkpoint/"metadata.json").read_text())
    if meta.get("implementation")!="official-ceil-qwen-adapter-v1": raise ValueError("EPR checkpoint is legacy; retrain EPR first")
    tokenizer=AutoTokenizer.from_pretrained(enc_dir); model=MeanEncoder(str(enc_dir)).to(device).eval()
    rng=random.Random(args.seed); rows=read_jsonl(args.input)
    with torch.inference_mode():
      for row in rows:
        if len(row["candidates"])<args.dpp_topk: raise ValueError(f"Need >= {args.dpp_topk} candidates per query")
        texts=[input_to_text(row["input"])]+[input_to_text(x["input"]) for x in row["candidates"]]
        vectors=[]
        for start in range(0,len(texts),64): vectors.append(model(tokenize_texts(tokenizer,texts[start:start+64],args.max_length,device)).cpu())
        vectors=torch.cat(vectors); scores=vectors[1:]@vectors[0]
        ranked=[row["candidates"][i]["id"] for i in torch.argsort(scores,descending=True).tolist()[:args.dpp_topk]]
        sets=[ranked[:args.num_shots]]; seen={tuple(range(args.num_shots))}
        while len(sets)<args.sets_per_query:
            indices=tuple(sorted(rng.sample(range(args.dpp_topk),args.num_shots)))
            if indices not in seen: seen.add(indices); sets.append([ranked[i] for i in indices])
        row["candidate_sets"]=sets; row["num_shots"]=args.num_shots
        row["set_construction"]="official_cand_random"; row["dpp_topk"]=args.dpp_topk
    write_jsonl(args.output,rows); print(f"Wrote {args.sets_per_query} official candidate sets for {len(rows)} queries to {args.output}")
if __name__=="__main__": main()

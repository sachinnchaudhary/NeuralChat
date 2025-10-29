"""This is for basic evals where we compare the base, sft and rlhf model answer's"""

"!pip install sentencepiece pandas numpy"

from google.colab import drive
import os
drive.mount("/content/drive")
import sys
sys.path.append("/content/drive/MyDrive/Colab Notebooks/NeuralChat")
from model import MiniGPT

from init import setup_env
ROOT, DEVICE = setup_env()

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter,   defaultdict
import re
import math
from datasets import load_dataset
import sentencepiece as spm 
import pandas as pd
import numpy as np  

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(ROOT, "tokenizer", "neuralchat_sp.model"))   

#loading models.......................................................

def load_model(name):  
     
    m = MiniGPT(vocab_size=sp.get_piece_size()).to(DEVICE)
    ckpt = os.path.join(ROOT, "models", name)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval()
    return m

models = {
    "Pretrain": load_model("minigpt_wiki.pt"),
    "SFT": load_model("neuralchat_sft/sft.pt"),
    "RLHF": load_model("neuralchat_rlhf.pt"),
}

print("All evaluation models loaded")  

#generating responds......................................................

import torch.nn.functional as F
@torch.no_grad()
def generate(model, prompt, max_new=60, temp=0.8):
    ids = torch.tensor([sp.encode(prompt, out_type=int)], device=DEVICE)
    for _ in range(max_new):
        logits, _ = model(ids)
        logits = logits[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_tok], dim=1)
        if next_tok.item() == sp.eos_id():
            break
    return sp.decode(ids[0].tolist())
  
#evaluative prompt.........................................................

prompts = [
    "Explain gravity in one sentence.",
    "Why do humans play games?",
    "What is deep learning?",
    "Books are good because?",
    "Summarize photosynthesis briefly."
]  

records = []
for p in prompts:
    print(f"\n🔹 Prompt: {p}")
    for name, m in models.items():
        out = generate(m, f"<|user|> {p} <|assistant|>", max_new=80)
        print(f"{name}: {out}\n")
        records.append({"prompt": p, "model": name, "output": out})
df = pd.DataFrame(records)
df.to_csv(os.path.join(ROOT, "human_eval_outputs.csv"), index=False)
print("Generated responses saved for manual scoring")


#rating answers..............................................................
for i, row in df.iterrows():
    print("\nPrompt:", row.prompt)
    print("Model:", row.model)
    print("Output:", row.output)
    f = int(input("Fluency (1-5): "))
    h = int(input("Helpfulness (1-5): "))
    a = int(input("Alignment (1-5): "))
    df.loc[i, ["fluency","helpfulness","alignment"]] = [f,h,a]

df.to_csv(os.path.join(ROOT, "human_eval_scored.csv"), index=False)
print("✅ Saved scored CSV to Drive.")

#basic visulization code.....................................................

import pandas as pd, os
ROOT = "/content/drive/MyDrive/Colab Notebooks/NeuralChat"

df = pd.read_csv(os.path.join(ROOT, "human_eval_scored.csv"))
scores = df.groupby("model")[["fluency","helpfulness","alignment"]].mean().round(2)
print(scores)
scores.plot.bar(title="Human Evaluation Results (↑ better)")

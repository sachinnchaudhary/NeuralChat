""""This file contain code for pretraining of LLM with writing transformer from scratch."""


from google.colab import drive
import shutil, os

import sys, importlib.util, os

setup_path = "/content/drive/MyDrive/Colab Notebooks/NeuralChat/setup/init.py"
spec = importlib.util.spec_from_file_location("init", setup_path)
init_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(init_module)
ROOT, DEVICE = init_module.setup_env()



"""drive.flush_and_unmount()
drive.mount("/content/drive")
import sys
sys.path.append("/content/drive/MyDrive/Colab Notebooks/NeuralChat/setup")

from init import setup_env
ROOT, DEVICE = setup_env()"""


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter,   defaultdict
import re
import math
from datasets import load_dataset
import sentencepiece as sp

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
train_data = ds["train"]
test_data = ds["test"]
validation_data = ds["validation"]

train_text_path = os.path.join(ROOT, "data", "sp_train.txt")

#saving in drive..................................................
os.makedirs(os.path.dirname(train_text_path), exist_ok=True)

with open(train_text_path, "w", encoding="utf-8") as f:

     for line in ds["train"]["text"]:
        if line.strip():
            f.write(line.strip() + "\n")
print(f"Saved training text to: {train_text_path}") 

tok_dir = os.path.join(ROOT, "tokenizer")
os.makedirs(tok_dir, exist_ok=True)

model_prefix = os.path.join(tok_dir, "neuralchat_sp")

sp.SentencePieceTrainer.train(
    input=train_text_path,
    model_prefix=model_prefix,
    vocab_size=32000,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=["<|user|>", "<|assistant|>", "<|system|>"]
)


#now we can get token ids and text.
sentpiece  =  sp.SentencePieceProcessor()
sentpiece.load(os.path.join(ROOT, "tokenizer", "neuralchat_sp.model"))

ids = sentpiece.encode("Hello, NeuralChat is learning!", out_type=int)
print(ids)

de = sentpiece.decode(ids)
print(de)


#sentencepiece bpe tokenizer...................................................

sentpiece  =  sp.SentencePieceProcessor()
sentpiece.load(os.path.join(ROOT, "tokenizer", "neuralchat_sp.model"))
class ChatDataset(Dataset):

   def __init__(self, data, tokenizer, max_len=512):
        # save only non-empty lines
        self.texts = [t for t in data["text"] if t.strip()]
        self.tokenizer = tokenizer
        self.max_len = max_len

   def __len__(self):
        return len(self.texts)

   def __getitem__(self, idx):

      text = self.texts[idx]
      ids = self.tokenizer.encode(text, out_type=int)[:self.max_len]
      pad_id = self.tokenizer.pad_id() if hasattr(self.tokenizer, 'pad_ids') else 0
      if len(ids) < self.max_len:
        ids = ids + [pad_id] * (self.max_len - len(ids))

train_dataset = ChatDataset(train_data, sentpiece)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


#transformer architecture for next token prediction...........................

class MultiHeadattention(nn.Module):

    def __init__(self, n_embed, n_head):

       super().__init__()
       assert n_embed % n_head == 0
       self.n_head = n_head
       self.head_dim = n_embed // n_head
       self.qkv =  nn.Linear(n_embed, 3 * n_embed)
       self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x, mask = None):

      B, T, C = x.shape
      qkv = self.qkv(x).view(B,T,3,self.n_head,self.head_dim).transpose(1,3)
      q,k,v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
      att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
      if mask is not None: att = att.masked_fill(mask==0, -1e9)
      att = F.softmax(att, dim=-1)
      out = (att @ v).transpose(1,2).contiguous().view(B,T,C)
      return self.proj(out)

class Feedforward(nn.Module):

   def __init__(self, n_embd):
     super().__init__()
     self.net = nn.Sequential(
         nn.Linear(n_embd, 4 * n_embd),
         nn.GELU(),
         nn.Linear(4 * n_embd, n_embd)
     )

   def forward(self, x): return self.net(x)


class Block(nn.Module):

   def __init__(self, n_embd, n_head):
      super().__init__()
      self.attn = MultiHeadattention(n_embd, n_head)
      self.ff = Feedforward(n_embd)
      self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

   def forward(self, x, mask):

    x = x + self.attn(self.ln1(x), mask)
    x = x + self.ff(self.ln2(x))
    return x

class MiniGPT(nn.Module):

   def __init__(self, vocab_size, n_layer=6, n_head=8, n_embd=512, block_size=512):
     super().__init__()
     self.tok_emb = nn.Embedding(vocab_size, n_embd)
     self.pos_emb = nn.Embedding(block_size, n_embd)
     self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
     self.ln_f = nn.LayerNorm(n_embd)
     self.head = nn.Linear(n_embd, vocab_size, bias=False)
     self.block_size = block_size

   def forward(self, idx, targets = None):

       B, T = idx.shape
       pos = torch.arange(0, T, device=idx.device)
       x = self.tok_emb(idx) + self.pos_emb(pos)
       mask = torch.tril(torch.ones(T,T,device=idx.device)).unsqueeze(0).unsqueeze(0)
       for blk in self.blocks: x = blk(x, mask)
       logits = self.head(self.ln_f(x))
       loss = None
       if targets is not None:
          loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
       return logits, loss


import numpy as np

def encode(txt): return sentpiece.encode(txt, out_type=int)
def decode(ids): return sentpiece.decode(ids)

text = "\n".join([t for t in train_data["text"] if t.strip()])

ids = np.array(encode(text), dtype=np.int32)
block_size, batch_size = 256, 8

def get_batch(split="train"):
    ix = np.random.randint(0, len(ids)-block_size-1, (batch_size,))
    x = [ids[i:i+block_size] for i in ix]
    y = [ids[i+1:i+block_size+1] for i in ix]
    return torch.tensor(x,device=DEVICE), torch.tensor(y,device=DEVICE)

CUDA_LAUNCH_BLOCKING=1
DEVICE = "cpu"
model = MiniGPT(vocab_size=sentpiece.get_piece_size()).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
steps, eval_interval = 2000, 200

for step in range(steps):
    xb,yb = get_batch()
    xb, yb = xb.long(), yb.long()
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step%eval_interval==0:
        print(f"Step {step} | Loss {loss.item():.3f}")

ckpt_path = f"{ROOT}/checkpoints"
os.makedirs(ckpt_path, exist_ok=True)
torch.save(model.state_dict(), f"{ckpt_path}/minigpt_wiki.pt")

model.eval()
context = torch.tensor([encode("The meaning of life is")], device=DEVICE)
for _ in range(200):
    logits,_ = model(context[:, -512])
    next_token = torch.multinomial(F.softmax(logits[0,-1]/0.7,dim=-1),1)
    context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
print(decode(context[0].tolist()))

with torch.no_grad():
    xb,yb = get_batch()
    xb, yb = xb.long(), yb.long()
    _, val_loss = model(xb, yb)
print("Validation loss:", val_loss.item())

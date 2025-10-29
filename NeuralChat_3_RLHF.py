"""This file is about doing RLHF using PPO"""

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


#loading dataset..............................

ds = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")
"""print(ds[0])--->  printing the data"""


#building reward model........................

class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_dim=512):
        super().__init__()
        self.base = base_model
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        _, hidden, _ = self.base(input_ids, return_hidden=True)
        pooled = hidden.mean(dim=1)       
        reward = self.value_head(pooled)   
        return reward 

#preparing tokenizer and loading data.................

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(ROOT, "tokenizer", "neuralchat_sp.model"))


def encode_text(text):  
  ids = sp.encode(text, out_type= int)[:512]
  return torch.tensor(ids, dtype=torch.long)

def collate_fn(batch):   
    chosen = [encode_text(x["chosen"]) for x in batch]
    rejected = [encode_text(x["rejected"]) for x in batch]
    
    return {"chosen": torch.nn.utils.rnn.pad_sequence(chosen, batch_first=True),
            "rejected": torch.nn.utils.rnn.pad_sequence(rejected, batch_first=True)}

#training loop......................................................

model_base = MiniGPT(vocab_size=sp.get_piece_size())  
reward_model = RewardModel(model_base).to(DEVICE)
optimizer = torch.optim.AdamW(reward_model.parameters(), lr = 1e-5)

loader= DataLoader(ds, batch_size =2,shuffle=True, collate_fn=collate_fn)

for step, batch in enumerate(loader):  

   chosen, rejected = batch["chosen"].to(DEVICE), batch["rejected"].to(DEVICE)
   r_c = reward_model(chosen)
   r_r = reward_model(rejected)
   loss = -torch.mean(torch.log(torch.sigmoid(r_c - r_r)))   
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   if step % 100 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")   

torch.save(reward_model.state_dict(), os.path.join(ROOT, "models", "reward_model.pt"))
print("Reward model saved.")


#loading the model............

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(ROOT, "tokenizer", "neuralchat_sp.model"))

# --- rebuild same architecture ---
base = MiniGPT(vocab_size=sp.get_piece_size())

# --- define RewardModel class exactly as before ---
class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_dim=512):
        super().__init__()
        self.base = base_model
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        # ask MiniGPT to return its hidden states
        _, hidden, _ = self.base(input_ids, return_hidden=True)
        # hidden: [B, T, n_embd=512]
        pooled = hidden.mean(dim=1)         
        reward = self.value_head(pooled)  
        return reward

# --- load weights you already saved ---
reward_model = RewardModel(base).to(DEVICE)
reward_model.load_state_dict(torch.load(os.path.join(ROOT, "models", "reward_model.pt"), map_location=DEVICE))
reward_model.eval()
print("Loaded reward model from checkpoint")  


#PPO training............................................................

from torch import optim
import torch.nn.functional as F

# Build / load policy model
policy_model = MiniGPT(vocab_size=sp.get_piece_size()).to(DEVICE)
print(" Policy model initialized for PPO training")

# Optional: load from your SFT checkpoint
# policy_model.load_state_dict(torch.load(os.path.join(ROOT, "models", "neuralchat_sft.pt"), map_location=DEVICE))

def sample_next_token(logits, temperature=1.0, top_k=40):
    logits = logits / temperature
    vals, idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
    probs = F.softmax(vals, dim=-1)
    next_tok = idx[torch.multinomial(probs, 1)]
    return next_tok.item()

@torch.no_grad()
def generate_response(model, sp, prompt, max_new_tokens=64, temperature=0.8):
    model.eval()
    ids = torch.tensor([sp.encode(prompt, out_type=int)], device=DEVICE)
    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        next_token = sample_next_token(logits[0, -1], temperature)
        ids = torch.cat([ids, torch.tensor([[next_token]], device=DEVICE)], dim=1)
        if next_token == sp.eos_id():
            break
    return ids[0].tolist()

def compute_logprobs(logits, actions):
    logprobs = F.log_softmax(logits, dim=-1)
    return logprobs.gather(2, actions.unsqueeze(-1)).squeeze(-1)

def ppo_loss(ratio, advantage, clip_range=0.2):
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantage
    return -torch.mean(torch.min(unclipped, clipped))

optimizer = optim.Adam(policy_model.parameters(), lr=1e-5)
eps_clip = 0.2
prompts = [
    "<|user|> books are good ? <|assistant|>",
    "<|user|> what is LLM ? <|assistant|>",
    "<|user|> why do humans play games ? <|assistant|>"
]

for epoch in range(3):
    total_loss, total_reward = 0, 0
    for prompt in prompts:
        # 1. rollout --------------------------------------------------------
        token_ids = generate_response(policy_model, sp, prompt)
        input_ids = torch.tensor([token_ids[:-1]], device=DEVICE)
        actions = torch.tensor([token_ids[1:]], device=DEVICE)
        logits, _ = policy_model(input_ids)
        old_logprobs = compute_logprobs(logits, actions).detach()

        # 2. reward ---------------------------------------------------------
        with torch.no_grad():
            reward = reward_model(torch.tensor([token_ids], device=DEVICE))
        total_reward += reward.mean().item()

        # 3. advantage (≈ reward)
        advantage = reward.detach()

        # 4. policy update --------------------------------------------------
        logits_new, _ = policy_model(input_ids)
        new_logprobs = compute_logprobs(logits_new, actions)
        ratio = torch.exp(new_logprobs - old_logprobs)
        loss = ppo_loss(ratio, advantage, clip_range=eps_clip)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | avg reward {total_reward/len(prompts):.3f} | loss {total_loss/len(prompts):.3f}")

# Save the PPO-tuned model
save_path = os.path.join(ROOT, "models", "neuralchat_rlhf.pt")
torch.save(policy_model.state_dict(), save_path)
print(" PPO fine-tuned model saved:", save_path)

# Quick test
prompt = "<|user|> why do humans play games ? <|assistant|>"
out_ids = generate_response(policy_model, sp, prompt, max_new_tokens=80)
print("Output:", sp.decode(out_ids))

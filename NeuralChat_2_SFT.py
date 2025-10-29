"""This file contain the SFT(supervised fine tuning) of pre-trained model."""


from google.colab import drive
import os
drive.mount("/content/drive")
import sys
sys.path.append("/content/drive/MyDrive/Colab Notebooks/NeuralChat/setup")
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

#loading pre-trained model..............................................

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(ROOT, "tokenizer", "neuralchat_sp.model"))

vocab_size = sp.get_piece_size()
model = MiniGPT(vocab_size=vocab_size)

ckpt_path = os.path.join(ROOT, "checkpoints", "minigpt_wiki.pt")
model.load_state_dict(torch.load(ckpt_path, map_location= "cpu"))
model.to(DEVICE)

print("Loaded pretrained model from:", ckpt_path)


#SFTing the base model...................................................

data = load_dataset("yahma/alpaca-cleaned", split= "train[:2000]")
"""print(data[0])  ---- > for data checking"""

def format_example(example):

  if example["input"]:
       prompt = f"<|user|> {example['instruction']}\n{example['input']}\n<|assistant|> {example['output']}"
  else:
       prompt = f"<|user|> {example['instruction']}\n<|assistant|> {example['output']}"

  ids = sp.encode(prompt, out_type =int)
  return {"input_ids": ids, "len": len(ids)}

dataset = data.map(format_example, remove_columns= data.column_names)
dataset = dataset.filter(lambda x: x["len"] <= model.block_size)

def collate_fn(batch):
    max_len = max(x["len"] for x in batch)
    pad_id = 0
    input_ids = [x["input_ids"] + [pad_id]*(max_len - x["len"]) for x in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(DEVICE)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


from transformers import Trainer, TrainingArguments

train_args = TrainingArguments(
    output_dir=os.path.join(ROOT, "sft_out"),
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    num_train_epochs=1,
    fp16=True,
    logging_steps=20,
    save_steps=200,
    overwrite_output_dir=True,
    remove_unused_columns=False,
)


trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

trainer.train()

save_path = os.path.join(ROOT, "models", "neuralchat_sft")
os.makedirs(save_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_path, "sft.pt"))
print("SFT model saved to:", save_path)

#evaluating the SFT model....................................................

model.eval()
prompt = "<|user|> why game is fun to play ? <|assistant|>"
tokens = sp.encode(prompt, out_type=int)
context = torch.tensor([tokens], dtype=torch.long, device=DEVICE)


for _ in range(120):
  logits, _ = model(input_ids=context)
  next_token = torch.multinomial(
        torch.softmax(logits[0,-1]/0.7, dim=-1), 1
    )
  context = torch.cat([context, next_token.unsqueeze(0)], dim=1)

print(sp.decode(context[0].tolist()))

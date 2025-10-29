# NeuralChat  
**Zero-rupee cost LLM for understanding the end-to-end process of building, fine-tuning, and aligning a language model using free colab resource.**

**Note: The focus of building NueralChat is to understand the process which happens when you train this LLM from pre-training to RLHF using PPO, So you can understand all aspects of it. 
Additionally this entire project is built on Colab free tier resource using T4-gpu with cost of 0 bucks. So having 0 buck budget we not get optimized answer from LLM and it produces giberrish answer. 
But it's totally possible to get decent coherent answer with increasing compute on this same code. and i will eventually try this same code once i'll have enough compute :)**
---

## Overview
**NeuralChat** is a lightweight, educational LLM project designed to help students, researchers, and enthusiasts understand the **complete lifecycle of training and aligning a transformer-based chatbot** — all using free resources like Google Colab and Hugging Face Spaces.

This repository reproduces the full journey from scratch:
1.  **Pretraining** — learn next-token prediction on WikiText-2.  
2.  **Supervised Fine-Tuning (SFT)** — train on instruction-response datasets (e.g., Alpaca).  
3.  **Reinforcement Learning with Human Feedback (RLHF)** — apply PPO with a learned reward model.  
4.  **Human Evaluation** — compare model quality across stages.  
5.  **Inference / Deployment** — deploy on Hugging Face Spaces via Gradio.

All steps are implemented in modular Python scripts so you can reproduce and extend every phase independently.



---

## ⚙️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/sachinnchaudhary/NeuralChat.git
cd NeuralChat
pip install -r requirements.txt


# Overview

Scale: 小规模的模型和大规模的模型在很多方面不同
- MLP 在计算中的占比随着规模增大而增大，而 attn 占比下降
- 涌现现象

The bitter lesson: Scale 很重要，但是 algorithm 也很重要。**Algorithm that scale is what matters**.

语言模型 landscape：
- Pre-neural: 香农测量英语的熵、N-gram
- Neural ingredients:
    - LSTM: 1997
    - 首个 neural language model: Bengio et al. 2003
    - seq2seq
    - Adam
    - Attention
    - Transformer
    - MoE
    - Model parallelism GPipe, ZeRO
- 早期基座模型：ELMo, BERT, T5. 已经出现 Pretrain-finetune 范式
- 拥抱 scaling: OpenAI 的 GPT-2 (1.5B), GPT-3 (175B)，Google 的 PaLM (540B)


## Syllabus

- 基础
    - Tokenization: 
        - BPE
        - tokenizer-free approaches: 直接使用 raw bytes
    - Architecture: Transformer 基础上的改进
        - 激活函数：ReLU, SwiGLU
        - PE: sin, RoPE
        - norm: LayerNorm, RMSNorm; norm 的位置：pre-norm vs. post-norm
        - Attn: full, sparse/local attention, group-query attention (GQA), multi-head latent attention (MLA)
        - MLP：dense, MoE
        - Recurrence/state-space models/linear attention: Mamba, Gated DeltaNet
    - Training
        - Optimizer: AdamW, Muon, SOAP
        - Loss function: multi-token prediction, DeepSeek V3 tech report
        - LR schedule: cosize, WSD
        - Initialization scale: Xavier init, muP
        - Batchsize: critical batch size
        - Regularization: dropout, weight decay
        - MoE specific: load balancing (e.g., aux-free)
- Systems
    - Kernels
    - Parallelism: 当我们有成千上万块 GPU 时
    - Inference: Prefill + decoding
        - 加速推理的方法：Model pruning, quantization, disstillation
        - Speculative decoding
- Scaling laws: 训练超大规模的模型通常只有一次机会
    - 考虑一个 scaling recipe (FLOPS -> hyperparameters), 在小规模模型上做实验, 拟合一个 scaling law, 用它来预测大规模模型的超参数
    - Scaling laws 不会自动出现，需要细心地构建 scaling recipe, 从而实现超参数的 transfer（大规模模型的超参数要么就是小规模模型的，要么是小规模模型超参数的函数）——可预测性很重要
    - 给定一个 FLOPS 预算，训练更大的模型还是在更多数据上训练？
        - Training Compute-Optimal Large Language Models
        - TL;DR: $D = 20 N$ is roughly optimal (e.g., 70B parameter model should be trained on ~1.4T tokens)
- Data
    - Evaluation
        - 内部评: 指引模型开发（smoothness across scale, 相对指标更重要）
        - 外部评：向外界展示模型能力
    - Data curation
    - Data processing
        - Transformation：将 HTML/PDF 转成纯文本
        - Filtering: 保留高质量数据，去掉垃圾/有害数据
        - Dedup：Bloom filter, MinHash
        - Mixing: 哪些数据源权重更高？
        - Rewriting / synthetic data: 用 LM 增强真实数据
    - Types of data
        - 预训练：大规模、多样
        - 中训练（mid-training）：高质量，包括长上下文
        - 后训练（post-training）：监督式微调（对话、带 tool calling 的 agentic traces）
- Alignment
    - Weak supervision
        - 基本模板：让模型生成回答 -> 人类（或其他语言模型等）对回答质量评分 -> 更新模型以使它能偏好更好的回答
    - 算法：强化学习
        - Proximal Policy Optimization (PPO)
        - Direct Policy Optimization (DPO)
        - Group Relative Preference Optimization (GPRO)
    - 挑战：
        - 强化学习算法不稳定，难以调参
        - 大规模时，需要很多 infra 相关的优化（inference with async rollouts）
        - 在 efficiency 和 on-policyness 之间不断 trade-off

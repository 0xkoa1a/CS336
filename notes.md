- [Stanford CS336: Language Modeling from Scratch](#stanford-cs336-language-modeling-from-scratch)
    - [Lec 01](#lec-01)
    - [Lec 05](#lec-05)
    - [Lec 06](#lec-06)



# Stanford CS336: Language Modeling from Scratch

## Lec 01

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


课程大纲
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


## Lec 05

CPU 重控制逻辑（latency），GPU 重计算（throughput）。

GPU 解剖——执行：
- 流式多处理器（Streaming Multiprocessor）：
    - 内部有多个流处理器（Streaming Processor）。SP 又叫 CUDA core，是最基本的计算单元，相当于 CPU 的核心
    - 调度器：以线程束（warp）为单位调度
    - 寄存器文件：每个线程私有
    - 共享内存（shared memory）：线程块（block）内的线程共享

GPU 的执行模型：SIMT
- **线程**：只处理一个数据点
    - 每个线程有独立的 PC、寄存器、私有数据
- **线程束**（warp）：一组并行执行的线程，一般是 32 个。
    - SM 内的 warp 调度器以 warp 为单位调度
    - warp 内的所有线程执行相同的指令流
    - Warp Divergence：当一个 warp 遇到条件分支时，它会串行化不同的执行路径，通过 mask 屏蔽不需要的结果
- **线程块**（thread block）：一组线程的集合
    - **一个 Block 会被调度到一个 SM 上执行**。此操作由 GPU 的硬件调度器完成。
        - 这种分配是 in wave 的，最后一个 wave 的 block 往往更少，导致低利用率
        - Wave quantization：让 block 数是 SM 数量的整数倍
        - rule of thumb: #thread blocks >= 4x #SMs
    - 同一个块内的线程可以通过**共享内存**通信
    - 块内的线程可以**同步**
    - 一个 Block 不会跨 SM 执行，否则无法使用 shared mem 和 block 内同步
    - 一个 SM 可以并发执行多个 Block
    - Block 中的线程，在硬件层面会被自动、确定地划分为 warp，这对程序员是不可见的

CUDA 的编程模型：
- Kernel：GPU 上运行的函数，由 `__global__` 关键字声明。
- 线程：执行 kernel 代码的最小单位，有唯一的 thread ID
- 线程块（thread block）: 线程的集合，有唯一的 block ID
- 网格（Grid）：线程块的集合。
    - 一个 kernel launch 会启动一个网格
    - 一个网格会在整个 GPU 上运行
    - 同一 grid 内的不同线程块之间不能直接通信或同步

软硬件模型的映射：
- Grid <-> GPU device
- Block <-> SM
- Thread <-> CUDA core

Triton 的编程模型：
- **Program Instance**: **基本执行单位**，相当于 CUDA 的线程块
    - 每个线程块负责一个 tile
    - 编译器自动管理低于块的细节：shared memory、线程块内同步、DRAM coalescing、边界情况
- **Grid of Program Instances**: 相当于 CUDA 的 grid

GPU 解剖——存储：
- 寄存器：SM 内部，每个线程私有
- **Shared Memory**：SM 内部，block 内的线程共享。程序员可以显式控制
- L1 Cache：SM 内部，硬件自动管理，是 DRAM 的缓存。
- L2 Cache：GPU 芯片内，但在所有 SM 之外，所有 SM 共享。由硬件自动管理，是 DRAM 的缓存
- **Global Memory** (DRAM)：GPU 芯片外，GPU 上所有线程以及 CPU 都可以访问。
    - **Memory Coalescing**：buy 1 get 3 free

CUDA 的存储模型：
- 寄存器/局部内存：kernel 函数内部声明定义的，没有其他存储限定符的变量
    - 寄存器放不下就会放到全局内存
- 共享内存：`__shared__` 关键字声明的变量
    - 程序员需要显式将数据从全局内存拷贝到共享内存，并管理共享内存在块内的并发访问
- 全局内存：`__device__` 关键字声明的变量，或者通过 `cudaMalloc` 分配的内存
    - 所有线程都可以访问
    - 程序员需要显式合并内存访问
- 常量内存：`__constant__` 关键字声明的变量
    - 只读，所有线程都可以访问

Triton 的存储模型：
- 只需要考虑 DRAM 就行了，其下的细节由编译器自动管理

TPU：
- 轻控制、快的矩阵乘法单元、快的存储
- 与 GPU 的不同：
    - how the accelerators are networked
    - no warps, 只有 blocks（trade-off between matmul vs. non-matmul）

GPU 模型的强大之处：
- 容易 scale-up：加 SM
- 编程模型简单：SIMT
- 线程非常轻，调度开销小

计算 scale 的速度比存储快得多——存储更可能成为瓶颈。

**Roofline model**: perf = min(compute roof, memory roof)
- 开始线性增长（memory bound），之后变平（thruput bound）
- 我们的目标是尽可能往右上的屋顶线走，这样才能最大化 gpu util%。
- 类比：网络 delivery rate 和 amount inflight 的关系
    - 路由器们的 queue 充满之前，delivery rate 关于 amount inflight 线性增长，latency 不变
    - queue 充满之后，delivery rate 饱和不再增长。此时 latency 开始随着 amount inflight 增加而线性增加
    - 拥塞控制算法的理想工作点：折线的拐点处——带宽饱和，延迟最低

让 GPU 更快：
- control divergence（并非存储瓶颈）
    - SIMT：warp 内的所有线程必须执行相同的指令流
    - 条件分支：一部分线程执行 if 分支，同时另一部分线程休眠，然后再反过来
    - 条件分支开销很大
- 存和算的 trade-off：
    - 低精度计算
    - Recomputation
- 减少存储访问次数
    - Operator Fusion
    - Coalescing
- 让存储发生在 shared mem 而非 global mem
    - Tiling

低精度计算优化算术强度（arithmetic intensity）
- ReLU on a vector of size n
- FP32
    - 存储：1 read(8B), 1 write(8B) if x < 0
    - 计算：1 FLOP（比较）
    - Intensity（存算比）：8B / 1 FLOP
- FP16
    - Intensity: 4B / 1 FLOP
    - 仿佛内存带宽提升了一倍
- FP16/BP16 存 matmul 结果、大部分 pointwise 操作（relu, tanh, +, -, *）
- FP32/FP16 
    - 存需要更高精度的值：累加值、reduction operations（sum, softmax, norm）
    - 存需要更大范围的值：exp, log, pow、loss func

Operator fusion
- 糟糕的情况：很多无意义的 load/store！存储开销非常大
    - load A from global mem
    - compute B = f(A)
    - store B to global mem
    - load B from global mem
    - compute C = g(B)
    - store C to global mem
    - ...
- 理想的情况
    - load A from global mem
    - compute B = f(A)
    - compute C = g(B)
    - ...
    - store final result to global mem
- 简单的融合可以由编译器自动完成：torch.compile

Recomputation
- 既然存储比计算贵，那还存什么，每次用的时候重新计算一遍就好了
- 不但时间变快了，占用的空间也更小了 
- 例子
    - Forward pass：不存中间结果，只 load 输入和 store 最终输出
    - Backward pass：load 输入和 loss，重新跑 fwd pass 计算中间结果并 BP，store 最终梯度
    - 有多少中间结果，就能减少多少个 load 和 store

Memory coalescing and DRAM
- DRAM (global memory) 是以 burst mode 读取的
- 地址空间被分成很多 burst section，每次读一整个 burst section
- buy 1 get x free
- 因此可以利用空间局部性，将一个 warp 的内存访问的位置尽量安排在同一个 burst section 里，硬件就能自动 group 这些访问，减少 DRAM 访问次数，提高带宽。
- 例如：
    ```py
    # 矩阵中，同一行的元素更可能在同一个/相邻的 burst section，不同行的元素一般不在
    # 如果 thread 1 和 thread 2 负责处理不同的行，即每次它们都访问同一列的不同行
    # 那么这两个线程的访问每次都会命中不同的 burst section，无法 coalesce
    for r in rows:
        thread = get_thread_for_row(r)
        for c in cols:
            thread.process(data[r, c])    
    # 如果 thread 1 和 thread 2 负责处理不同的列，即每次它们都访问同一行的不同列
    # 那么这两个线程的访问每次都可能命中相同的 burst section，从而能 coalesce
    for c in cols:
        thread = get_thread_for_col(c)
        for r in rows:
            thread.process(data[r, c])
    ```

Tiling
- 矩阵乘法为例：
    - 积矩阵的每个元素等于因子的横行乘纵列，横行访问无法 coalesce
    - 不同的线程会访问相同的元素，导致重复的 load
    - 解决方法：将矩阵划分为多个 tile
        - load M(0, 0) and N(0, 0) into SHM
        - compute partial sum for P
        - Load M(0, 0) and N(2, 0) into SHM
        - ...
    - 重复的 load 现在访问 SHM，而非 global mem
    - 内存访问能够被 coalesce（一个 tile 的内容处于同一个 burst section 里）
    - non-tiled matmul：每个输入从 global mem 被 load N 次（N 是矩阵维度）
    - tiled matmul：每个输入从 global mem 被 load N/T 次，从 SHM 被 load T 次（T = #tile） 
- 麻烦：
    - 需要小心选取 tile size：
        - 不能太大，否则不好合并内存访问，且 SHM 空间有限
        - 要使得矩阵被均匀分割
    - 内存对齐
        - 理想情况：一个四行的 tile，每一行都刚好是一个 burst section
        - 不理想的情况：一个四行的 tile，第一个刚好是一个 burst section，其他行跨两个 burst section
        - 如果 矩阵维度或者 tile size 不是 burst section 的整数倍，就容易出现不理想情况
        - 解决方法：padding 以使得矩阵维度是 burst section 的整数倍
        - 显著影响性能！

Matrix mystery:
- **Tiling alignment**：显著影响性能
- **Wave quantization**：
    - 假设 N = 1792, tile size = (256, 128)
    - 则一共有 $\frac{1792}{256} \times \frac{1792}{128} = 7 \times 14 = 98$ 个 tile
    - 当 $N = 1793$ 时，tile 数变为 $8 \times 15 = 120$ 个——多了很多个 tile！
        - 多出来的这些 tile 非常 sparse，利用率会很差
    - Even worse，A100 只有 108 个 SM，一次性装不下这么多 tile
        - 还需要再多一遍 pass，利用率更差

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 softmax 操作是对每一行进行的（attn matrix 的每一行的行和都是 1）。然而，按行访问的存储效率不好。

$$
\text{Softmax}(X_{ij}) = \frac{e^{X_{ij}}}{\sum_{k} e^{X_{ik}}}
$$

为了避免 overflow，注意到

$$
\text{Softmax}(X_{ij}) = \frac{C\cdot e^{X_{ij}}}{\sum_{k} C\cdot e^{X_{ik}}} = \frac{e^{X_{ij} + \log C}}{\sum_{k} e^{X_{ik} + \log C}}
$$

对任意非零常数 $C$ 成立，我们有 Safe Softmax：

$$
\text{Softmax}(X_{ij}) = \frac{e^{X_{ij} - m_i}}{\sum_{k} e^{X_{ik} - m_i}}
$$

然而 Safe Softmax 需要三次 pass：一次找最大值 $m_i$，一次计算分母，一次计算每个 softmax 值。

Online Softmax 是 one-pass 的 Safe Softmax：
- 维护当前最大值 $m_{ij}$ 和归一化分母 $d_{ij}$：
    - $m_{ij} = \max(m_{i,j-1}, X_{ij})$
    - $d_{ij} = d_{i,j-1} \cdot e^{m_{i,j-1} - m_{ij}} + e^{X_{ij} - m_{ij}}$

Fwd Pass of flash-attn:
- Tile-wise attn
- Fusion of exp operator
- Tile-wise softmax (online softmax)

Backward Pass: 
- Recomputation of any $n^2$-sized values

## Lec 06

benchmark/profile 很重要。想象中的瓶颈不一定是瓶颈。
- benchmarking 测量 wall-clock time
    - 先 warm-up，再测多次取平均
    - `torch.cuda.synchronize()` 确保 CPU 和 GPU 同步
- profiling 测量各个操作的时间
    - warm-up 和 `torch.cuda.synchronize()` 同上
    - `torch.profiler`，`with torch.profiler.profile(...) as prof: ...`
    - NVIDIA Nsight Systems
        - 在代码中添加 annotation: with nvtx.range("my_op"): ...
        - MLP 例子：首先是加载库、on the fly compile 之类的初始化
        - CPU 向 GPU 发送 CUDA kernel。CPU 会维护一个队列/窗口。一般来说，CPU 的进度总是领先于 GPU
        - 假如在每次迭代间打印 loss，则 CPU 必须等待 GPU 计算出 loss，然后打印，然后才能继续发下一个 kernel
        - 如果 print 太多，可能使 GPU 一直在等 CPU，造成效率问题

实现 GeLU:
- 调 pytorch
- 用 pytorch 写
- 用 CUDA 写
- 用 triton 写
- 用 pytorch 写，然后 torch.compile

```py
def create_cuda_gelu():
    # CUDA is an extension of C/C++ with APIs for managing GPUs.
    # Simplified picture: write f(i), CUDA kernel computes f(i) for all i.
    
    # Grid: collection of thread blocks: numBlocks = (2, 4), blockDim = (1, 8)
    # Thread block: collection of threads: blockIdx = (0, 1)
    # Thread: single unit of operation: threadIdx = (0, 3).
    # You write code that a thread execute, using (blockIdx, blockDim, threadIdx) to determine what to do.
    # Set CUDA_LAUNCH_BLOCKING so that if there are errors, CUDA will tell you what went wrong.
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # The load_inline function makes it convenient to write CUDA code and bind it to a Python module for immediate use.
    # CUDA code: has the full logic
    cuda_gelu_src = open("gelu.cu").read()
```

```c++
#include <c10/cuda/CUDAException.h>
#include <math.h>
#include <torch/extension.h>

__global void gelu_kernel(float *in, float *out, int num_elements) {
    // Get the index into the tensor
    // 第几个块 * 每个块的线程数 + 块内的第几个线程
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) { // To handle the case when n < numBlocks * blockDim
        // Do the actual computation
        out[i] = 0.5 * in[i] *
                 (1.0 + tanh(0.79788456 *
                             (in[i] + 0.044715 * in[i] * in[i] * in[i])));
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    // Compute ceil(a / b)
    return (a + b - 1) / b;
}

torch::Tensor gelu(torch::Tensor x) {
    // 必要的检查
    TORCH_CHECK(x.device().is_cuda());
    TORCH_CHECK(x.is_contiguous());
    // Allocate empty tensor
    torch::Tensor y = torch::empty_like(x);    // 分配输出张量, 不需要调用 torch.zeros, 因为反正会写入
    // Determine grid (elements divided into blocks)
    int num_elements = x.numel();
    int block_size = 1024;  // Number of threads
    int num_blocks = cdiv(num_elements, block_size);
    // Launch the kernel
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();  // Catch errors immediately
    return y;
}
```

```py
    # C++ code: defines the gelu function
    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"
    # Compile the CUDA code and bind it to a Python module.
    ensure_directory_exists("var/cuda_gelu")
    if not torch.cuda.is_available():
        return None
    module = load_inline(
        cuda_sources=[cuda_gelu_src],
        cpp_sources=[cpp_gelu_src],
        functions=["gelu"],
        extra_cflags=["-O2"],
        verbose=True,
        name="inline_gelu",
        build_directory="var/cuda_gelu",
    )
    cuda_gelu = getattr(module, "gelu")
    return cuda_gelu
```

manual_gelu 比 cuda_gelu 慢很多，并不是因为 CPU-GPU 通信开销，而主要是因为 GPU 内 global memory 的访问开销（因为 CUDA 实现是 operator fused 的）

Triton:    
- 考虑线程块，而不是线程
- 自动进行 DRAM coalescing（burst mode 读取）
- 自动管理 shared memory
- 自动在 SM 之内调度线程块
- 仍然需要手动在 SM 之间调度线程块

```py
def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()
    # Allocate output tensor
    y = torch.empty_like(x)
    # Determine grid (elements divided into blocks)
    num_elements = x.numel()
    block_size = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, block_size)
    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)
    return y

@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input is at `x_ptr` and output is at `y_ptr`
    #     |        Block 0            |          Block 1          |      ...      |
    #                            BLOCK_SIZE                                 num_elements
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    # Indices where this thread block should operate
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Handle boundary
    mask = offsets < num_elements
    # Read
    x = tl.load(x_ptr + offsets, mask=mask)
    # Approx gelu is 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Compute (tl.tanh doesn't exist, use tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)
    # Store
    tl.store(y_ptr + offsets, y, mask=mask)
```

编程模型：
- CUDA: Grid -> Block -> Thread
    - BlockIdx.x 是块的编号
    - BlockIdx.x * BlockDim.x + ThreadIdx.x 是线程的编号
- Triton: Grid -> Program
    - tl.program_id(axis=0) 是块的编号
    - tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) 是块内所有线程的编号
- 主要的区别就是 CUDA 操作标量，Triton 操作向量。

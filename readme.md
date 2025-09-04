# 🧠 LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking

<p>
  <a href="https://arxiv.org/abs/2508.03440">
    <img src="https://img.shields.io/badge/arXiv-2508.03440-b31b1b.svg?style=flat" alt="arXiv">
  </a>

<a href="https://huggingface.co/papers/2508.03440">
    <img src="https://img.shields.io/badge/HuggingFace-Paper-orange.svg?style=flat" alt="Hugging Face Papers">
  </a>
</p>

This is the official implementation of the paper: [LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking](https://arxiv.org/abs/2508.03440)

## 🛠️ Re-development
If you would like to build on top of this project, refer to `sglang_soft_thinking_pkg/README.md`, or review the differences from SGLang v0.4.6.post1 in `sglang_soft_thinking_pkg/diff_0.4.6.post1.txt`.

## 🎲 Soft Thinking with Random Perturbation

This is a forked version of [Soft-Thinking](https://github.com/eric-ai-lab/Soft-Thinking), which includes support for Dirichlet and Gumbel-Softmax noise in Soft Thinking sampling. For more details, see `scripts/st/qwq32b_gumble.sh`.

Relevant parameters:
```bash
--add_noise_gumbel_softmax \
--gumbel_softmax_temperature 0.5
--add_noise_dirichlet \
--dirichlet_temperature 1.0 \
```

<!-- ## 📂 Directory Structure

```plaintext
soft_thinking/
├── datasets/
│   ├── aime2024.json
│   └── ... (other datasets)
├── models/
│   └── download.py
├── scripts/
│   ├── baseline/
│   └── st/
├── sglang_soft_thinking_pkg/
│   └── (sglang files)
├── configure.sh
├── codeeval.py
├── convert_livecodebench.py
├── humanevaleval.py
├── mbppeval.py
├── matheval.py
├── run_sglang_softthinking.py
├── run_sglang_nothinking.py
└── ... (other files)
``` -->

## ⚙️ Environment Setup

To set up the virtual environment for SGLang Soft Thinking inference, execute each line in `configure.sh`:

```bash
conda create -n st python=3.11 -y && conda activate st
pip install --upgrade pip
pip install torch transformers accelerate jsonlines math_verify openai torch_memory_saver
pip install flash_attn --no-build-isolation # may take more time (20min). try `pip install flash_attn==2.7.3 --no-build-isolation` if find undefined symbol bug

# Install SGLang (0.4.6.post1) tailored for Soft Thinking
cd sglang_soft_thinking_pkg
pip install -e "python[all]"
cd ..
```

### 🐳 Docker
We find it hard to reproduce some results across different devices due to precision issues. We recommend installing the environment with Docker by following `docker.sh`:

```bash
# For Docker
cd Soft-Thinking
docker build -t soft-thinking:st-cu124-py311 .
# NVIDIA Container Toolkit is required
docker run --gpus all --ipc=host --rm -it \
  -v $PWD:/workspace \
  soft-thinking:st-cu124-py311 bash
```

## 🚀 Quick Start

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/soft_thinking.git
    cd soft_thinking
    ```
2. **Set up the environment**:
   Follow the [Environment Setup](#environment-setup) instructions (Docker is recommended).
3. **Run a baseline test**:
    ```bash
    bash scripts/baseline/qwq32b.sh
    ```
   

## 🔄 Reproduction Instructions

### ⚖️ 1. LLM Judge
**Use your own OpenAI key in each script.**
```bash
export OPENAI_API_KEY=""
```
We use `gpt-4.1-2025-04-14` as the LLM judge.

### 🧪 2. Baseline

Run the baseline script:

```bash
bash scripts/baseline/qwq32b.sh
```

#### 📥 Download the Model

First, download the model to the `models/` directory:

```bash
python ./models/download.py --model_name "Qwen/QwQ-32B"
```

#### 🧠 Run Inference

Then, run the baseline inference:

```bash
export OPENAI_API_KEY=""
python run_sglang_softthinking.py \
    --dataset "aime2024" \
    --model_name "./models/Qwen/QwQ-32B" \ # you can use Qwen/QwQ-32B without downloading to ./models
    --max_topk 10 \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.0 \
    --after_thinking_temperature 0.6 \
    --after_thinking_top_p 0.95 \
    --after_thinking_top_k 30 \
    --after_thinking_min_p 0.0 \
    --early_stopping_entropy_threshold 0.0 \
    --early_stopping_length_threshold 256 \
    --mem_fraction_static 0.8 \
    --start_idx 0 \
    --end_idx 100000 \
    --num_gpus 8 \
    --num_samples 16 \
    --use_llm_judge \
    --judge_model_name "gpt-4.1-2025-04-14" 
```


---

### 🧠 3. Soft Thinking

Run the Soft Thinking script:

```bash
bash scripts/st/qwq32b_gumble.sh
```

When running **coding benchmarks (HumanEval, MBPP, and LiveCodeBench)**, start by executing without the `--reeval` flag. Then, run it again with the `--reeval` flag for evaluation. This is due to a multiprocessing bug.

> **Note:**
>
> - <span style="color:red;">Results may vary across different devices even with the same hyperparameters, due to differences in computation precision. We use NVIDIA H100 GPUs for all experiments. We recommend using Docker for reproduction.</span>


## 🪪 Licensing
This project utilizes a modified version of the [SGLang](https://github.com/sgl-project/sglang) library. The licensing structure is as follows:
- **Our Original Code**: The code original to this project (i.e. all code outside the `./sglang_soft_thinking_pkg` directory) is licensed under the **MIT License**. A copy of the MIT License can be found in the root `LICENCE` file.

- **Modified SGLang**: The code within the `./sglang_soft_thinking_pkg` directory is a derivative work of `SGLang` (version 0.4.6.post1) and is therefore licensed under **Apache License 2.0**. The orginal Apache 2.0 license is included in the `./sglang_soft_thinking_pkg/LICENSE` file. We have provide a `changes_0.4.6.post1.diff` file in that directory to show our modifications.





## 📜 Citation

If you use this code or dataset, please cite these papers:

```bibtex
@article{zhang2025soft,
  title={Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space},
  author={Zhang, Zhen and He, Xuehai and Yan, Weixiang and Shen, Ao and Zhao, Chenyang and Wang, Shuohang and Shen, Yelong and Wang, Xin Eric},
  journal={arXiv preprint arXiv:2505.15778},
  year={2025}
}

@article{wu2025llms,
  title={LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking},
  author={Wu, Ch{\"u}nhung and Lu, Jinliang and Ren, Zixuan and Hu, Gangqiang and Wu, Zhi and Dai, Dai and Wu, Hua},
  journal={arXiv e-prints},
  pages={arXiv--2508},
  year={2025}
}
```

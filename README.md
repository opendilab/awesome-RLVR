# Awesome RLVR — Reinforcement Learning with **Verifiable** Rewards

[![Stars](https://img.shields.io/github/stars/opendilab/awesome-RLVR?style=flat-square&logo=github)](https://github.com/opendilab/awesome-RLVR/stargazers)
[![Forks](https://img.shields.io/github/forks/opendilab/awesome-RLVR?style=flat-square&logo=github)](https://github.com/opendilab/awesome-RLVR/network/members)
[![Contributors](https://img.shields.io/github/contributors/opendilab/awesome-RLVR?style=flat-square&logo=github)](https://github.com/opendilab/awesome-RLVR/graphs/contributors)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](./LICENSE)

> A curated collection of surveys, tutorials, codebases and papers on  
> **Reinforcement Learning with Verifiable Rewards (RLVR)**—  
> a rapidly emerging paradigm that aligns both LLMs *and* other agents through  
> **objective, externally verifiable signals.**
<!-- > external verification, self-consistency and iterative self-improvement. -->

<!-- <p align="center">
  <img
    src="./assets/rft.png"
    style="border-radius:0.3125em;
           box-shadow:0 2px 4px rgba(34,36,38,.12),
                      0 2px 10px rgba(34,36,38,.08);" />
  <br/>
  <em>
    Core techniques powering reasoning LLMs, where verifiable rewards are pivotal.  
    (Figure taken from <a href="https://arxiv.org/pdf/2502.17419">
    “From System 1 to System 2: A Survey of Reasoning Large Language Models”</a>)
  </em>
</p> -->

<p align="center">
  <img
    src="./assets/rlvr.png"
    style="border-radius:0.3125em;
           box-shadow:0 2px 4px rgba(34,36,38,.12),
                      0 2px 10px rgba(34,36,38,.08);" />
  <br/>
  <em>
    An overview of how Reinforcement Learning with Verifiable Rewards (RLVR) works.  
    (Figure taken from <a href="https://arxiv.org/pdf/2411.15124">
    “Tülu 3: Pushing Frontiers in
Open Language Model Post-Training”</a>)
  </em>
</p>

## Why RLVR?

RLVR couples reinforcement learning with **objective, externally verifiable signals**, yielding a training paradigm that is simultaneously powerful and trustworthy:

* **Ground-truth rewards** – unit tests, formal proofs, or fact-checkers provide binary, tamper-proof feedback.  
* **Intrinsic safety & auditability** – every reward can be traced back to a transparent verifier run, simplifying debugging and compliance.  
* **Strong generalization** – models trained on verifiable objectives tend to extrapolate to unseen tasks with minimal extra data.  
* **Emergent “aha-moments”** – sparse, high-precision rewards encourage systematic exploration that often yields sudden surges in capability when the correct strategy is discovered.  
* **Self-bootstrapping improvement** – the agent can iteratively refine or even generate new verifiers, compounding its own learning signal.  
* **Domain-agnostic applicability** – the same recipe works for code generation, theorem proving, robotics, games, and more.

## How does it work?

1. **Sampling.** We draw one or more candidate completions \( {a}_{1..k} \) from a policy model \( \pi_\theta \) given a prompt \( s \).  
2. **Verification.** A deterministic function \( r(s,{a}) \) checks each completion for correctness.  
3. **Rewarding.**  
   • If a completion is verifiably correct, it receives a reward \( r = \gamma \).  
   • Otherwise the reward is \( r = 0 \).  
4. **Policy update.** Using the rewards, we update the policy parameters via RL (e.g., PPO).  
5. **(Optional) Verifier refinement.** The verifier itself can be trained, hardened, or expanded to cover new edge cases.

Through repeated iterations of this loop, the policy learns to maximise the externally verifiable reward while maintaining a clear audit trail for every decision it makes.

---

Pull requests are welcome 🎉 — see [Contributing](#contributing) for guidelines.


<pre>
[2025-07-03] <b>New!</b> Initial public release of Awesome-RLVR 🎉
</pre>


## Table of Contents

- [Awesome RLVR — Reinforcement Learning with **Verifiable** Rewards](#awesome-rlvr--reinforcement-learning-with-verifiable-rewards)
  - [Why RLVR?](#why-rlvr)
  - [How does it work?](#how-does-it-work)
  - [Table of Contents](#table-of-contents)
  - [Surveys \& Tutorials](#surveys--tutorials)
  - [Codebases](#codebases)
  - [Papers](#papers)
    - [2025](#2025)
      - [NeurIPS 2025](#neurips-2025)
      - [ICML 2025](#icml-2025)
      - [Other 2025 Papers](#other-2025-papers)
    - [2024 \& Earlier](#2024--earlier)
  - [Other Awesome Lists](#other-awesome-lists)
  - [Contributing](#contributing)
  - [License](#license)


```
format:
- [title](paper link) (presentation type)
  - main authors or main affiliations
  - Key: key problems and insights
  - ExpEnv: experiment environments
```

## Surveys & Tutorials

<details open>
  <summary>Click to expand / collapse</summary>

- [Inference-Time Techniques for LLM Reasoning](https://rdi.berkeley.edu/adv-llm-agents/slides/inference_time_techniques_lecture_sp25.pdf) (Berkeley Lecture 2025)  
  - DeepMind & UC Berkeley (Xinyun Chen)  
  - Key: decoding-time search, self-consistency, verifier pipelines  
  - ExpEnv: code/math reasoning benchmarks  

- [Learning to Self-Improve & Reason with LLMs](https://rdi.berkeley.edu/adv-llm-agents/slides/Jason-Weston-Reasoning-Alignment-Berkeley-Talk.pdf) (Berkeley Talk 2025)  
  - Meta AI & NYU (Jason Weston)  
  - Key: continual self-improvement loops, alignment interplay  
  - ExpEnv: open-ended dialogue & retrieval tasks
  
- [LLM Reasoning: Key Ideas and Limitations](https://llm-class.github.io/slides/Denny_Zhou.pdf) (Tutorial Slides 2024)  
  - DeepMind (Denny Zhou)  
  - Key: theoretical foundations & failure modes of reasoning  
  - ExpEnv: slide examples, classroom demos  

- [Can LLMs Reason & Plan?](https://icml.cc/media/icml-2024/Slides/33965.pdf) (ICML Tutorial 2024)  
  - Arizona State University (Subbarao Kambhampati)  
  - Key: planning-oriented reasoning, agent integration  
  - ExpEnv: symbolic + LLM planning tasks  

- [Towards Reasoning in Large Language Models](https://jeffhj.github.io/files/acl2023-slides-llm-reasoning.pdf) (ACL Tutorial 2023)  
  - UIUC (Jie Huang)  
  - Key: survey of reasoning techniques & benchmarks  
  - ExpEnv: academic tutorial datasets  

- [From System 1 to System 2: A Survey of Reasoning Large Language Models](https://arxiv.org/pdf/2502.17419) (arXiv 2025)  
  - CAS & MBZUAI  
  - Key: cognitive-style taxonomy (fast vs. deliberative reasoning)  
  - ExpEnv: logical, mathematical, commonsense datasets  

- [Harnessing the Reasoning Economy: A Survey of Efficient Reasoning for Large Language Models](https://www.alphaxiv.org/abs/2503.24377) (arXiv 2025)  
  - Chinese Univ. of Hong Kong  
  - Key: efficient reasoning, test-time-compute scaling  
  - ExpEnv: math & code reasoning benchmarks  

- [What, How, Where, and How Well? A Survey on Test-Time Scaling in Large Language Models](https://www.alphaxiv.org/abs/2503.24235) (arXiv 2025)  
  - City University of Hong Kong  
  - Key: methods for scaling inference-time compute (CoT, search, self-consistency)  
  - ExpEnv: diverse reasoning datasets  

- [A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond](https://arxiv.org/pdf/2503.21614) (arXiv 2025)  
  - Shanghai AI Lab et al.  
  - Key: lifecycle-wide efficiency (pre-training → inference) for LRMs  
  - ExpEnv: language + vision reasoning tasks  

- [Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/pdf/2503.16419v1) (arXiv 2025)  
  - Rice University  
  - Key: “overthinking” phenomenon, length-control techniques  
  - ExpEnv: GSM8K, MATH-500, AIME-24    

- [A Visual Guide to Reasoning LLMs](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms) (Newsletter 2025)  
  - Maarten Grootendorst  
  - Key: illustrated test-time-compute concepts, DeepSeek-R1 case study  
  - ExpEnv: graphical explanations & code demos  

- [Understanding Reasoning LLMs – Methods and Strategies for Building and Refining Reasoning Models](https://sebastianraschka.com/blog/2025/understanding-reasoning-llms.html) (Blog 2025)  
  - Sebastian Raschka  
  - Key: practical tutorial on data, architectures, evaluation  
  - ExpEnv: Jupyter notebooks & open-source models  

- [An Illusion of Progress? Assessing the Current State of Web Agents](https://www.alphaxiv.org/abs/2504.01382) (arXiv 2025)  
  - Ohio State & UC Berkeley  
  - Key: empirical audit of LLM-based web agents, evaluation protocols  
  - ExpEnv: autonomous web-navigation tasks  

- [Agentic Large Language Models, A Survey](https://www.alphaxiv.org/abs/2503.23037) (arXiv 2025)  
  - Leiden University  
  - Key: taxonomy of agentic LLM architectures & planning mechanisms  
  - ExpEnv: multi-step reasoning / tool-use agents  

- [A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More](https://arxiv.org/pdf/2407.16216) (arXiv 2024)  
  - Salesforce AI  
  - Key: reward modeling & preference-optimization pipelines  
  - ExpEnv: alignment benchmarks, safety tasks  

- [Self-Improvement of LLM Agents through Reinforcement Learning at Scale](https://www.csail.mit.edu/event/scale-ml-self-improvement-llm-agents-through-reinforcement-learning-scale) (MIT Scale-ML Talk 2024)  
  - MIT CSAIL & collaborators  
  - Key: large-scale RL for autonomous agent refinement  
  - ExpEnv: simulated dialogue & tool-use agents

- [Reinforcement Learning from Verifiable Rewards](https://rlvrbook.com/) (Book 2026)
  - Kian Kyars
  - Key: verifier design, outcome and process rewards, GRPO training signals, reward hacking, search/test-time verification, frontier RLVR, and agentic RLVR
  - ExpEnv: math, code, proof, long-context QA, multimodal, and agentic verification examples

- [Reinforcement Learning from Verifiable Rewards](https://labelstud.io/blog/reinforcement-learning-from-verifiable-rewards/) (Blog 2025)  
  - Key: Uses binary, verifiable reward functions to inject precise, unbiased learning signals into RL pipelines for math, code, and other accuracy-critical tasks.  
  - ExpEnv: Easily reproducible in Jupyter notebooks or any RL setup by plugging in auto-grading tools such as compilers, unit tests, or schema validators.  

</details>

## Codebases


<details open>
  <summary>Click to expand / collapse</summary>

| Project | Stars | Description |
|---------|:------:|-------------|
| [**open-r1**](https://github.com/huggingface/open-r1) | [![Stars](https://img.shields.io/github/stars/huggingface/open-r1?style=flat-square&logo=github)](https://github.com/huggingface/open-r1/stargazers) | Fully open reproduction of the DeepSeek-R1 pipeline (SFT, distillation, GRPO, evaluation) |
| [**OpenRLHF**](https://github.com/OpenRLHF/OpenRLHF) | [![Stars](https://img.shields.io/github/stars/OpenRLHF/OpenRLHF?style=flat-square&logo=github)](https://github.com/OpenRLHF/OpenRLHF/stargazers) | An Easy-to-use, Scalable and High-performance RLHF Framework based on Ray (PPO & GRPO & REINFORCE++ & vLLM & Ray & Dynamic Sampling & Async Agentic RL)* |
| [**verl**](https://github.com/volcengine/verl) | [![Stars](https://img.shields.io/github/stars/volcengine/verl?style=flat-square&logo=github)](https://github.com/volcengine/verl/stargazers)  | a flexible, efficient and production-ready RL training library for large language models |
| [**TinyZero**](https://github.com/Jiayi-Pan/TinyZero) |  [![Stars](https://img.shields.io/github/stars/Jiayi-Pan/TinyZero?style=flat-square&logo=github)](https://github.com/Jiayi-Pan/TinyZero/stargazers) | Minimal reproduction of DeepSeek R1-Zero |
| [**AReaL**](https://github.com/inclusionAI/AReaL) |  [![Stars](https://img.shields.io/github/stars/inclusionAI/AReaL?style=flat-square&logo=github)](https://github.com/inclusionAI/AReaL/stargazers) | Ant Reasoning Reinforcement Learning for LLMs |
| [**Open-Reasoner-Zero**](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) |  [![Stars](https://img.shields.io/github/stars/Open-Reasoner-Zero/Open-Reasoner-Zero?style=flat-square&logo=github)](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/stargazers) | one open source implementation of large-scale reasoning-oriented RL training focusing on scalability, simplicity and accessibility |
| [**ROLL**](https://github.com/alibaba/ROLL) |  [![Stars](https://img.shields.io/github/stars/alibaba/ROLL?style=flat-square&logo=github)](https://github.com/alibaba/ROLL/stargazers) | an Efficient and User-Friendly Scaling Library for Reinforcement Learning with Large Language Models |
| [**slime**](https://github.com/THUDM/slime) |  [![Stars](https://img.shields.io/github/stars/THUDM/slime?style=flat-square&logo=github)](https://github.com/THUDM/slime/stargazers) | an LLM post-training framework for RL scaling with high-performance training and flexible data generation|
| [**RAGEN**](https://github.com/RAGEN-AI/RAGEN) |  [![Stars](https://img.shields.io/github/stars/RAGEN-AI/RAGEN?style=flat-square&logo=github)](https://github.com/RAGEN-AI/RAGEN/stargazers) | RAGEN (Reasoning AGENt, pronounced like "region") leverages reinforcement learning (RL) to train LLM reasoning agents in interactive, stochastic environments. |
| [**PRIME**](https://github.com/PRIME-RL/PRIME) |  [![Stars](https://img.shields.io/github/stars/PRIME-RL/PRIME?style=flat-square&logo=github)](https://github.com/PRIME-RL/PRIME/stargazers) | PRIME (Process Reinforcement through IMplicit REwards), an open-source solution for online RL with process rewards |
| [**rllm**](https://github.com/agentica-project/rllm) |  [![Stars](https://img.shields.io/github/stars/agentica-project/rllm?style=flat-square&logo=github)](https://github.com/agentica-project/rllm/stargazers) | an open-source framework for post-training language agents via reinforcement learning |
| [**Nemo-Aligner**](https://github.com/NVIDIA/NeMo-Aligner) |  [![Stars](https://img.shields.io/github/stars/NVIDIA/NeMo-Aligner?style=flat-square&logo=github)](https://github.com/NVIDIA/NeMo-Aligner/stargazers) | Scalable toolkit for efficient model alignment |
| [**Trinity-RFT**](https://github.com/modelscope/Trinity-RFT) |  [![Stars](https://img.shields.io/github/stars/modelscope/Trinity-RFT?style=flat-square&logo=github)](https://github.com/modelscope/Trinity-RFT/stargazers) | A unified RFT framework with plug-and-play modules (for algorithms, data pipelines, and synchronization) |


</details>


## Papers

### 2025

#### NeurIPS 2025

<details open>
  <summary>Click to expand / collapse</summary>

- [ViCrit: A Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs](https://arxiv.org/abs/2506.10128)
  - Xiyao Wang, Zhengyuan Yang, Chao Feng, Yuhang Zhou, Xiaoyu Liu, Yongyuan Liang, Ming Li, Ziyi Zang, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, Lijuan Wang
  - Key: Visual reasoning, Vision-Language Model, Visual captioning, Reward Model, Visual Hallucination, fine-grained hallucination criticism as RL objective
  - ExpEnv: ViCrit-Bench, natural-image reasoning, abstract image reasoning, visual math

- [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) 
  - Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang
  - Key: RLVR capability boundaries, pass@k analysis, base model vs RLVR-trained models, reasoning pattern emergence, distillation vs RL
  - ExpEnv: Math/coding/visual reasoning benchmarks, multiple model families and RL algorithms

- [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335) 
  - Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, Gao Huang
  - Key: Self-play reasoning, zero external data, self-proposed tasks, curriculum learning, code executor as unified feedback
  - ExpEnv: Coding and mathematical reasoning tasks, cross-model-scale experiments

- [CURE: Co-Evolving Coders and Unit Testers via Reinforcement Learning](https://arxiv.org/abs/2506.03136) 
  - Yinjie Wang, Ling Yang, Ye Tian, Ke Shen, Mengdi Wang
  - Key: Co-evolution of code generation and unit test generation, interaction-based rewards without ground-truth code, test-time scaling, agentic unit test generation
  - ExpEnv: SWE-bench Verified, unit test generation benchmarks, 4B model achieving 64.8% inference efficiency

- [Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles](https://arxiv.org/abs/2505.19914) 
  - Jiangjie Chen, Qianyu He, Siyu Yuan, Aili Chen, Zhicheng Cai, Weinan Dai, Hongli Yu, Jiaze Chen, Xuefeng Li, Qiying Yu, Hao Zhou, Mingxuan Wang
  - Key: Puzzle reasoning, generator-verifier design, 36 tasks across 7 categories, multi-task RLVR, 418K competition-level problems
  - ExpEnv: ENIGMATA-Eval, ARC-AGI (32.8%), ARC-AGI 2 (0.6%), AIME (2024-2025), BeyondAIME, GPQA (Diamond)

- [To Think or Not To Think: A Study of Thinking in Rule-Based Visual Reinforcement Fine-Tuning](https://openreview.net/forum?id=YexxvBGwQM&referrer=%5Bthe%20profile%20of%20Kaipeng%20Zhang%5D(%2Fprofile%3Fid%3D~Kaipeng_Zhang1)) 
  - Ming Li, Jike Zhong, Shitian Zhao, Yuxiang Lai, Haoquan Zhang, Wang Bill Zhu, Kaipeng Zhang
  - Key: Thinking vs No-Thinking RFT, visual perception tasks, overthinking in MLLMs, equality accuracy reward, Adaptive-Thinking method
  - ExpEnv: Six diverse visual tasks across different model sizes and types, image classification benchmarks

- [Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2505.13445) 
  - Xiaoyuan Liu, Tian Liang, Zhiwei He, Jiahao Xu, Wenxuan Wang, Pinjia He, Zhaopeng Tu, Haitao Mi, Dong Yu
  - Key: Self-verification in RLVR, RISE framework, simultaneous training of problem-solving and self-verification, online RL for both tasks
  - ExpEnv: Mathematical reasoning benchmarks, verification compute analysis

- [Beyond Verifiable Rewards: Scaling Reinforcement Learning in Language Models to Unverifiable Data](https://arxiv.org/abs/2503.19618) 
  - Yunhao Tang, Sid Wang, Lovish Madaan, Remi Munos
  - Key: JEPO algorithm, Jensen's evidence lower bound, unverifiable data, chain-of-thought as latent variable, extending RLVR to semi-verifiable data
  - ExpEnv: Math (verifiable), numina and numina-proof (semi-verifiable/unverifiable), test set likelihood evaluation

- [RLVR-World: Training World Models with Reinforcement Learning](https://openreview.net/forum?id=jpiSagi8aV) 
  - Jialong Wu, Shaofeng Yin, Ningya Feng, Mingsheng Long
  - Key: World models with RLVR, task-specific optimization beyond MLE, transition prediction metrics as rewards, autoregressive tokenized sequences
  - ExpEnv: Text games, web navigation, robot manipulation, language-based and video-based world models

- [SeRL: Self-play Reinforcement Learning for Large Language Models with Limited Data](https://arxiv.org/abs/2505.20347) 
  - Wenkai Fang, Shunyu Liu, Yang Zhou, Kongcheng Zhang, Tongya Zheng, Kaixuan Chen, Mingli Song, Dacheng Tao
  - Key: Self-instruction generation, self-rewarding via majority-voting, bootstrapping with limited initial data, online filtering strategies
  - ExpEnv: Various reasoning benchmarks across different LLM backbones

- [Learning to Reason under Off-Policy Guidance](https://arxiv.org/abs/2504.14945) 
  - Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, Yue Zhang
  - Key: LUFFY framework, off-policy reasoning traces, Mixed-Policy GRPO, policy shaping via regularized importance sampling, learning beyond initial capabilities
  - ExpEnv: Six math benchmarks, out-of-distribution tasks, weak model training scenarios

- [SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](https://arxiv.org/abs/2502.18449) 
  - Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, Gabriel Synnaeve, Rishabh Singh, Sida Wang
  - Key: Software engineering with RL, lightweight rule-based reward, open-source software evolution data, Llama3-SWE-RL-70B achieving 41.0% on SWE-bench Verified
  - ExpEnv: SWE-bench Verified, function coding, library use, code reasoning, mathematics, general language understanding

- [rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset](https://arxiv.org/abs/2505.21297) 
  - Yifei Liu, Li Lyna Zhang, Yi Zhu, Bingcheng Dong, Xudong Zhou, Ning Shang, Fan Yang, Cheng Li, Mao Yang
  - Key: 418K competition-level code problems, 580K long-reasoning solutions, input-output test case synthesis pipeline, mutual verification mechanism
  - ExpEnv: LiveCodeBench (57.3% for 7B), USA Computing Olympiad (16.15% avg for 7B, outperforming QWQ-32B)

- [SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond](https://openreview.net/pdf?id=XtNiw8OQsy) 
  - Junteng Liu, Yuanxiang Fan, Jiang Zhuo, Han Ding, Yongyi Hu, Chi Zhang, Yiqi Shi, Shitong Weng, Aili Chen, Shiqi Chen, Mozhi Zhang, Pengyu Zhao, Junxian He
  - Key: Logical reasoning as foundation for general reasoning, 35 diverse logical reasoning tasks, controlled synthesis with adjustable difficulty, mixing with math/coding tasks
  - ExpEnv: BBEH (outperforming DeepSeek-R1-Distill-Qwen-32B by 6 points), state-of-the-art logical reasoning performance

- [SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning](https://openreview.net/pdf/c3403842de341324f63358f1732f1518761661f2.pdf) 
  - Xiao Liang, Zhong-Zhi Li, Yeyun Gong, Yang Wang, Hengyuan Zhang, Yelong Shen, Ying Nian Wu, Weizhu Chen
  - Key: Self-aware weakness identification, problem synthesis targeting model deficiencies, core concept extraction from failure cases, weakness-driven augmentation
  - ExpEnv: Eight mainstream reasoning benchmarks, 10% gain on 7B models, 7.7% gain on 32B models

- [AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning](https://arxiv.org/abs/2505.16400) 
  - Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping
  - Key: Math-only then code-only RL training sequence, curriculum learning with progressive response lengths, robust data curation for challenging prompts, on-policy parameter updates
  - ExpEnv: AIME 2025 (+14.6%/+17.2% for 7B/14B), LiveCodeBench (+6.8%/+5.8% for 7B/14B)

- [Agentic RL Scaling Law: Spontaneous Code Execution for Mathematical Problem Solving](https://openreview.net/pdf?id=kXieirlPjF) 
  - Xinji Mai, Haotian Xu, Xing W, Weinong Wang, Yingying Zhang, Wenqiang Zhang
  - Key: ZeroTIR (Tool-Integrated Reasoning), spontaneous code execution without supervised tool-use examples, RL scaling laws for tool use, predictable metrics scaling
  - ExpEnv: Math benchmarks, standard RL algorithms comparison

- [Rethinking Verification for LLM Code Generation: From Generation to Testing](https://arxiv.org/abs/2507.06920) 
  - Zihan Ma, Taolin Zhang, Maosongcao, Junnan Liu, Wenwei Zhang, Minnan Luo, Songyang Zhang, Kai Chen
  - Key: Test-case generation (TCG) task, multi-dimensional thoroughness metrics, human-LLM collaborative method (SAGA), TCGBench benchmark
  - ExpEnv: TCGBench (90.62% detection rate), LiveCodeBench-v6 (10.78% higher Verifier Acc), test suite quality evaluation

- [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models](https://arxiv.org/abs/2505.24864) 
  - Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, Yi Dong
  - Key: Prolonged RL training, KL divergence control, reference policy resetting, diverse task suite, novel reasoning strategies inaccessible to base models
  - ExpEnv: Wide range of pass@k evaluations, base model competence vs training duration analysis

- [Scaling Code-Assisted Chain-of-Thoughts and Instructions for Model Reasoning](https://arxiv.org/abs/2510.04081) 
  - Honglin Lin, Qizhi Pei, Zhuoshi Pan, Yu Li, Xin Gao, Juntao Li, Conghui He, Lijun Wu
  - Key: Caco framework, code-assisted CoT, automated validation via code execution, reverse-engineering to natural language instructions, Caco-1.3M dataset
  - ExpEnv: Mathematical reasoning benchmarks, code-anchored verification, instruction diversity analysis

- [The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](https://arxiv.org/abs/2506.01347) 
  - Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, Yu Meng
  - Key: Negative Sample Reinforcement (NSR), training with only negative samples, Pass@k across entire spectrum, gradient analysis, upweighting NSR
  - ExpEnv: MATH, AIME 2025, AMC23, Qwen2.5-Math-7B, Qwen3-4B, Llama-3.1-8B-Instruct

- [Solver-Informed RL: Grounding Large Language Models for Authentic Optimization Modeling](https://arxiv.org/abs/2505.11792) 
  - Yitian Chen, Jingfan Xia, Siyu Shao, Dongdong Ge, Yinyu Ye
  - Key: Optimization problem formulation, Solver-Informed RL (SIRL), executable code and .lp file assessment, instance-level mathematical model verification, instance-enhanced self-consistency
  - ExpEnv: Diverse public optimization benchmarks, surpassing DeepSeek-V3 and OpenAI-o3

- [ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data](https://openreview.net/pdf?id=MlJyAvQaxp) 
  - Xiaoyang Liu, Kangjie Bao, Jiashuo Zhang, Yunqi Liu, Yu Chen, Yuntian Liu, Yang Jiao, Tao Luo
  - Key: Autoformalization, Lean 4, expert iteration with knowledge distillation, structural augmentation strategies, 117k undergraduate-level theorem statements
  - ExpEnv: All benchmarks (p<0.05, two-sided t-test), outperforming Herald Translator and Kimina-Autoformalizer

- [QiMeng-CodeV-R1: Reasoning-Enhanced Verilog Generation](https://openreview.net/pdf/d8c56839b04b28b0b8cf304db2a8b3d12509386e.pdf) 
  - Yaoyu Zhu, Di Huang, Hanqi Lyu, Xiaoyun Zhang, Chongxiao Li, Wenxuan Shi, Yutong Wu, Jianan Mu, Jinghua Wang, Yang zhao, Pengwei Jin, Shuyao Cheng, Shengwen Liang, Xishan Zhang, Rui Zhang, Zidong Du, Qi Guo, Xing Hu, Yunji Chen
  - Key: Verilog generation with RLVR, rule-based testbench generator, round-trip data synthesis, distill-then-RL pipeline, adaptive DAPO algorithm
  - ExpEnv: VerilogEval v2 (68.6% pass@1), RTLLM v1.1 (72.9% pass@1), surpassing 671B DeepSeek-R1 on RTLLM

- [QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](https://openreview.net/pdf/4263b7b6c002fbd003c4d4349a6c8cb02ea3cfee.pdf) 
  - Yang Zhang, Rui Zhang, Jiaming Guo, Huang Lei, Di Huang, Yunpu Zhao, Shuyao Cheng, Pengwei Jin, Chongxiao Li, Zidong Du, Xing Hu, Qi Guo, Yunji Chen
  - Key: Signal-level optimization, verified signal-aware implementations extraction, Abstract Syntax Tree (AST) for code segments, signal-aware DPO
  - ExpEnv: VerilogEval, RTLLM, 7B matching DeepSeek v3 671B performance

- [miniF2F-Lean Revisited: Reviewing Limitations and Charting a Path Forward](https://arxiv.org/abs/2511.03108) 
  - Azim Ospanov, Farzan Farnia, Roozbeh Yousefzadeh
  - Key: Formal reasoning, automated theorem proving, Lean prover, miniF2F benchmark analysis, discrepancy correction, miniF2F-v2 with verified statements
  - ExpEnv: miniF2F original vs miniF2F-v2, full theorem proving pipeline evaluation (70% on v2 vs 40% on original)

- [SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications](https://openreview.net/forum?id=yRxXTdElLv) 
  - Jinyang Li, Xiaolong Li, Ge Qu, Per Jacobsson, Bowen Qin, Binyuan Hui, Shuzheng Si, Nan Huo, Xiaohan Xu, Yue Zhang, Ziwei Tang, Yuanshuai Li, Florensia Widjaja, Xintong Zhu, Feige Zhou, Yongfeng Huang, Yannis Papakonstantinou, Fatma Ozcan, Chenhao Ma, Reynold Cheng
  - Key: SQL issue debugging, BIRD-CRITIC benchmark (530 PostgreSQL + 570 multi-dialect tasks), SQL-Rewind strategy, f-Plan Boosting, BIRD-Fixer agent
  - ExpEnv: BIRD-CRITIC-PG (38.11% for 14B), BIRD-CRITIC-Multi (29.65% for 14B), surpassing Claude-3.7-Sonnet and GPT-4.1

</details>

#### ICML 2025

<details open>
  <summary>Click to expand / collapse</summary>

- [SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training](https://openreview.net/pdf?id=dYur3yabMj)
  - Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, Yi Ma
  - Key: Reinforcement Learning, Supervised Fine-tuning, Generalization, Memorization
  - ExpEnv: GeneralPoints, V-IRL

- [VinePPO: Refining Credit Assignment in RL Training of LLMs](https://openreview.net/pdf?id=Myx2kJFzAn)
  - Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, Nicolas Le Roux
  - Key: reinforcement learning, large language models, credit assignment, PPO, Monte Carlo estimation
  - ExpEnv: MATH, GSM8K

- [Controlling Large Language Model with Latent Action](https://openreview.net/pdf?id=cEKrGCFXPA)
  - Chengxing Jia, Ziniu Li, Pengyuan Wang, Yi-Chen Li, Zhenyu Hou, Yuxiao Dong, Yang Yu
  - Key: reinforcement learning, latent action space, controllable language models, inverse dynamics, policy learning
  - ExpEnv: math500, Countdown Game, Alfworld, Scienceworld

- [Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs](https://openreview.net/pdf?id=aOIJ2gVRWW)
  - Jan Betley, Daniel Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, Owain Evans
  - Key: emergent misalignment, insecure code, deceptive behavior, alignment, dataset intent, finetuning impact
  - ExpEnv: GPT-4o, Qwen2.5-32B-Instruct, Mistral-Small-2409, HumanEval, TruthfulQA, StrongREJECT, Machiavelli

- [Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation](https://openreview.net/pdf?id=pLQtovjXiw)
  - Zhuohao Yu, Weizheng Gu, Yidong Wang, Xingru Jiang, Zhengran Zeng, Jindong Wang, Wei Ye, Shikun Zhang
  - Key: code generation, process supervision, outcome supervision, reasoning, execution verification
  - ExpEnv: HumanEval, MBPP, LBPP

- [Demystifying Long Chain-of-Thought Reasoning](https://openreview.net/pdf?id=OLodUbcWjB)
  - Shiming Yang, Yuxuan Tong, Xinyao Niu, Graham Neubig, Xiang Yue
  - Key: long chain-of-thought, reinforcement learning, supervised fine-tuning, reward shaping, verifiable rewards
  - ExpEnv: MATH-500, AIME 2024, TheoremQA, MMLU-Pro-1k

- [MA-LoT: Model-Collaboration Lean-based Long Chain-of-Thought Reasoning enhances Formal Theorem Proving](https://openreview.net/pdf?id=AzF9xAMrBK)
  - Ruida Wang, Rui Pan, Yuxin Li, Jipeng Zhang, Yizhen Jia, Shizhe Diao, Renjie Pi, Junjie Hu, Tong Zhang
  - Key: theorem proving, formal verification, Lean4, large language models, model collaboration, chain-of-thought
  - ExpEnv: MiniF2F-Test

- [SHIELDAGENT: Shielding Agents via Verifiable Safety Policy Reasoning](https://openreview.net/pdf?id=DkRYImuQA9)
  - Zhaorun Chen, Mintong Kang, Bo Li
  - Key: LLM agents, safety policy, guardrails, probabilistic reasoning, policy verification
  - ExpEnv: SHIELDAGENT-BENCH (6 web environments), ST-WebAgentBench, VWA-Adv, AgentHarm

- [TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference](https://openreview.net/pdf?id=8PJmKfeDdp)
  - Jack Min Ong, Matthew Di Ferrante, Aaron Pazdera, Ryan Garner, Sami Jaghouar, Manveer Basra, Max Ryabinin, Johannes Hagemann
  - Key: verifiable inference, locality-sensitive hashing, polynomial encoding, trustless AI, LLM verification
  - ExpEnv: Llama 3.1-8B-Instruct, Intellect-1-Instruct, Gemma-2-9b-it, UltraChat dataset

- [Brain Bandit: A Biologically Grounded Neural Network for Efficient Control of Exploration](https://openreview.net/forum?id=RWJX5F5I9g)
  - Chen Jiang, Jiahui An, Yating Liu, Ni Ji
  - Key: explore-exploit, stochastic Hopfield net, Thompson sampling, brain-inspired RL
  - ExpEnv: MAB tasks, MDP tasks

</details>

#### Other 2025 Papers

<details open>
  <summary>Click to expand / collapse</summary>

- [AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress](https://arxiv.org/abs/2511.08325)
  - Zhiheng Xi, Chenyang Liao, Guanyu Li, Yajie Yang, Wenxiang Chen, Zhihao Zhang, Binghai Wang, Senjie Jin, Yuhao Zhou, Jian Guan, Wei Wu, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang
  - Key: Process Reward Models (PRM), Agentic Tasks, Generalized Advantage Estimation (GAE), Promise, Progress
  - ExpEnv: WebShop, HotpotQA (Agent setting), Interactive Environments

- [Towards Agentic Self-Learning LLMs in Search Environment](https://arxiv.org/abs/2510.14253)
  - Wangtao Sun, Xiang Cheng, Jialin Fan, Xing Yu, Yao Xu, Shizhu He, Jun Zhao, Kang Liu
  - Key: Self-Learning, Co-Evolution, Generative Reward Model (GRM), Search Agents
  - ExpEnv: NQ, TriviaQA, HotpotQA, 2WikiMultiHopQA, Qwen-2.5-7B-Instruct

- [Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers](https://arxiv.org/abs/2510.00915)
  - Xin-Qiang Cai, Wei Wang, Feng Liu, Tongliang Liu, Gang Niu, Masashi Sugiyama
  - Key: Noisy Rewards, Verifier Hacking, Forward/Backward Correction, GRPO, Reference-Free RL
  - ExpEnv: GSM8K, MATH, Qwen2.5-Math-7B

- [Language Models that Think, Chat Better](https://arxiv.org/abs/2509.20357)
  - Adithya Bhaskar, Xi Ye, Danqi Chen
  - Key: Long CoT Reasoning, Large Language Models, GRPO, SFT
  - ExpEnv: WildBench, AlpacaEval2, ArenaHardV2, CreativeWritingV3, IFBench, MMLU-Redux, PopQA

- [RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments](https://arxiv.org/abs/2511.07317)
  - Zhiyuan Zeng, Qinyuan Cheng, Zhangyue Yin, Yefei He, Jinpeng Wang, et al.
  - Key: Adaptive Environments, Verifiable Rewards, Environment Scaling, Curriculum Learning
  - ExpEnv: RLVE-Gym

- [Game-RL: Synthesizing Multimodal Verifiable Game Data to Boost VLMs' General Reasoning](https://openreview.net/pdf?id=e4FqU4SyHL)
  - Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou, Ming Zhang, Jun Zhao, Yanbo Wen, et al.
  - Key: Vision Language Model, Reasoning, Data Synthesis, Game Playing, Visual Question Answering, Data Sets or Data Repositories, Benchmarks
  - ExpEnv: GameQA, CharXiv, MathVerse, MathVision, MathVista, MMBench, MMMU-Pro, MMMU

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)
  - Daya Guo, Dejian Yang, Haowei Zhang *et al.* (DeepSeek-AI)
  - Key: GRPO, pure-RL reasoning, distillation to 1.5 B–70 B, open checkpoints
  - ExpEnv: AIME-2024, MATH-500, Codeforces, LiveCodeBench, GPQA-Diamond, SWE-Bench

- [Demystifying Long Chain-of-Thought Reasoning in LLMs](https://www.alphaxiv.org/abs/2502.03373)
  - IN.AI Research Team
  - Key: cosine length-scaling reward, repetition penalty, stable long CoT
  - ExpEnv: GSM8K, MATH, mixed STEM sets

- [Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning](https://www.alphaxiv.org/abs/2502.06781)
  - Shanghai AI Lab
  - Key: outcome-only reward, sparse-signal RL, math-centric limits
  - ExpEnv: MATH-Benchmark, GSM8K, AIME, proof datasets

- [Kimi K 1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/pdf/2501.12599)
  - Moonshot AI
  - Key: curriculum RL, large-batch PPO, scalable infra
  - ExpEnv: multi-domain reasoning, long-context writing, agent benchmarks

- [S²R: Teaching LLMs to Self-Verify and Self-Correct via Reinforcement Learning](https://arxiv.org/pdf/2502.12853)
  - Tencent AI Lab
  - Key: self-verification & correction loops, dual-reward, safety alignment
  - ExpEnv: math QA, code generation, natural-language inference

- [Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling](https://www.alphaxiv.org/abs/2502.06703)
  - Tsinghua University
  - Key: compute-optimal scaling, small-vs-large model trade-offs
  - ExpEnv: reasoning benchmarks, test-time compute scaling

- [QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search](https://arxiv.org/pdf/2502.02584)
  - UCLA (Yizhou Sun Lab)
  - Key: Q-guided stepwise search, agent inference efficiency
  - ExpEnv: web-agent tasks, reasoning QA

- [Process Reward Models That Think](https://arxiv.org/abs/2504.16828)
  - University of Michigan
  - Key: process reward modelling, reasoning guidance
  - ExpEnv: reasoning QA, code tasks

- [THINKPRUNE: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning](https://arxiv.org/pdf/2504.01296)
  - *Anonymous*
  - Key: CoT pruning through RL, latency reduction
  - ExpEnv: GSM8K, assorted reasoning sets

- [GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning](https://arxiv.org/pdf/2504.02546)
  - *TBD*
  - Key: lightweight RL baseline, strong reasoning gains
  - ExpEnv: diverse reasoning benchmarks

- [When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning](https://arxiv.org/pdf/2504.01005)
  - Google DeepMind
  - Key: dynamic solve-vs-verify decision, compute optimality
  - ExpEnv: math & code tasks

- [SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks](https://arxiv.org/pdf/2503.15478)
  - Meta, UC Berkeley
  - Key: multi-turn agent RL, collaborative reasoning
  - ExpEnv: agent task suites

- [L1: Controlling How Long a Reasoning Model Thinks With Reinforcement Learning](https://www.arxiv.org/pdf/2503.04697)
  - Carnegie Mellon University
  - Key: explicit control of reasoning steps via RL
  - ExpEnv: GSM8K, MATH

- [Scaling Test-Time Compute Without Verification or RL is Suboptimal](https://arxiv.org/pdf/2502.12118)
  - CMU, UC Berkeley
  - Key: verifier-based vs verifier-free compute scaling
  - ExpEnv: reasoning benchmarks

- [DAST: Difficulty-Adaptive Slow-Thinking for Large Reasoning Models](https://arxiv.org/pdf/2503.04472)
  - Unicom Data Intelligence
  - Key: difficulty-adaptive thinking length
  - ExpEnv: reasoning sets

- [Reasoning with Reinforced Functional Token Tuning](https://arxiv.org/pdf/2502.13389)
  - Zhejiang University, Alibaba Cloud Computing
  - Key: functional token tuning, RL-aided reasoning
  - ExpEnv: reasoning QA, code

- [Provably Optimal Distributional RL for LLM Post-Training](https://arxiv.org/pdf/2502.20548)
  - Cornell & Harvard
  - Key: distributional RL theory for LLM post-training
  - ExpEnv: synthetic reasoning, math tasks

- [On the Emergence of Thinking in LLMs I: Searching for the Right Intuition](https://www.alphaxiv.org/abs/2502.06773)
  - MIT
  - Key: self-play RL, emergent reasoning patterns
  - ExpEnv: reasoning games, maths puzzles

- [STP: Self-Play LLM Theorem Provers with Iterative Conjecturing and Proving](https://arxiv.org/pdf/2502.00212)
  - Stanford (Tengyu Ma)
  - Key: theorem proving via self-play, sparse-reward tackling
  - ExpEnv: proof assistant datasets

- [A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility](https://arxiv.org/pdf/2504.07086)
  - University of Cambridge, University of Tübingen
  - Key: evaluation pitfalls, reproducibility guidelines
  - ExpEnv: multiple reasoning benchmarks

- [Recitation over Reasoning: How Cutting-Edge LMs Fail on Elementary Reasoning Problems](https://arxiv.org/pdf/2504.00509)
  - ByteDance Seed
  - Key: fragility to minor perturbations, arithmetic reasoning
  - ExpEnv: elementary school-level arithmetic tasks

- [Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad](https://arxiv.org/pdf/2503.21934v1)
  - ETH Zurich, INSAIT
  - Key: Olympiad-level evaluation, zero-score phenomenon
  - ExpEnv: 2025 USAMO problems

- [(REINFORCE++) A Simple and Efficient Approach for Aligning Large Language Models](https://arxiv.org/pdf/2501.03262)
  - Jian Hu *et al.*
  - Key: REINFORCE++ algorithm, stability vs PPO/GRPO
  - ExpEnv: RLHF alignment suites

- [ReFT v3: Reasoning with Reinforced Fine-Tuning](https://arxiv.org/abs/2401.08967) (ACL 2025)
  - Trung Le, Jiaqi Zhang *et al.*
  - Key: single-stage RLFT, low-cost math alignment
  - ExpEnv: GSM8K, MATH, SVAMP

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)
  - DeepSeek-AI
  - Key: GRPO, math-only RL, verifier-guided sampling
  - ExpEnv: MATH-500, AIME-2024, CNMO-2024

- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/pdf/2405.14734)
  - Shanghai AI Lab
  - Key: reference-free preference optimisation, KL-free objective
  - ExpEnv: AlpacaEval, helpful/harmless RLHF sets

- [DeepSeek-Prover v1.5: Harnessing Proof Assistant Feedback for RL and MCTS](https://arxiv.org/abs/2408.08152)
  - DeepSeek-AI
  - Key: proof-assistant feedback, Monte-Carlo Tree Search
  - ExpEnv: Lean theorem-proving benchmarks

- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/pdf/2411.15124)
  - Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Xinxi Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Øyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, Hannaneh Hajishirzi
  - Key: post-training, supervised finetuning (SFT), Direct Preference Optimization (DPO), RLVR, open LLMs
  - ExpEnv: multi-task language-model benchmarks (Tülu 3 Eval, decontaminated standard suites)

- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)
  - Kimi Team – Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, … , Zongyu Lin
  - Key: RL with LLMs, long-context scaling, policy optimization, long2short CoT, multi-modal reasoning
  - ExpEnv: AIME, MATH 500, Codeforces, MathVista, LiveCodeBench

- [Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/pdf/2402.01306)
  - Stanford University, Contextual AI
  - Key: prospect-theoretic objective for alignment
  - ExpEnv: alignment evaluation suites

- [UI-R1: Enhancing Efficient Action Prediction of GUI Agents by Reinforcement Learning](https://arxiv.org/pdf/2503.21620)
  - Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Han Xiao, Shuai Ren, Guanjing Xiong, Hongsheng Li
  - Key: rule-based rewards, GRPO, multimodal LLM, GUI grounding & action, data-efficient RFT (136 samples)
  - ExpEnv: ScreenSpot, ScreenSpot-Pro, AndroidControl

- [GUI-R1: A Generalist R1-Style Vision-Language Action Model for GUI Agents](https://arxiv.org/pdf/2504.10458)
  - Run Luo, Lu Wang, Wanwei He, Xiaobo Xia
  - Key: unified action space, GRPO, high-level GUI tasks, cross-platform (Win/Linux/Mac/Android/Web), data-efficient RFT (3 K samples)
  - ExpEnv: ScreenSpot, ScreenSpot-Pro, GUI-Act-Web, OmniAct-Web, OmniAct-Desktop, AndroidControl-Low/High, GUI-Odyssey

- [MT-R1-Zero: Advancing LLM-based Machine Translation via R1-Zero-like Reinforcement Learning](https://aclanthology.org/2025.findings-emnlp.1015/)
  - Zhaopeng Feng, Shaosheng Cao, Jiahan Ren, Jiayuan Su, Ruizhe Chen, Yan Zhang, Jian Wu, Zuozhu Liu
  - Key: Machine Translation, Rule-metric Mixed Reward
  - ExpEnv: BLEU, COMETKiwi, XCOMET

- [Smart-Searcher: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](https://aclanthology.org/2025.findings-emnlp.731/)
  - Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao Zhao, Yingqian Min, Xin Zhao, Lei Fang, Ji-Rong Wen
  - Key: Retrieval-Augmented Generation (RAG), Reinforcement Learning, Internal vs External Knowledge, Dynamic Switching
  - ExpEnv: Multi-hop QA benchmarks, Retrieval tasks

</details>

### 2024 & Earlier

<details open>
  <summary>Click to expand / collapse</summary>

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290) (ICLR 2024)  
  - Rafael Raffel *et al.*  
  - Key: preference optimisation without RL, DPO objective  
  - ExpEnv: summarisation, dialogue alignment 
  
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations](https://arxiv.org/abs/2312.08935) (NeurIPS 2023)  
  - Peking University, DeepSeek-AI  
  - Key: step-checker, verifier RL, zero human labels  
  - ExpEnv: GSM8K-Step, MATH-Step  

- [Let’s Verify Step by Step](https://arxiv.org/pdf/2305.20050) (ICML 2023)  
  - OpenAI  
  - Key: verifier prompts, iterative self-improvement  
  - ExpEnv: GSM8K, ProofWriter  

- [Solving Olympiad Geometry without Human Demonstrations](https://www.nature.com/articles/s41586-023-06747-5.pdf) (Nature 2023)  
  - DeepMind  
  - Key: formal geometry solving, RL without human demos  
  - ExpEnv: geometry proof tasks

- [Training Language Models to Follow Instructions with Human Feedback](https://www.mikecaptain.com/resources/pdf/InstructGPT.pdf) (NeurIPS 2022)  
  - OpenAI  
  - Key: PPO-based RLHF, instruction-following alignment  
  - ExpEnv: broad instruction-following tasks (InstructGPT)  

</details>

## Other Awesome Lists

* **[Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)**

* **[Awesome-LLM-Reasoning](https://github.com/atfortes/Awesome-LLM-Reasoning)**

* **[Awesome-RL-Based-LLM-Reasoning](https://github.com/bruno686/Awesome-RL-based-LLM-Reasoning)**
  
* **[Awesome-LLM-RLVR](https://github.com/smiles724/Awesome-LLM-RLVR)**



## [Contributing](CONTRIBUTING.md)

1. **Fork** this repo.  
2. Add a paper/tool entry under the correct section (keep reverse-chronological order, follow the three-line format).  
3. **Open a Pull Request** and briefly describe your changes.  



## License
Awesome-RLVR © 2025 OpenDILab & Contributors
Apache 2.0 License

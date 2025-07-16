# Awesome RLVR – Reinforcement Learning with **Verifiable** Rewards

[![Stars](https://img.shields.io/github/stars/opendilab/awesome-RLVR?style=flat-square&logo=github)](https://github.com/opendilab/awesome-RLVR/stargazers)
[![Forks](https://img.shields.io/github/forks/opendilab/awesome-RLVR?style=flat-square&logo=github)](https://github.com/opendilab/awesome-RLVR/network/members)
[![Contributors](https://img.shields.io/github/contributors/opendilab/awesome-RLVR?style=flat-square&logo=github)](https://github.com/opendilab/awesome-RLVR/graphs/contributors)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](./LICENSE)

> A curated list of surveys, tutorials, codebases, and papers around  
> **Reinforcement Learning with Verifiable Rewards (RLVR)** –  
> an emerging paradigm that aligns LLMs *and* other agents through external verification, self-consistency and bootstrapped improvement.

---

## Why RLVR?

* **External, objective rewards** – unit tests, formal checkers, fact-checkers, etc.  
* **Built-in safety & auditability** – every reward is traceable to a verifier.  
* **Bootstrapped self-improvement** – agents can refine their own verifiers.  
* **Domain-agnostic** – code generation, theorem proving, robotics, games …

## How does it work?

1. **Policy** proposes one or more candidate outputs.  
2. **Verifier** (independent or learned) checks each output.  
3. **Reward** is emitted & the policy is updated via RL (e.g. PPO).  
4. *Optionally* the verifier itself is trained or hardened over time.

<details>
<summary>▶ Show RLVR training loop diagram</summary>

<!-- self-contained, responsive SVG -->
<div align="center" role="img" aria-label="RLVR training loop">
  <svg viewBox="0 0 800 400" width="100%" height="auto">
    <rect x="50" y="160" width="180" height="80" fill="#3b82f6" rx="10"/>
    <text x="140" y="205" text-anchor="middle" font-size="16" fill="#fff">Policy (LLM)</text>

    <rect x="300" y="40" width="200" height="120" fill="#10b981" rx="10"/>
    <text x="400" y="100" text-anchor="middle" font-size="16" fill="#fff">Environment / Task</text>

    <rect x="300" y="240" width="200" height="120" fill="#f97316" rx="10"/>
    <text x="400" y="300" text-anchor="middle" font-size="16" fill="#fff">Verifier</text>

    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto">
        <path d="M0,0 L0,6 L9,3 z" fill="#444"/>
      </marker>
    </defs>

    <line x1="230" y1="200" x2="300" y2="100" stroke="#444" stroke-width="2" marker-end="url(#arrow)"/>
    <text x="265" y="140" text-anchor="middle" font-size="14">Action</text>

    <line x1="400" y1="160" x2="400" y2="240" stroke="#444" stroke-width="2" marker-end="url(#arrow)"/>
    <text x="417" y="205" font-size="14">Output</text>

    <line x1="300" y1="300" x2="230" y2="200" stroke="#444" stroke-width="2" marker-end="url(#arrow)"/>
    <text x="265" y="270" text-anchor="middle" font-size="14">Reward</text>
  </svg>
</div>

</details>


Pull requests are welcome 🎉 — see [Contributing](#contributing) for details.


<pre>
<font color="red">[2025-07-03] <b>New!</b> Initial public release of Awesome-RLVR 🎉</font>
</pre>


## Table of Contents

- [Awesome RLVR – Reinforcement Learning with **Verifiable** Rewards](#awesome-rlvr--reinforcement-learning-with-verifiable-rewards)
  - [Why RLVR?](#why-rlvr)
  - [How does it work?](#how-does-it-work)
  - [Table of Contents](#table-of-contents)
  - [Surveys \& Tutorials](#surveys--tutorials)
  - [Codebases](#codebases)
  - [Papers](#papers)
    - [2025](#2025)
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

## Codebases

| Project | Description |
|---------|-------------|
| [**open-r1**](https://github.com/huggingface/open-r1) | Fully open reproduction of the DeepSeek-R1 pipeline (SFT → GRPO → evaluation). |
| [**OpenRLHF**](https://github.com/OpenRLHF/OpenRLHF) | An Easy-to-use, Scalable and High-performance RLHF Framework based on Ray (PPO & GRPO & REINFORCE++ & vLLM & Ray & Dynamic Sampling & Async Agentic RL) |
| [**verl**](https://github.com/volcengine/verl) | Volcano Engine reinforcement-learning framework; supports APPO, GRPO, TPPO. |
| [**TinyZero**](https://github.com/Jiayi-Pan/TinyZero) | ~200-line minimal reproduction of DeepSeek R1-Zero; 4 × RTX 4090 is enough for a 0.5 B LLM. |
| [**PRIME**](https://github.com/PRIME-RL/PRIME) | Efficient RL-VR (value/reward) training stack for reasoning LLMs. |
| [**simpleRL-reason**](https://github.com/hkust-nlp/simpleRL-reason) | Minimal, didactic RL-VR trainer for reasoning research. |
| [**rllm**](https://github.com/agentica-project/rllm) | General-purpose “RL-for-LLM” toolbox (algos, logging, evaluators). |
| [**OpenR**](https://github.com/openreasoner/openr) | Modular framework for advanced reasoning (process supervision, MCTS, verifier RMs). |
| [**Open-Reasoner-Zero**](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) | a self-play “Zero-RL” framework focused on advanced mathematical / coding reasoning.  Includes process-supervision data pipelines, verifier-based dense rewards, and async multi-agent RL training scripts. |
| [**ROLL**](https://github.com/alibaba/ROLL) |  An Efficient and User-Friendly Scaling Library for Reinforcement Learning with Large Language Models |

## Papers

### 2025

- [Brain Bandit: A Biologically Grounded Neural Network for Efficient Control of Exploration](https://openreview.net/forum?id=RWJX5F5I9g)  
  - Chen Jiang, Jiahui An, Yating Liu, Ni Ji  
  - Key: explore-exploit, stochastic Hopfield net, Thompson sampling, brain-inspired RL  
  - ExpEnv: MAB tasks, MDP tasks  

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

- [SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training](https://www.alphaxiv.org/abs/2501.17161v1)  
  - The University of Hong Kong & UC Berkeley  
  - Key: SFT vs RLHF/RLVR, memorization-generalization trade-off  
  - ExpEnv: held-out reasoning & knowledge shift tests  

- [Kimi K 1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/pdf/2501.12599)  
  - Moonshot AI  
  - Key: curriculum RL, large-batch PPO, scalable infra  
  - ExpEnv: multi-domain reasoning, long-context writing, agent benchmarks  

- [S²R: Teaching LLMs to Self-Verify and Self-Correct via Reinforcement Learning](https://arxiv.org/pdf/2502.12853)  
  - Tencent AI Lab  
  - Key: self-verification & correction loops, dual-reward, safety alignment  
  - ExpEnv: math QA, code generation, natural-language inference  

- [Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling](https://www.alphaxiv.org/abs/2502.06703) (arXiv)  
  - Tsinghua University  
  - Key: compute-optimal scaling, small-vs-large model trade-offs  
  - ExpEnv: reasoning benchmarks, test-time compute scaling  

- [QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search](https://arxiv.org/pdf/2502.02584) (arXiv)  
  - UCLA (Yizhou Sun Lab)  
  - Key: Q-guided stepwise search, agent inference efficiency  
  - ExpEnv: web-agent tasks, reasoning QA  

- [Solving Math Word Problems with Process- and Outcome-Based Feedback](https://arxiv.org/pdf/2211.14275) (NeurIPS 2023)  
  - DeepMind  
  - Key: process & outcome rewards, verifier feedback for math  
  - ExpEnv: GSM8K, MATH  

- [Process Reward Models That Think](https://arxiv.org/abs/2504.16828) (arXiv)  
  - University of Michigan  
  - Key: process reward modelling, reasoning guidance  
  - ExpEnv: reasoning QA, code tasks  

- [Learning to Reason under Off-Policy Guidance](https://arxiv.org/abs/2504.14945) (arXiv)  
  - Shanghai AI Lab  
  - Key: off-policy guidance for reasoning RL  
  - ExpEnv: math and code benchmarks  

- [THINKPRUNE: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning](https://arxiv.org/pdf/2504.01296) (arXiv)  
  - *Anonymous*  
  - Key: CoT pruning through RL, latency reduction  
  - ExpEnv: GSM8K, assorted reasoning sets  

- [GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning](https://arxiv.org/pdf/2504.02546) (arXiv)  
  - *TBD*  
  - Key: lightweight RL baseline, strong reasoning gains  
  - ExpEnv: diverse reasoning benchmarks  

- [When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning](https://arxiv.org/pdf/2504.01005) (arXiv)  
  - Google DeepMind 
  - Key: dynamic solve-vs-verify decision, compute optimality  
  - ExpEnv: math & code tasks  

- [SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks](https://arxiv.org/pdf/2503.15478) (arXiv)  
  - Meta, UC Berkeley  
  - Key: multi-turn agent RL, collaborative reasoning  
  - ExpEnv: agent task suites  

- [L1: Controlling How Long a Reasoning Model Thinks With Reinforcement Learning](https://www.arxiv.org/pdf/2503.04697) (arXiv)  
  - Carnegie Mellon University  
  - Key: explicit control of reasoning steps via RL  
  - ExpEnv: GSM8K, MATH  

- [Scaling Test-Time Compute Without Verification or RL is Suboptimal](https://arxiv.org/pdf/2502.12118) (arXiv)  
  - CMU, UC Berkeley  
  - Key: verifier-based vs verifier-free compute scaling  
  - ExpEnv: reasoning benchmarks  

- [DAST: Difficulty-Adaptive Slow-Thinking for Large Reasoning Models](https://arxiv.org/pdf/2503.04472) (arXiv)  
  - Unicom Data Intelligence 
  - Key: difficulty-adaptive thinking length  
  - ExpEnv: reasoning sets  
  
- [Reasoning with Reinforced Functional Token Tuning](https://arxiv.org/pdf/2502.13389) (arXiv)  
  - Zhejiang University, Alibaba Cloud Computing 
  - Key: functional token tuning, RL-aided reasoning  
  - ExpEnv: reasoning QA, code  

- [Provably Optimal Distributional RL for LLM Post-Training](https://arxiv.org/pdf/2502.20548) (arXiv)  
  - Cornell & Harvard  
  - Key: distributional RL theory for LLM post-training  
  - ExpEnv: synthetic reasoning, math tasks  

- [On the Emergence of Thinking in LLMs I: Searching for the Right Intuition](https://www.alphaxiv.org/abs/2502.06773) (arXiv)  
  - MIT  
  - Key: self-play RL, emergent reasoning patterns  
  - ExpEnv: reasoning games, maths puzzles  

- [STP: Self-Play LLM Theorem Provers with Iterative Conjecturing and Proving](https://arxiv.org/pdf/2502.00212) (arXiv)  
  - Stanford (Tengyu Ma)  
  - Key: theorem proving via self-play, sparse-reward tackling  
  - ExpEnv: proof assistant datasets  

- [A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility](https://arxiv.org/pdf/2504.07086) (arXiv)  
  - University of Cambridge, University of Tübingen
  - Key: evaluation pitfalls, reproducibility guidelines  
  - ExpEnv: multiple reasoning benchmarks  

- [Recitation over Reasoning: How Cutting-Edge LMs Fail on Elementary Reasoning Problems](https://arxiv.org/pdf/2504.00509) (arXiv)  
  - ByteDance Seed  
  - Key: fragility to minor perturbations, arithmetic reasoning  
  - ExpEnv: elementary school-level arithmetic tasks  

- [Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad](https://arxiv.org/pdf/2503.21934v1) (arXiv)  
  - ETH Zurich, INSAIT 
  - Key: Olympiad-level evaluation, zero-score phenomenon  
  - ExpEnv: 2025 USAMO problems  

- [(REINFORCE++) A Simple and Efficient Approach for Aligning Large Language Models](https://arxiv.org/pdf/2501.03262) (arXiv)  
  - Jian Hu *et al.*   
  - Key: REINFORCE++ algorithm, stability vs PPO/GRPO  
  - ExpEnv: RLHF alignment suites  

- [ReFT v3: Reasoning with Reinforced Fine-Tuning](https://arxiv.org/abs/2401.08967) (ACL 2025 Long)  
  - Trung Le, Jiaqi Zhang *et al.*  
  - Key: single-stage RLFT, low-cost math alignment  
  - ExpEnv: GSM8K, MATH, SVAMP  

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300) (Technical Report)  
  - DeepSeek-AI  
  - Key: GRPO, math-only RL, verifier-guided sampling  
  - ExpEnv: MATH-500, AIME-2024, CNMO-2024  

- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/pdf/2405.14734) (arXiv)  
  - Shanghai AI Lab  
  - Key: reference-free preference optimisation, KL-free objective  
  - ExpEnv: AlpacaEval, helpful/harmless RLHF sets  

- [DeepSeek-Prover v1.5: Harnessing Proof Assistant Feedback for RL and MCTS](https://arxiv.org/abs/2408.08152) (arXiv)  
  - DeepSeek-AI  
  - Key: proof-assistant feedback, Monte-Carlo Tree Search  
  - ExpEnv: Lean theorem-proving benchmarks  

- [Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/pdf/2402.01306) (arXiv)  
  - Stanford University, Contextual AI. 
  - Key: prospect-theoretic objective for alignment  
  - ExpEnv: alignment evaluation suites  

### 2024 & Earlier
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

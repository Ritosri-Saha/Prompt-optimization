# 🧬 Evolutionary Prompt Optimization for LLMs & VLMs

> **Automating prompt engineering through evolutionary search for improved reasoning performance.**

## 🔍 Overview

This project investigates **automated prompt optimization using Genetic Algorithms (GAs)** for Large Language Models (LLMs) and Vision-Language Models (VLMs).

Instead of manually designing prompts, the framework evolves a population of candidate prompts through **fitness-based selection, mutation, and crossover**, progressively searching for more effective reasoning strategies.

### 🔄 Optimization Pipeline

```text
Initial Prompt Population
          ↓
    Fitness Evaluation
          ↓
       Selection
       ↙       ↘
   Mutation   Crossover
       ↘       ↙
   New Prompt Population
          ↓
     Next Generation
          ↓
    Optimized Prompt
```

## 🧠 LLM Track

The LLM pipeline evaluates evolved prompts on reasoning benchmarks, with the current implementation focusing on selected **Big-Bench Hard (BBH)** tasks.

It investigates:

- Zero-shot and few-shot prompting
- Chain-of-Thought reasoning
- Automated prompt mutation
- Prompt crossover and selection
- Performance across diverse reasoning tasks

## 👁️ VLM Track

The framework is extended to **Vision-Language Models** to investigate whether evolutionary prompt optimization can improve multimodal reasoning involving both visual and textual information.

## 📊 Evaluation

Candidate prompts are evaluated using task performance, allowing the Genetic Algorithm to retain stronger prompts and generate improved candidates over successive generations.

Baseline prompting strategies are compared against the evolved prompts to study the effectiveness of automated optimization.

## 📁 Repository Structure

```text
prompt-optimization-ga/
│
├── llm_prompt_optimization.py    # LLM prompt optimization
├── vlm_prompt_optimization.py    # VLM prompt optimization
└── README.md
```

## 🛠️ Tech Stack

`Python` · `Marimo` · `PyTorch` · `Hugging Face` · `Genetic Algorithms`

## 🎯 Research Focus

**Automated Prompt Optimization**  
**LLM & VLM Reasoning**  
**Evolutionary AI**  
**Efficient & Reliable AI**

## 🚧 Research Status

**Ongoing Research**

The implementation and experiments are continuously being refined, with future work focused on broader benchmarks, additional model families, and deeper analysis of evolved reasoning prompts.

---

*This repository accompanies ongoing research on evolutionary prompt optimization for language and vision-language models.*

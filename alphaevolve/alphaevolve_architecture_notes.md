# AlphaEvolve Architecture - Key Components (from DeepMind Paper)

## Core Loop (Figure 2 - Distributed Controller Loop)
```
parent_program, inspirations = database.sample()
prompt = prompt_sampler.build(parent_program, inspirations)
diff = llm.generate(prompt)
child_program = apply_diff(parent_program, diff)
results = evaluator.execute(child_program)
database.add(child_program, results)
```

## 5 Key Components

### 1. Task Specification (Section 2.1)
- User provides: evaluation function h, initial solution, EVOLVE-BLOCK markers
- Evaluation function returns dict of scalar metrics (maximized)
- Supports evolving entire code files, not just single functions
- EVOLVE-BLOCK-START / EVOLVE-BLOCK-END markers in code

### 2. Prompt Sampler (Section 2.2)
- Constructs rich prompts from Program Database
- Includes: explicit context, stochastic formatting, rendered evaluation results
- **Meta prompt evolution**: instructions suggested by LLM itself, co-evolved
- Multiple previously discovered solutions sampled as inspiration
- Template placeholders with probability distributions for diversity

### 3. Creative Generation / LLM Ensemble (Section 2.3)
- **Ensemble of models**: Flash (high throughput) + Pro (high quality)
- Output format: SEARCH/REPLACE diff blocks
- Can also output entire code block for short code
- Model-agnostic design (benefits from better LLMs)

### 4. Evaluation (Section 2.4)
- **Evaluation cascade**: increasing difficulty stages, prune early
- **LLM-generated feedback**: grade properties like simplicity
- **Parallelized evaluation**: embarrassingly parallel, async
- **Multiple scores**: optimize multiple metrics simultaneously
- Even for single target, multi-metric improves results

### 5. Evolution / Program Database (Section 2.5)
- **MAP-Elites algorithm** + **island-based population models**
- Balances exploration vs exploitation
- Stores solutions with scores and program outputs
- Resurfaces previously explored ideas in future generations
- Maintains diversity while improving best programs

### 6. Distributed Pipeline (Section 2.6)
- Asynchronous pipeline using asyncio
- Controller + LLM samplers + evaluation nodes
- Optimized for throughput (not single computation speed)
- Many computations run concurrently

## What Our Current Implementation is MISSING

### Critical Gaps:
1. **No Program Database** - we don't store/resurface past solutions with MAP-Elites
2. **No diff-based evolution** - we generate entire functions, not SEARCH/REPLACE diffs
3. **No prompt sampling with inspirations** - we don't sample multiple past solutions
4. **No evaluation cascade** - we evaluate everything at full scale
5. **No MAP-Elites + island model** - we use simple tournament selection
6. **No meta prompt evolution** - no co-evolved prompts
7. **No multi-metric optimization** - we use single combined fitness
8. **No async pipeline** - we run sequentially
9. **No LLM ensemble** - we use single model at a time
10. **No EVOLVE-BLOCK markers** - we evolve a fixed function signature

### What We DO Have (correctly):
- LLM generates code (but whole function, not diffs)
- Evaluation function (fitness.py)
- Iterative improvement loop
- Seed strategies as initial solutions

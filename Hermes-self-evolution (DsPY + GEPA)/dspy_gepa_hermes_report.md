# DSPy, GEPA, and Hermes Self-Evolution
## A Deep-Dive Reference for Researchers and Practitioners

> **Scope:** This report covers the architecture, internal mechanisms, library code, and integration patterns of DSPy, GEPA, and the Hermes Agent Self-Evolution repository. It is written to serve both as a study reference and as a presentation guide.

---

## Table of Contents

1. [Part I — DSPy](#part-i--dspy)
   - [What Problem DSPy Solves](#what-problem-dspy-solves)
   - [Core Programming Model](#core-programming-model)
   - [Signatures — The Declarative Contract](#signatures--the-declarative-contract)
   - [Modules — The Strategy Layer](#modules--the-strategy-layer)
   - [Optimizers — The Compilation Layer](#optimizers--the-compilation-layer)
   - [How Compilation Works Internally](#how-compilation-works-internally)
   - [DSPy Traces and Why They Matter](#dspy-traces-and-why-they-matter)
   - [Code Examples: Building a DSPy Program](#code-examples-building-a-dspy-program)
   - [Integrating DSPy with LangGraph and External Agents](#integrating-dspy-with-langgraph-and-external-agents)

2. [Part II — GEPA](#part-ii--gepa)
   - [What GEPA Is and Where It Came From](#what-gepa-is-and-where-it-came-from)
   - [Why GEPA Is Different from RL](#why-gepa-is-different-from-rl)
   - [The Core Algorithm: Reflect → Mutate → Select](#the-core-algorithm-reflect--mutate--select)
   - [Pareto-Aware Selection](#pareto-aware-selection)
   - [The Feedback Function](#the-feedback-function)
   - [GEPA Inside DSPy: dspy.GEPA](#gepa-inside-dspy-dspygepa)
   - [GEPA Standalone: The gepa-ai/gepa Library](#gepa-standalone-the-gepa-aigepa-library)
   - [Available Adapters](#available-adapters)
   - [Key Benchmarks and Results](#key-benchmarks-and-results)

3. [Part III — Hermes Agent Self-Evolution](#part-iii--hermes-agent-self-evolution)
   - [What the Repo Does](#what-the-repo-does)
   - [Repository Structure](#repository-structure)
   - [The Five-Phase Optimization Plan](#the-five-phase-optimization-plan)
   - [Engine 1: DSPy + GEPA (Skill and Prompt Evolution)](#engine-1-dspy--gepa-skill-and-prompt-evolution)
   - [Engine 2: Darwinian Evolver (Code Evolution)](#engine-2-darwinian-evolver-code-evolution)
   - [Eval Data Strategy: Synthetic vs SessionDB](#eval-data-strategy-synthetic-vs-sessiondb)
   - [Guardrails and Safety Constraints](#guardrails-and-safety-constraints)
   - [The PR-Based Deployment Model](#the-pr-based-deployment-model)

4. [Part IV — Integration Patterns](#part-iv--integration-patterns)
   - [DSPy + LangGraph](#dspy--langgraph)
   - [GEPA + MCP Tools](#gepa--mcp-tools)
   - [Hermes Self-Evolution + Your Cursor/Claude Code Setup](#hermes-self-evolution--your-cursorclaude-code-setup)
   - [The Full Envisioned Architecture](#the-full-envisioned-architecture)

5. [Part V — Presentation Guide](#part-v--presentation-guide)

---

## Part I — DSPy

### What Problem DSPy Solves

Before DSPy, building LLM applications meant writing prompts by hand. This created a fundamental software engineering problem: the "program" of an LLM application was encoded in fragile natural-language strings, not in structured code.

When you changed the underlying model (e.g., from GPT-3.5 to Claude), the prompt often broke. When you chained multiple LLM calls together, there was no principled way to optimize the whole pipeline — you could only tune each prompt in isolation. And as systems grew more complex, the gap between what you wanted the system to do (a high-level goal) and how you had to specify it (a precisely engineered prompt) widened.

**DSPy's answer:** separate the *what* from the *how*. You declare what transformations you need (in structured signatures), and the framework figures out how to prompt the model to achieve them. The prompt itself becomes a compiled artifact, not a hand-crafted string.

The analogy from the original Stanford paper is PyTorch vs. earlier deep learning frameworks. In PyTorch, you write a forward pass in Python and the framework figures out the backward pass. In DSPy, you write a pipeline in Python and the framework figures out the prompts.

---

### Core Programming Model

DSPy has three core abstractions, and they map cleanly to three questions:

| Abstraction | Question it answers |
|---|---|
| **Signature** | *What* should this LLM call do? (inputs/outputs) |
| **Module** | *How* should this LLM call be executed? (strategy) |
| **Optimizer** | *How good* is this pipeline, and how can it be improved? (tuning) |

Every DSPy program is: signatures inside modules, modules composed into pipelines, pipelines compiled by optimizers.

```
Your code (Python)
    │
    ├── Signatures  ──► "question, context -> answer: str"
    │
    ├── Modules     ──► dspy.ChainOfThought(signature)
    │
    └── Optimizer   ──► dspy.MIPROv2 / dspy.GEPA
                         │
                         ▼
                   Compiled program with optimized prompts
```

---

### Signatures — The Declarative Contract

A **signature** is a typed declaration of what an LLM call should consume and produce. It is the most fundamental building block in DSPy.

#### String form (simple)

```python
import dspy

# Minimal: "inputs -> outputs"
qa = dspy.Predict("question -> answer")
result = qa(question="What is Bayesian optimization?")
print(result.answer)

# With types
classify = dspy.Predict("sentence -> sentiment: bool")
result = classify(sentence="The experiment worked perfectly.")
print(result.sentiment)  # True
```

#### Class form (full power)

```python
class GenerateLiteratureReview(dspy.Signature):
    """
    Given a research topic and a list of abstracts, generate a concise
    literature review section with Vancouver-style citations.
    """
    topic: str = dspy.InputField(desc="The research topic or question")
    abstracts: list[str] = dspy.InputField(desc="List of relevant paper abstracts")
    
    review: str = dspy.OutputField(desc="Literature review paragraph, 200-300 words")
    citations: list[str] = dspy.OutputField(desc="List of cited paper titles")
```

**Key insight:** The docstring becomes part of the instruction given to the LLM. The field names and `desc` parameters guide what the LLM puts in each field. DSPy's optimizer can later rewrite both the docstring and the field descriptions to improve performance.

#### What the optimizer sees

Internally, every `Signature` is a Pydantic `BaseModel`. The optimizer can read and modify:
- The docstring (the task instruction)
- Each field's `desc` (what to put here)
- The few-shot demonstrations (examples injected into context)

This is what makes optimization possible: the optimizer has structured access to every tunable part of the prompt.

---

### Modules — The Strategy Layer

A **module** wraps a signature and adds a *prompting strategy*. Two different modules can share the same signature but prompt the LLM completely differently.

#### Built-in modules

| Module | Strategy | When to use |
|---|---|---|
| `dspy.Predict` | Direct prediction | Simple single-step tasks |
| `dspy.ChainOfThought` | Step-by-step reasoning (`rationale -> answer`) | Tasks requiring reasoning |
| `dspy.ChainOfThoughtWithHint` | CoT + user hint | When you can provide partial guidance |
| `dspy.ReAct` | Reason + Act loop (uses tools) | Agentic tool-use tasks |
| `dspy.ProgramOfThought` | Generate and execute code | Math, computation |
| `dspy.MultiChainComparison` | Generate N chains, compare | High-stakes decisions |
| `dspy.Retry` | Re-call with constraints | When outputs must satisfy assertions |

#### Using a module

```python
# Predict: direct answer
predictor = dspy.Predict("question -> answer")

# ChainOfThought: adds rationale field automatically
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="Why does SVGD produce mode collapse on multimodal targets?")
print(result.rationale)  # Reasoning trace
print(result.answer)     # Final answer

# ReAct: tool-using agent
def search_arxiv(query: str) -> str:
    """Search for papers on arxiv."""
    # ... your search implementation
    return results

react_agent = dspy.ReAct(
    "research_topic -> literature_summary",
    tools=[search_arxiv]
)
```

#### Composing modules into pipelines

```python
class BayesianOptPaper(dspy.Module):
    def __init__(self):
        self.generate_outline   = dspy.ChainOfThought("topic, prior_work -> outline")
        self.write_introduction = dspy.ChainOfThought("topic, outline -> introduction")
        self.find_citations     = dspy.ReAct("introduction -> citations", tools=[search_arxiv])
    
    def forward(self, topic: str, prior_work: str) -> dspy.Prediction:
        outline = self.generate_outline(topic=topic, prior_work=prior_work)
        intro   = self.write_introduction(topic=topic, outline=outline.outline)
        cites   = self.find_citations(introduction=intro.introduction)
        return dspy.Prediction(
            introduction=intro.introduction,
            citations=cites.citations
        )
```

The `forward` method is pure Python. You can use any control flow: `if`, `for`, function calls. DSPy doesn't impose a graph structure — the structure is implicit in the Python code.

---

### Optimizers — The Compilation Layer

The optimizer (called **teleprompter** in early DSPy versions) takes a compiled program, a training set, and a metric function, then searches for better prompts.

#### The optimizer API

```python
# Define your metric
def paper_quality_metric(example, prediction, trace=None) -> float:
    """Returns 0.0-1.0 score for prediction quality."""
    # Your evaluation logic
    score = evaluate_citation_format(prediction.citations)
    score += evaluate_intro_clarity(prediction.introduction)
    return score / 2.0

# Load training examples
trainset = [
    dspy.Example(
        topic="Gaussian Process regression",
        prior_work="[list of abstracts]"
    ).with_inputs("topic", "prior_work")
    for ... in your_data
]

# Compile
optimizer = dspy.MIPROv2(metric=paper_quality_metric, auto="medium")
compiled_program = optimizer.compile(BayesianOptPaper(), trainset=trainset)

# Save
compiled_program.save("compiled_paper_writer.json")

# Load later — no re-optimization needed
program = BayesianOptPaper()
program.load("compiled_paper_writer.json")
```

#### The optimizer landscape

| Optimizer | Strategy | Best for | Cost |
|---|---|---|---|
| `BootstrapFewShot` | Traces successful completions as few-shot examples | Quick baseline | Very low |
| `BootstrapFewShotWithRandomSearch` | BootstrapFewShot + random search over candidates | Better accuracy | Low |
| `MIPROv2` | Bayesian optimization over instructions + few-shots | General purpose | Medium |
| `GRPO` | Gradient-based RL over LLM weights | Fine-tuning | High (GPU) |
| `GEPA` | Reflective evolutionary search (see Part II) | Agentic/complex tasks | Medium |

---

### How Compilation Works Internally

When you call `optimizer.compile(program, trainset)`, this is what happens:

```
1. BOOTSTRAP PHASE
   ├── Run the uncompiled program on each training example
   ├── Collect execution traces (all LLM calls + their inputs/outputs)
   └── Filter traces where the program succeeded (metric > threshold)

2. CANDIDATE GENERATION PHASE
   ├── Extract successful traces as potential few-shot demonstrations
   ├── Generate candidate instructions (by prompting another LLM to
   │   propose improvements to the current docstring)
   └── Build a search space: (instruction variant) × (demo subset)

3. OPTIMIZATION PHASE
   ├── Evaluate candidate combinations on a held-out validation set
   ├── Score each combination with your metric function
   └── Select the combination with the highest score

4. OUTPUT
   └── A compiled .json file containing:
       ├── The winning instruction for each module
       ├── The winning few-shot demo set for each module
       └── The program's full state (loadable without re-optimizing)
```

The key insight: **no gradients, no GPU.** Everything is discrete search over prompt text, evaluated by running the program and scoring the outputs. This is why DSPy works with closed-source APIs.

---

### DSPy Traces and Why They Matter

A **trace** is the full record of a program's execution: every LLM call, every input passed to it, every output it produced, and every tool call made. Traces are the central data structure that connects DSPy to GEPA.

```python
# Traces are captured automatically during compilation
# But you can also inspect them manually:

with dspy.context(trace=[]):
    result = compiled_program(topic="POGPN", prior_work="...")
    trace = dspy.settings.trace  # Full execution record
```

A trace for a 3-module pipeline might look like:
```
[
  {"module": "generate_outline", "input": {"topic": "...", "prior_work": "..."}, 
   "output": {"outline": "..."}, "score": null},
  {"module": "write_introduction", "input": {"topic": "...", "outline": "..."},
   "output": {"introduction": "..."}, "score": null},
  {"module": "find_citations", "input": {"introduction": "..."},
   "output": {"citations": [...]}, "tool_calls": [...], "score": 0.82}
]
```

Optimizers consume these traces to understand what the program did and why a particular run succeeded or failed. GEPA, in particular, passes these raw traces to an LLM and asks it to explain what went wrong and propose targeted fixes.

---

### Code Examples: Building a DSPy Program

#### Example 1: A research paper section writer

```python
import dspy

# Configure the LM
lm = dspy.LM("anthropic/claude-sonnet-4-20250514")
dspy.configure(lm=lm)

# Signature
class WriteMethodsSection(dspy.Signature):
    """
    Write a methods section for a machine learning paper.
    Use passive voice, present tense, and be precise about mathematical notation.
    Reference equations with \\eqref{} syntax.
    """
    topic: str       = dspy.InputField(desc="The method being described")
    equations: str   = dspy.InputField(desc="Key equations in LaTeX format")
    experiment: str  = dspy.InputField(desc="Experimental setup details")
    
    methods_tex: str = dspy.OutputField(desc="Methods section in LaTeX format, 400-600 words")
    label_list: list = dspy.OutputField(desc="List of equation labels referenced in the text")

# Module
class PaperSectionModule(dspy.Module):
    def __init__(self):
        self.writer = dspy.ChainOfThought(WriteMethodsSection)
    
    def forward(self, topic, equations, experiment):
        result = self.writer(topic=topic, equations=equations, experiment=experiment)
        return result

# Usage (no compilation needed to run, just to optimize)
program = PaperSectionModule()
output = program(
    topic="Whitened particle-EM for POGPN inference",
    equations=r"\mathcal{L}_{ELBO} = \mathbb{E}_{q}[\log p(y|f)] - KL[q||p]",
    experiment="GPyTorch implementation, 50 inducing points, RBF kernel"
)
print(output.methods_tex)
```

#### Example 2: Literature review with citations

```python
class LitReviewPipeline(dspy.Module):
    def __init__(self):
        self.search   = dspy.ReAct("topic -> relevant_papers: list", tools=[search_semantic_scholar])
        self.summarize = dspy.ChainOfThought("topic, papers -> synthesis, gaps: list")
        self.write    = dspy.Predict("topic, synthesis, gaps -> review_paragraph: str")
    
    def forward(self, topic):
        papers    = self.search(topic=topic)
        synthesis = self.summarize(topic=topic, papers=papers.relevant_papers)
        review    = self.write(
            topic=topic,
            synthesis=synthesis.synthesis,
            gaps=synthesis.gaps
        )
        return review

# Compile against your own papers as training data
optimizer = dspy.MIPROv2(metric=citation_accuracy_metric)
compiled  = optimizer.compile(LitReviewPipeline(), trainset=your_papers)
```

---

### Integrating DSPy with LangGraph and External Agents

DSPy and LangGraph solve different but complementary problems. LangGraph manages **workflow orchestration** (which step runs next, how state flows, how to handle branching). DSPy manages **LLM call quality** (how to prompt the LLM within each step).

The integration pattern: put DSPy modules *inside* LangGraph nodes.

```python
from langgraph.graph import StateGraph
from typing import TypedDict
import dspy

# Your DSPy modules (pre-compiled)
paper_writer  = PaperSectionModule()
paper_writer.load("compiled_writer.json")

lit_reviewer = LitReviewPipeline()
lit_reviewer.load("compiled_reviewer.json")

# LangGraph state
class PaperState(TypedDict):
    topic: str
    equations: str
    literature: str
    methods_section: str
    introduction: str

# Nodes wrap DSPy calls
def literature_node(state: PaperState) -> PaperState:
    result = lit_reviewer(topic=state["topic"])
    return {**state, "literature": result.review_paragraph}

def methods_node(state: PaperState) -> PaperState:
    result = paper_writer(
        topic=state["topic"],
        equations=state["equations"],
        experiment="standard setup"
    )
    return {**state, "methods_section": result.methods_tex}

# Build graph
workflow = StateGraph(PaperState)
workflow.add_node("literature", literature_node)
workflow.add_node("methods",    methods_node)
workflow.add_edge("literature", "methods")
workflow.set_entry_point("literature")

app = workflow.compile()
result = app.invoke({"topic": "POGPN", "equations": r"...", "literature": "", "methods_section": "", "introduction": ""})
```

**Why this matters:** When you compile the DSPy modules separately, you can improve them independently without changing the LangGraph workflow. The workflow structure is stable; the prompt quality is separately optimizable.

For Cursor and Claude Code, the pattern is similar: DSPy handles the prompt optimization layer for specific sub-tasks (e.g., "generate a LaTeX equation from this description"), while the outer agent loop (Hermes, Claude Code) orchestrates which sub-task to call.

---

## Part II — GEPA

### What GEPA Is and Where It Came From

**GEPA** stands for **Genetic-Pareto**. It is a prompt and text optimizer introduced in the paper *"GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"* (Agrawal et al., 2025, arXiv:2507.19457), which received an ICLR 2026 Oral award.

It was released simultaneously as:
- A standalone library: `gepa-ai/gepa` on GitHub (MIT licensed)
- A native DSPy optimizer: `dspy.GEPA` (which wraps the gepa library)

The core claim is that you can replace reinforcement learning for prompt/agent optimization with a cheaper, faster, more interpretable approach: use an LLM to read execution traces, figure out why a variant failed, and propose targeted improvements. Repeat.

The Shopify CEO quoted the repo: *"Both DSPy and (especially) GEPA are currently severely underhyped in the AI context engineering world."*

---

### Why GEPA Is Different from RL

Traditional reinforcement learning for LLM optimization (e.g., GRPO, PPO) works like this:

```
Run policy → Get scalar reward → Compute gradient → Update weights
```

The problems:
- **Sparse rewards:** The scalar collapses rich failure information (a 400-word output with one citation error gives the same low score as a completely wrong output)
- **Expensive rollouts:** GRPO needs 5,000–25,000+ evaluations to converge
- **Requires parameter access:** You need to modify model weights (GPU required)
- **Low interpretability:** You can't read a gradient to understand why something failed

GEPA's approach:

```
Run candidate → Collect full execution trace (errors, reasoning, outputs)
     → LLM reads the trace and says "the system failed because X"
     → LLM proposes a targeted fix: "change Y in the prompt to Z"
     → New candidate is generated with that targeted fix
     → Evaluate → Select via Pareto front → Repeat
```

The key differences:
- Uses **language** as the learning signal, not scalars
- Requires only **100–500 evaluations** (35× cheaper than GRPO)
- Works with **API-only models** (no GPU, no weight access)
- Produces **interpretable improvement logs** ("the prompt failed because it didn't specify Vancouver citations")

---

### The Core Algorithm: Reflect → Mutate → Select

Here is the full GEPA loop:

```
INITIALIZATION
├── Start with an initial population of prompt candidates (often just 1: your original prompt)
└── Set Pareto frontier = empty

FOR each iteration:
│
├── EVALUATE
│   ├── Run each candidate on the eval dataset
│   ├── Collect (score, execution_trace, feedback) for each
│   └── Update Pareto frontier with non-dominated candidates
│
├── REFLECT
│   ├── For each failed/low-scoring candidate:
│   │   ├── Pass the full execution trace to the LLM
│   │   ├── Ask: "Why did this fail? What went wrong specifically?"
│   │   └── LLM returns a natural-language diagnosis
│   └── Aggregate diagnoses across examples to find common failure modes
│
├── MUTATE
│   ├── For each candidate to evolve (sampled from Pareto frontier):
│   │   ├── Pass: current prompt + diagnosis + few winning examples
│   │   ├── Ask LLM: "Propose a targeted improvement"
│   │   └── Generate N mutation proposals
│   └── New candidate pool = mutations + Pareto frontier survivors
│
└── SELECT
    ├── Evaluate new candidates
    ├── Update Pareto frontier (keep non-dominated candidates on all objectives)
    └── Discard dominated candidates

OUTPUT: The prompt/program variant on the Pareto frontier with best target metric
```

The mutation step is guided by the diagnosis. This is what makes GEPA sample-efficient: instead of randomly trying prompt variants, it reads the error and proposes a fix.

---

### Pareto-Aware Selection

Why Pareto and not just "pick the best"?

In practice, you often optimize against **multiple objectives simultaneously**:
- Quality (accuracy, relevance)
- Cost (number of tokens used, API cost)
- Latency (response time)
- Safety (no harmful content)

A Pareto front keeps every candidate that is not dominated — i.e., for which no other candidate is better on all objectives simultaneously. This prevents GEPA from collapsing to a single mode.

```
Example Pareto front (2 objectives: quality vs. cost):

Quality
  |   
1.0|        *  (Pareto optimal: high quality, high cost)
   |     *     (Pareto optimal: medium quality, medium cost)
0.5|  *        (Pareto optimal: low quality, low cost)
   |___________
      Low     High    Cost

All three * are on the Pareto front. None dominates the others.
A fourth point at (medium quality, high cost) would be dominated by the first two
and would be removed.
```

This gives you a menu of candidates at different quality/cost tradeoffs, rather than a single output.

---

### The Feedback Function

The feedback function is what makes GEPA's reflective step possible. It is the key customization point.

```python
def my_feedback_function(gold, pred, trace=None, pred_name=None) -> tuple[float, str]:
    """
    Returns: (score, feedback_string)
    
    score: float 0.0-1.0, how good this prediction was
    feedback: natural language explanation of what went wrong (or right)
              This is what GEPA passes to the LLM for reflection.
    """
    # Evaluate correctness
    score = 0.0
    feedback_parts = []
    
    if pred.citations_match(gold.citations):
        score += 0.5
    else:
        feedback_parts.append(
            f"Citation format was wrong. Expected Vancouver style like '[1] Smith et al. (2024)',"
            f" got '{pred.citations[:50]}'"
        )
    
    if pred.word_count() in range(200, 300):
        score += 0.3
    else:
        feedback_parts.append(
            f"Word count {pred.word_count()} outside target range 200-300."
        )
    
    if pred.uses_passive_voice():
        score += 0.2
    else:
        feedback_parts.append("Should use passive voice throughout.")
    
    feedback = " ".join(feedback_parts) if feedback_parts else "Looks good."
    return score, feedback
```

The richer and more specific your feedback string, the better GEPA can direct mutations. A feedback string like *"The citation '[Smith2024]' uses the wrong format — it should be '[1]' in Vancouver style"* produces much better mutations than a scalar `0.6`.

---

### GEPA Inside DSPy: dspy.GEPA

`dspy.GEPA` is a drop-in replacement for any DSPy optimizer. It wraps the gepa library and adds trace integration.

```python
import dspy

# Define feedback metric (returns (score, feedback_str))
def citation_feedback(example, prediction, trace=None):
    score, text = evaluate_citation_quality(prediction)
    return score, text  # GEPA uses the text for reflection

# Configure GEPA
gepa_optimizer = dspy.GEPA(
    metric=citation_feedback,
    num_iterations=10,      # How many evolve-evaluate cycles
    population_size=5,      # Candidates on Pareto front at once
    num_mutations=3,        # Mutations per selected candidate
)

# Compile — same API as any other DSPy optimizer
compiled = gepa_optimizer.compile(
    BayesianOptPaper(),
    trainset=trainset
)

# Save the evolved program
compiled.save("gepa_evolved_paper_writer.json")
```

GEPA's advantage over MIPROv2 for agentic programs: it can read multi-step traces including tool calls, API errors, and intermediate reasoning. MIPROv2 works best on single-module programs; GEPA shines on complex pipelines where failures are distributed across multiple steps.

---

### GEPA Standalone: The gepa-ai/gepa Library

GEPA is not just a DSPy optimizer. It is a general framework for evolving any textual system parameter. The library structure:

```
gepa/
├── core/
│   ├── evolver.py        # Main evolution loop
│   ├── reflection.py     # LLM-based diagnosis of failures
│   ├── mutation.py       # Prompt mutation proposals
│   └── pareto.py         # Pareto frontier management
├── adapters/
│   ├── dspy_adapter.py   # Full DSPy program evolution
│   ├── langchain_adapter.py
│   ├── mcp_adapter.py    # MCP tool optimization
│   ├── rag_adapter.py    # RAG pipeline optimization
│   └── terminal_bench.py # External agentic pipelines
└── examples/
    ├── math_reasoning.py
    ├── code_generation.py
    └── agent_architecture.py
```

#### Minimal standalone usage

```python
from gepa import GEPA, TextComponent

# Define what you want to optimize
component = TextComponent(
    text="You are a helpful assistant. Answer concisely.",
    name="system_prompt"
)

# Define your evaluation function (returns score, feedback)
def evaluate(prompt_text: str, test_cases: list) -> tuple[float, str]:
    score = run_test_cases(prompt_text, test_cases)
    feedback = diagnose_failures(prompt_text, test_cases)
    return score, feedback

# Run evolution
gepa = GEPA(
    components=[component],
    metric=evaluate,
    num_iterations=15,
    model="anthropic/claude-sonnet-4-20250514"  # The reflection LLM
)
best = gepa.run(test_cases=your_test_cases)
print(best.text)  # The evolved system prompt
```

---

### Available Adapters

The gepa-ai/gepa library ships with adapters for the most common use cases:

| Adapter | What It Optimizes | Status |
|---|---|---|
| **DSPy Full Program** | Entire DSPy programs (signatures + modules + control flow) | ✅ Stable |
| **LangChain** | Any LangChain pipeline: single-turn, tool-using agents, LangGraph graphs | ✅ Stable |
| **Generic RAG** | Query reformulation, context synthesis, answer generation, document reranking | ✅ Stable |
| **MCP Adapter** | MCP tool descriptions and system prompts; local stdio + remote SSE/HTTP servers | ✅ Stable |
| **TerminalBench** | External agentic pipelines (agents running in a terminal/subprocess) | ✅ Stable |
| **AnyMaths** | Mathematical problem-solving and reasoning tasks | ✅ Community |
| **Google ADK** | Google Agent Development Kit / Gemini Enterprise | ✅ Stable |

**MCP Adapter detail** (most relevant to your setup):

```python
from gepa.adapters.mcp import MCPAdapter

# Optimize tool descriptions for a local Hermes MCP server
adapter = MCPAdapter(
    server_config={"type": "stdio", "command": "hermes", "args": ["mcp", "serve"]},
    tools_to_optimize=["search_papers", "write_latex", "compile_tex"],
    feedback_fn=your_feedback_function
)

evolved_descriptions = adapter.run(num_iterations=10)
# evolved_descriptions contains better tool names/descriptions that help
# the LLM call them more accurately and effectively
```

---

### Key Benchmarks and Results

From the paper and the gepa-ai/gepa README:

| Task | Baseline | With GEPA | Notes |
|---|---|---|---|
| AIME 2025 (GPT-4.1 Mini) | ~23% (base) | +10% absolute | Single ChainOfThought module |
| ARC-AGI agent accuracy | 32% | 89% | Via architecture discovery |
| Databricks enterprise agents | Claude Opus 4.1 | 90× cheaper with open-source | Prompt-only optimization |
| RL comparison (GRPO) | 5,000–25,000 evaluations | 100–500 evaluations | 35× more sample-efficient |
| Cloud scheduling policy | Expert heuristics | 40.2% cost savings | Non-LLM system optimization |
| Code backdoor detection | 67% (basic DSPy CoT) | 93% | DSPy Full Program adapter |

The 90× cheaper result (open-source models + GEPA beating Claude Opus 4.1) is the headline claim: once prompts are evolved correctly, you can often swap to a much cheaper model and maintain quality.

---

## Part III — Hermes Agent Self-Evolution

### What the Repo Does

`NousResearch/hermes-agent-self-evolution` is a wrapper around DSPy + GEPA specifically for evolving Hermes Agent's own components. It is not a general-purpose optimizer — it is a pipeline that takes Hermes's internal files (skills, tool descriptions, system prompts, code) and produces measurably better versions of them through evolutionary search.

The key claim: **no GPU training required.** Everything is API-based text mutation. A full optimization run costs approximately $2–10.

The repo has 3.3k GitHub stars as of the time of this writing, indicating significant community interest.

---

### Repository Structure

```
hermes-agent-self-evolution/
├── datasets/           # Eval datasets for each optimization target
│   ├── skills/         # Task examples for skill file evaluation
│   └── system_prompts/ # Benchmark tasks for system prompt evaluation
│
├── evolution/          # The core optimization code
│   ├── skills/
│   │   └── evolve_skill.py    # Main CLI for skill evolution
│   ├── tools/
│   │   └── evolve_tools.py    # (Planned) Tool description evolution
│   └── system/
│       └── evolve_system.py   # (Planned) System prompt evolution
│
├── reports/            # Generated optimization reports (stored per run)
├── tests/              # Test suite (must pass 100% before any PR)
├── generate_report.py  # Generates HTML/markdown summary of a run
├── PLAN.md             # Full architecture and roadmap
├── pyproject.toml      # Dependencies: dspy, gepa, pytest, click
└── README.md
```

---

### The Five-Phase Optimization Plan

The repo is organized around a phased rollout:

| Phase | Target | Engine | Status |
|---|---|---|---|
| **Phase 1** | Skill files (`~/.hermes/skills/*.md`) | DSPy + GEPA | ✅ Implemented |
| **Phase 2** | Tool descriptions (the short strings that tell the LLM what each tool does) | DSPy + GEPA | 🔲 Planned |
| **Phase 3** | System prompt sections | DSPy + GEPA | 🔲 Planned |
| **Phase 4** | Tool implementation code | Darwinian Evolver | 🔲 Planned |
| **Phase 5** | Continuous improvement loop | Automated pipeline | 🔲 Planned |

**Why this order matters:** Skills are self-contained markdown files with clear inputs/outputs, making them easy to define evaluation metrics for. Tool descriptions are shorter but affect every tool call globally. System prompt sections affect everything and are the highest-leverage but hardest to evaluate. Code evolution requires a separate engine (Darwinian Evolver) because mutation of code is structurally different from mutation of text.

---

### Engine 1: DSPy + GEPA (Skill and Prompt Evolution)

The skill evolution loop in detail:

```
INPUT: A Hermes skill file (e.g., github-code-review.md)
       ├── Contains: name, description, procedure steps, examples
       └── Measured by: how well an agent following it completes benchmark tasks

STEP 1 — BUILD EVAL DATASET
├── Option A (--eval-source synthetic):
│   ├── Use an LLM to generate N representative tasks that should trigger this skill
│   ├── Generate expected outcomes for each task
│   └── Store as eval examples in datasets/skills/<skill_name>/
│
└── Option B (--eval-source sessiondb):
    ├── Read ~/.claude/history.jsonl (Claude Code sessions)
    ├── Read ~/.copilot/session-state/*/events.jsonl (Copilot sessions)
    ├── Filter for conversations where this skill would have been relevant
    │   (two-stage: keyword heuristic → LLM-as-judge scoring via DSPy)
    └── Use real past interactions as eval examples

STEP 2 — DEFINE METRIC
├── For each eval example:
│   ├── Run Hermes with the current skill on the task
│   ├── Score the result (task completion rate, correctness)
│   └── Generate feedback: "The skill failed because it didn't specify X"
└── GEPA receives (score, feedback_text) for each example

STEP 3 — GEPA EVOLUTION LOOP
├── Current skill file = initial candidate
├── For iteration in range(num_iterations):
│   ├── GEPA evaluates current Pareto population on all eval examples
│   ├── Reads traces from failed examples
│   ├── Reflection LLM diagnoses: "Step 3 of the skill says X, but
│   │   the agent needs to do Y first — the ordering is wrong"
│   ├── Mutation LLM proposes: "Move step 3 after step 1, add clarification
│   │   that the code must compile before proceeding"
│   ├── Generate N new skill variants with this fix applied
│   └── Update Pareto front (quality vs. skill file size)
│
STEP 4 — GUARDRAIL CHECKS
├── pytest tests/ -q (must pass 100%)
├── Skill size ≤ 15KB
├── Caching compatibility check (no mid-conversation state)
└── Semantic preservation check (LLM judge: does this still do what it claims?)

OUTPUT: Evolved SKILL.md + optimization report + PR against hermes-agent
```

#### Running it

```bash
# Evolve the github-code-review skill using synthetic eval data
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source synthetic

# Evolve using real Claude Code/Copilot sessions
python -m evolution.skills.evolve_skill \
    --skill latex-document-writing \
    --iterations 15 \
    --eval-source sessiondb

# View the report
python generate_report.py --skill github-code-review
```

---

### Engine 2: Darwinian Evolver (Code Evolution)

Phase 4 uses a different engine: the `imbue-ai/darwinian_evolver`. This is for evolving *Python code*, not markdown text.

The reason for a separate engine: GEPA mutates text via LLM reflection, which works beautifully for prompts and structured documents. But code has **hard correctness constraints** (it must execute, compile, pass tests) that require a different selection mechanism.

The Darwinian Evolver treats code as organisms:
- Each candidate is a git commit (an "organism")
- Mutations are code edits (generated by an LLM)
- Selection is based on test suite pass rate + performance benchmarks
- Crossover combines code sections from two high-performing organisms

This is analogous to genetic algorithms but operates at the level of git history. Every candidate organism exists as an actual runnable version of the code, tracked in git.

For Hermes self-evolution, Phase 4 plans to use this to evolve the Python implementation of tools — not just their descriptions (Phase 2) but their actual behavior.

---

### Eval Data Strategy: Synthetic vs SessionDB

This is architecturally important and worth understanding in detail.

#### Synthetic eval

An LLM is asked to generate realistic tasks that would trigger the skill, plus expected outcomes. This is fast and always available, but the tasks may not reflect real usage patterns.

```python
# What synthetic generation does internally:
prompt = f"""
Skill name: {skill_name}
Skill description: {skill_description}

Generate 20 realistic tasks that a developer would ask Hermes to do
where this skill would be relevant. For each task:
1. Write the user's natural language request
2. Write what a successful outcome looks like
3. Write what a failure looks like (for use in evaluation)

Format as JSON list of {{request, success_criteria, failure_indicators}}.
"""
```

#### SessionDB eval

The real session history path reads actual conversations from your tools:

```
~/.claude/history.jsonl       ← Claude Code user prompts
~/.hermes/sessions/           ← Hermes's own session store
~/.copilot/session-state/*/events.jsonl  ← GitHub Copilot
```

The filtering pipeline has two stages:

**Stage 1 — Keyword heuristic (fast, cheap):**
- Scan each conversation for keywords related to the skill
- Discard obvious mismatches (a conversation about Python dependencies when the skill is about LaTeX)

**Stage 2 — LLM-as-judge (slower, expensive, accurate):**
- For conversations that pass Stage 1, ask an LLM: "Would the `latex-document-writing` skill have been relevant to this conversation? Score 0.0–1.0."
- Implemented as a DSPy module (so it too can be optimized)
- Keep conversations with judge score > threshold

The result: a curated eval dataset derived from your actual usage patterns, reflecting your specific workflows, writing style, and tool preferences.

**Important caveat:** As of the initial release, the Claude Code and Copilot session readers were filed as a feature proposal on top of the already-declared `--eval-source sessiondb` option. Verify the reader is fully implemented in your installed version before relying on it.

---

### Guardrails and Safety Constraints

Every evolved variant must pass a set of hard constraints before it can become a PR:

| Constraint | Check | Rationale |
|---|---|---|
| Full test suite | `pytest tests/ -q` passes 100% | Evolved variants must not break existing functionality |
| Size limit | Skills ≤ 15KB, tool descriptions ≤ 500 chars | Prevents context-bloating over-specified skills |
| Caching compatibility | No mid-conversation state changes | Hermes caches skills; a skill that changes meaning mid-session breaks the cache |
| Semantic preservation | LLM judge: "Does this still accomplish what the original claimed?" | Prevents skill drift where the evolution changes the skill's purpose |
| Human review | All changes go through PR, never direct commit | Final human-in-the-loop gate |

The semantic preservation check is the most interesting: it asks an LLM to compare the original and evolved skill and flag if the evolved version has drifted to do something fundamentally different. This prevents GEPA from "gaming" the metric by producing a skill that scores well on the eval but doesn't match the original intent.

---

### The PR-Based Deployment Model

The repo is designed to produce **pull requests against hermes-agent**, not to directly modify your live Hermes installation.

```
hermes-agent-self-evolution (this repo)
        │
        │ runs GEPA evolution
        ▼
Evolved SKILL.md + diff report
        │
        │ opens PR against
        ▼
hermes-agent (the main Hermes repo)
        │
        │ human reviews and approves
        ▼
Merged → available in your next Hermes install
```

This is a conservative deployment model. The evolved content goes through human review before it affects the live agent. This is appropriate given that skill changes affect all future behavior of the agent.

For personal use, you can short-circuit this: copy the evolved skill directly to `~/.hermes/skills/` and the live Hermes will pick it up.

---

## Part IV — Integration Patterns

### DSPy + LangGraph

The integration principle is clean: **LangGraph owns the workflow, DSPy owns the LLM calls.** Neither framework constrains the other.

```python
# Pattern 1: DSPy module as a LangGraph node
from langgraph.graph import StateGraph
import dspy

# Pre-compile DSPy programs offline
# (don't run optimizer during the live graph execution)
lit_reviewer = LitReviewPipeline()
lit_reviewer.load("compiled_lit_reviewer.json")

def lit_review_node(state):
    result = lit_reviewer(topic=state["topic"])
    return {"literature": result.review_paragraph}

# Pattern 2: DSPy for dynamic routing decisions inside LangGraph
class RouteDecision(dspy.Signature):
    "Decide which section to write next based on the current paper state."
    paper_state: str = dspy.InputField()
    next_section: str = dspy.OutputField(desc="One of: introduction, methods, results, discussion")
    reason: str = dspy.OutputField()

router = dspy.Predict(RouteDecision)

def routing_node(state):
    decision = router(paper_state=str(state))
    return {"next": decision.next_section}
```

---

### GEPA + MCP Tools

One of the most powerful GEPA adapters for your use case is the MCP adapter. The idea: the string description of each MCP tool is a textual parameter that can be evolved to help the LLM use it more correctly.

```python
from gepa.adapters.mcp import MCPAdapter

# Your tool description (initial)
initial_description = """search_semantic_scholar: Search for academic papers."""

# Better (evolved by GEPA):
evolved_description = """
search_semantic_scholar: Search for academic papers by topic, author, or keyword.
Returns titles, abstracts, and DOIs.
Use for: finding prior work, verifying citations, checking novelty.
Do NOT use for: fetching full paper text (use fetch_pdf for that).
Query format: natural language or author:name or DOI:10.xxx/yyy
"""

# GEPA finds the better description automatically by observing
# how often the LLM uses the tool correctly given each description
```

---

### Hermes Self-Evolution + Your Cursor/Claude Code Setup

Here is the concrete architecture for your research workflow:

```
PHASE 1 — FOUNDATION (do today)
│
├── Install Hermes + configure Anthropic provider (your Claude sub)
├── Set up Hermes ACP server in VS Code
└── Create a shared Hindsight bank_id for Hermes + Claude Code

PHASE 2 — SKILLS FROM YOUR WORK (recurring)
│
├── When you do important agentic work in VS Code via Hermes ACP,
│   skill creation fires automatically (5+ tool calls)
│
├── Weekly cron: python -m evolution.skills.evolve_skill \
│       --skill <skill_name> \
│       --eval-source sessiondb \
│       --iterations 10
│   (reads your Claude Code history and refines existing skills)
│
└── For your LaTeX writing style specifically:
    ├── Create a skill file manually: latex-bo-paper-style.md
    │   with your style preferences, citation format, notation conventions
    ├── Run GEPA to evolve it from your actual .tex files as examples
    └── Pin it (pinned skills are never touched by the curator)

PHASE 3 — OPTIMIZATION (when you have a specific workflow to improve)
│
└── Identify a workflow that should be a skill
    (e.g., "compile LaTeX, read errors, fix, recompile — 5+ steps")
    ├── Run this through Hermes in VS Code (native skill creation fires)
    ├── Then run GEPA to refine the created skill
    └── Result: a highly optimized, personalized skill for your exact workflow
```

---

### The Full Envisioned Architecture

When all components are connected, this is what the system looks like:

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR EDITOR (VS Code)                     │
│                                                              │
│  ┌──────────────────┐        ┌─────────────────────────┐    │
│  │  Cursor / Copilot │        │   Hermes ACP panel      │    │
│  │  (inline edits,   │        │   (agentic tasks,       │    │
│  │   tab-complete)   │        │    multi-step work)     │    │
│  └──────────────────┘        └─────────────────────────┘    │
│                                        │                     │
└────────────────────────────────────────│─────────────────────┘
                                         │ ACP (JSON-RPC stdio)
                                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    HERMES DAEMON (persistent)                │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Agent Loop                              │    │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  │    │
│  │  │ DSPy     │  │ Skill Memory │  │ Working      │  │    │
│  │  │ Programs │  │ (~/.hermes/  │  │ Memory       │  │    │
│  │  │ (LLM     │  │  skills/)    │  │ (context     │  │    │
│  │  │  calls)  │  │              │  │  window)     │  │    │
│  │  └──────────┘  └──────────────┘  └──────────────┘  │    │
│  │                       │                              │    │
│  │        5+ tool calls → skill_manage fires            │    │
│  └───────────────────────│──────────────────────────────┘    │
│                           │                                   │
└───────────────────────────│───────────────────────────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
          ▼                 ▼                  ▼
  ┌───────────────┐  ┌────────────┐  ┌──────────────────┐
  │  HINDSIGHT    │  │  ANTHROPIC │  │  MCP SERVERS     │
  │  (shared      │  │  API       │  │  (GitHub, arxiv, │
  │   memory      │  │  (your     │  │   filesystem,    │
  │   bank_id)    │  │   Claude   │  │   latexmk, etc.) │
  │               │  │   sub)     │  │                  │
  │ facts, prefs, │  └────────────┘  └──────────────────┘
  │ writing style │
  └───────────────┘
          ▲
          │ shared bank_id
          │ (auto-recall + extraction)
          │
  ┌───────────────────┐
  │  CLAUDE CODE      │
  │  (your other      │
  │   sessions)       │
  │                   │
  │  ~/.claude/       │
  │  history.jsonl  ──┼──► sessiondb eval source
  └───────────────────┘      (weekly GEPA run)

NIGHTLY / WEEKLY:
┌────────────────────────────────────────────────────────┐
│  hermes-agent-self-evolution                           │
│                                                        │
│  python -m evolution.skills.evolve_skill               │
│    --eval-source sessiondb   ← reads Claude Code logs  │
│    --iterations 15                                     │
│                                                        │
│  GEPA → reflects on failures → mutates skills          │
│       → evolved SKILL.md in ~/.hermes/skills/          │
└────────────────────────────────────────────────────────┘
```

---

## Part V — Presentation Guide

This section provides talking points, key claims, and a suggested flow for a presentation on these three systems.

### Suggested Presentation Flow (45 min)

| Segment | Time | Key point |
|---|---|---|
| The problem with prompting | 5 min | Prompts are programs. Hard-coding them is technical debt. |
| DSPy fundamentals | 10 min | Signatures/Modules/Optimizers. The compilation analogy. |
| Live DSPy demo | 5 min | Show a ChainOfThought + MIPROv2 compile |
| Why GEPA | 5 min | RL is expensive and opaque. GEPA uses language as the signal. |
| GEPA algorithm | 10 min | Walk the Reflect→Mutate→Select loop with a concrete example |
| Hermes self-evolution | 7 min | Skills as the optimization target. The 5-phase plan. SessionDB. |
| Your architecture | 3 min | The integration diagram. What you're building. |

### Key Claims to Drive Discussion

1. **"The best prompt is a compiled artifact, not a hand-written string."** DSPy's optimizer finds better instructions than humans can write by hand, because it tries more combinations and tunes the metric directly.

2. **"RL needs a scalar. GEPA reads the error message."** This is why GEPA is 35× more sample-efficient: a scalar tells you the answer was wrong; a trace tells you exactly where in the reasoning chain it went wrong.

3. **"Skill files are textual programs with measurable behavior."** This is the insight that makes Hermes self-evolution possible: a markdown skill file can be evaluated just like a prompt — run it on benchmark tasks, score the result, improve the text.

4. **"Shared memory + private skills."** Hindsight handles the shared cross-session knowledge (facts, preferences). Skills handle the procedural knowledge (how to do things). These are fundamentally different and should be stored differently.

5. **"No GPU required, $2–10 per optimization run."** For researchers without compute budgets, this is the compelling argument for GEPA over RL-based approaches.

### Common Questions and Answers

**Q: Is this just prompt engineering with extra steps?**
A: No. Prompt engineering is manual iteration. DSPy + GEPA is automated search over a structured space, with a formal metric, and convergence guarantees. The difference is the same as between grid search and Bayesian optimization.

**Q: Does DSPy lock me into a framework?**
A: The compiled artifact (a `.json` file) contains just the optimized prompt text. You can extract it and use it anywhere. The framework adds zero overhead at inference time.

**Q: Can GEPA optimize non-LLM systems?**
A: Yes. GEPA has been used to optimize cloud scheduling policies (40.2% cost savings) and vector graphics. Anything with a textual parameter and a measurable outcome is a target.

**Q: What's the difference between a DSPy Module and a LangChain Chain?**
A: A LangChain chain is a fixed composition of steps. A DSPy module has a `forward` method in plain Python (any control flow) plus the optimization hook. The key difference: DSPy modules are compiled — their prompts are improved by an optimizer. LangChain chains are not.

**Q: How does Hermes know when a skill is "done" evolving?**
A: GEPA stops either when `num_iterations` is exhausted, or when the Pareto front stops improving between iterations (convergence). There's no global "done" — it's budget-based.

---

## Quick Reference: Commands and Code

### DSPy
```bash
pip install dspy
```
```python
import dspy
dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-4-20250514"))

# Minimal program
predict = dspy.Predict("question -> answer")
result = predict(question="...")
```

### GEPA (standalone)
```bash
pip install gepa
```
```python
from gepa import GEPA, TextComponent
gepa = GEPA(components=[TextComponent(text="...", name="prompt")], metric=my_metric)
best = gepa.run(test_cases=[...])
```

### GEPA via DSPy
```python
optimizer = dspy.GEPA(metric=feedback_fn, num_iterations=10)
compiled = optimizer.compile(my_program, trainset=data)
compiled.save("evolved.json")
```

### Hermes Self-Evolution
```bash
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# Evolve a skill
python -m evolution.skills.evolve_skill --skill <name> --iterations 10 --eval-source synthetic

# Or from real session data
python -m evolution.skills.evolve_skill --skill <name> --iterations 15 --eval-source sessiondb
```

### Hermes ACP in VS Code
```json
// ~/.cursor/mcp.json or VS Code settings
{
  "acpClient.agents": [
    {
      "name": "hermes-agent",
      "command": "hermes",
      "args": ["acp"]
    }
  ]
}
```

---

*Report generated: June 2026. Sources: DSPy documentation (dspy.ai), GEPA paper (arXiv:2507.19457), gepa-ai/gepa GitHub, NousResearch/hermes-agent-self-evolution GitHub, Hermes Agent documentation (hermes-agent.nousresearch.com).*

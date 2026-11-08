# Senapathy — Implementation Plan

Companion to `SENAPATHY.md` (the design doc). That document says *what* and *why*.
This one says *in what order*, and hands each step to a Claude agent as an
executable prompt.

---

## 1. How this plan works

Work is cut into numbered phases. Each phase lives in `phases/Pnn-name.md` and is a
**self-contained prompt**: an agent (or you) can execute it with no context beyond
that file, `CLAUDE.md`, and the code already in the repo.

Every phase file has the same nine sections:

| Section | For | Purpose |
|---|---|---|
| 1 GOAL | both | one sentence |
| 2 WHY THIS PHASE | both | what it unlocks; why here and not earlier |
| 3 PRECONDITIONS | agent | what must already exist |
| 4 DELIVERABLES | agent | exact file paths |
| 5 SPEC | agent | signatures, behaviour, edge cases |
| 6 ACCEPTANCE TEST | agent | the complete test file, verbatim |
| 7 OUT OF SCOPE | agent | explicit prohibitions |
| 8 DEMO | you | the command that shows it works |
| 9 LEARNING | you | the concept, what to notice, the experiment |

### The rule that makes this work

**The acceptance test is written before the implementation, and it's in the prompt.**

A phase is complete when its test passes and no earlier test regressed. That makes
"done" machine-checkable instead of a judgement call, which is what lets you hand a
phase to Claude Code and walk away.

This is the same principle as the product itself — oracle before patch. We're
applying our own design philosophy to our build process.

### Three supporting rules

**Contracts are frozen in P00.** Every artifact schema and protocol is defined once,
before any logic. Later phases import them and never redefine a shape. Without this,
P13 renames a field and P04–P12 need rework.

**`CLAUDE.md` carries the invariants.** Don't paste the 1,200-line design doc into
every session. The repo-level `CLAUDE.md` is inherited automatically by every Claude
Code session and states the rules an agent must not violate.

**Every phase declares what NOT to build.** The dominant failure of agent-built
projects is scope creep across phases: you ask for the LLM client and get half an
agent loop bolted on. Section 7 exists to prevent that.

### How to run a phase

```bash
cd senapathy
claude                       # in the repo, so CLAUDE.md is picked up
> Execute phases/P05-environment.md
```

For HAND-WRITE phases, instead:

```bash
> Read phases/P02-agent-loop.md. Don't implement it. Walk me through the
> mechanics, then let me write it and review what I produce.
```

One phase per sitting. One commit per phase. Never two phases in one session — the
whole point is that each one has a checkpoint you can trust.

---

## 2. The thin thread — first working version

The full plan is 20 phases. The **critical path to `senapathy run` working end to
end** is 13 of them. Everything else is deliberately deferred until after you've
seen it work once.

| Day | Phases | You can run |
|---|---|---|
| 1 | P00 contracts · P01 LLM client · P02 agent loop | unit tests green |
| 2 | P03 read tools · P04 comprehension agent | `senapathy understand` |
| 3 | P05 environment · P06 workload · P07 benchmark | `senapathy baseline` |
| 4 | P09 diagnosis agent | `senapathy diagnose` |
| 5 | P12 patch loop | `senapathy patch` |
| 6 | P13 oracle · P14 measure | `senapathy verify` |
| 7 | P15 gate + PR + report · thin `run.py` | **`senapathy run`** |

Deferred out of the thread, and why:

| Phase | Why it waits |
|---|---|
| P08 red flags | Fold a crude version into P09; extract properly later |
| P10 `run_probe` | Diagnosis works on static evidence first |
| P11 plan + risk tiers | P09 → P12 directly; one candidate, no tiering |
| P16 orchestrator FSM | `run.py` calls functions in order until you know the real shape |
| P17–P20 | Post-MVP: evals, code index, Docker, repo profile |

**Do not compress further than this.** Each of the 13 is load-bearing; skip any and
what you have is a demo rather than a product.

---

## 3. Full phase list

### Milestone A — the agent, no infrastructure

| Phase | Name | Mode | Output |
|---|---|---|---|
| P00 | Scaffold, contracts, trace, artifact store | generate | typed foundation |
| P01 | LLM client — raw Messages API, retries, cost, cassettes | **HAND-WRITE** | `LLMClient` |
| P02 | Agent loop, tool registry, dispatcher, budgets | **HAND-WRITE** | `AgentLoop` |
| P03 | Read tools — `fs_read`, `code_search`, `list_dir` | **HAND-WRITE** | tool set |
| P04 | Comprehension agent + citation validation | mixed | `EndpointModel` |

Milestone demo: `senapathy understand --repo ../crm --endpoint "POST /api/customers/search"`

### Milestone B — evidence, zero AI

| Phase | Name | Mode | Output |
|---|---|---|---|
| P05 | Environment — worktrees, build, run, health | generate | running blue/green |
| P06 | Workload — seed script, request corpus | generate | `RequestCorpus` |
| P07 | Benchmark — latency, SQL capture, digest aggregation, variance | mixed | `BaselineReport` |
| P08 | Evidence bundle + computed red flags | **HAND-WRITE** | `EvidenceBundle` |

Milestone demo: `senapathy baseline` — a genuinely useful profiler with no AI in it.

### Milestone C — diagnosis and patching

| Phase | Name | Mode | Output |
|---|---|---|---|
| P09 | Diagnosis agent — hypotheses, citations, arithmetic validation | mixed | `HypothesisSet` |
| P10 | `run_probe` — sandboxed experiment tool | generate | probe results |
| P11 | Plan phase — candidates, risk tiers, selection | mixed | `CandidatePlan` |
| P12 | Patch loop — `apply_patch`, build, test, checkpoints | mixed | `Patch` |

### Milestone D — trust and ship

| Phase | Name | Mode | Output |
|---|---|---|---|
| P13 | Oracle — record, canonicalise, compare, hash | **HAND-WRITE** | `EquivalenceVerdict` |
| P14 | Interleaved A/B measurement + significance | mixed | `PerfReport` |
| P15 | Gate rules, PR creation, artifact-backed report | **HAND-WRITE** (gate) | PR |
| P16 | Orchestrator FSM, trace viewer, resume | mixed | `senapathy run` |

### Milestone E — reliability, post-MVP

| Phase | Name | Mode | Output |
|---|---|---|---|
| P17 | Scenario suite + eval harness + metrics | mixed | success rates |
| P18 | Tree-sitter code index — *retrofit, then measure the savings* | generate | `CodeIndex` |
| P19 | Docker environment, seeded image, `DataPlan` | generate | reproducible env |
| P20 | Repo profile, feasibility gate, `StackPack` seam | mixed | repo-agnostic |

---

## 4. Two ordering choices that look wrong

**The orchestrator is P16, not P00.** Until then, `run.py` calls functions in
sequence. Building the state machine before you know what the phases actually need
produces an abstraction that fights you for fifteen phases. Write the boring version;
extract the FSM once the shape is known.

**The code index is P18, not P03.** You start with ripgrep, feel the token cost in
your own bill, then retrofit tree-sitter and measure the difference. That experiment
teaches you more than the tokens cost — it's where "tool design is context design"
stops being a slogan and becomes a number you measured.

---

## 5. Why prompts are written one milestone ahead

Detailed prompts for P13 written today would be based on guesses about what P07's
`EvidenceBundle` looks like in practice. They'd be wrong, you'd fight them, and
rewriting costs more than writing fresh.

So: **full detail one milestone ahead, stubs beyond.** Milestone A is fully written.
When you finish it, we write B properly — informed by what actually happened. The
plan improves as you build instead of decaying.

---

## 6. The hand-write list

Six phases you type yourself. These are the ~800 lines that constitute the actual
AI engineering:

| Phase | Component | Why this one |
|---|---|---|
| P01 | LLM client | The LLM is a stateless function; everything else is your engineering |
| P02 | Agent loop | This *is* what an agent is. Not knowing it cold means not knowing agents |
| P03 | Read tools | Tool design is context design — the highest-leverage skill in the field |
| P08 | Red flags | Where you decide what the model gets to see. Determines everything downstream |
| P13 | Oracle | The trust boundary of the entire product |
| P15 | Gate rules | Where human/AI authority is encoded in code |

Everything else — scaffold, Docker, seeders, report templates, schemas — generate
freely. Burning weekends on boilerplate is how the project dies at week 6.

---

## 7. Progress log

Append one line per completed phase. This is your resume point after a break.

```
P00  ____-__-__  contracts frozen
P01  ____-__-__
```

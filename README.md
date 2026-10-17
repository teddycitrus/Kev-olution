<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

</div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Kev-olution</h3>
  <p align="center">
    Hypothesis triage and stagnation-triggered exploration for agentic research loops
    <br />
    <a href="https://github.com/teddycitrus/Kev-olution"><strong>Explore the docs &raquo;</strong></a>
    <br />
    <br />
    <a href="https://github.com/teddycitrus/Kev-olution/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/teddycitrus/Kev-olution/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#how-it-works">How It Works</a></li>
        <li><a href="#bring-your-own-task">Bring Your Own Task</a></li>
        <li><a href="#example-task-and-results">Example Task And Results</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Kev-olution is a harness for agentic research loops: an agent proposes hypotheses, an evaluator tests them, and the loop keeps whatever scores best. Any problem with an automatic score fits that shape, whether it is kernel tuning, hyperparameter or architecture search, query and compiler tuning, heuristic discovery, or prompt optimization.

Long-running loops like this tend to fail in the same three ways:

1. **Wasted evaluations.** Most proposed hypotheses are bad, and every one of them still pays for a full, expensive evaluation.
2. **Context rot.** Failed attempts pile up as raw traces until the agent's context is mostly noise and its reasoning degrades.
3. **Local optima.** Small edits stop helping, and the agent keeps polishing the same idea instead of changing approach.

Kev-olution addresses all three with Kev, Jared Palmer's family of small decision models. Kev answers typed questions about a state (yes/no, multiple choice, ordinal score) instead of generating text, which makes it fast and cheap enough to sit inside the loop on every cycle:

* **Hypothesis triage (Kev-9B)**: before anything is evaluated, Kev-9B scores the whole batch of proposed hypotheses in one call. Weak ones are discarded before they spend any evaluation budget
* **Stagnation-triggered exploration (Kev-27B)**: when the best score stops improving for three cycles, the stalled branch is purged and Kev-27B chooses a structurally different approach from the task's menu of rewrites, so the search jumps to a new region instead of creeping along the old one
* **Compressed trajectory memory**: every hypothesis lives in an event-sourced graph with its lineage and score. Failures are compressed into one-line notes under 280 characters, so the agent learns from dead ends without ever reading a raw log
* **Tiered evaluation with early exit**: cheap static checks first, then a small trial, then the full evaluation. Hypotheses that fall behind the current best stop early and never pay for the expensive tier
* **Deterministic replay**: Kev's answers are cached, so any recorded run replays exactly, with no network and no quota
* **Live dashboards**: a terminal dashboard and a localhost web dashboard showing the search graph, score curve, Kev telemetry, and a with/without-Kev comparison

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### How It Works

Each cycle of the loop:

1. **Propose**: the task's mutator generates a batch of hypotheses from the current best (small local edits).
2. **Filter**: free static checks reject malformed hypotheses before any model call.
3. **Triage**: Kev-9B answers one yes/no question per hypothesis ("will this beat its parent?"). Low scorers are pruned with a recorded reason.
4. **Evaluate**: survivors run through the tiered evaluator. A hypothesis whose quick-tier score is below 95% of the current best is cut before the full tier.
5. **Reflect**: failures are compressed into short notes and attached to the graph, where they feed back into the next Kev question and the next proposal.
6. **Detect stagnation**: if the best score has gained 1% or less for three straight cycles, the Paradigm Shift engine purges the stalled branch, asks Kev-27B to pick an untried structural rewrite, and evaluates that new direction. When no rewrites remain, the run stops as converged.

The loop, graph store, Kev routing, stagnation detector, shift engine, reflector, cache, and dashboards contain no task-specific logic.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Bring Your Own Task

A research problem plugs in by implementing the `ResearchTask` interface in `src/tasks/types.ts`. A task supplies four things:

| Piece | Methods | What it means for your problem |
|---|---|---|
| **Score** | `evaluate`, `metric`, `baselineScore`, `targetScore` | How a hypothesis is measured (higher is better), split into quick and full tiers |
| **Free gate** | `staticCheck` | Cheap validity checks that run before any Kev call or evaluation |
| **Hypothesis space** | `proposeLocal`, `structuralOptions`, `proposeShift` | Small edits for normal cycles, plus the menu of bigger rewrites Kev-27B chooses from on a plateau |
| **Kev wording** | `objective`, `kev.pruneInstructions`, `kev.shiftInstructions` | The plain-language questions Kev answers about your hypotheses |

Hypotheses are plain strings: source code, a JSON config, a prompt, a query plan. Pass the task to `evolve({ task, ... })` and everything else, including routing, memory, stagnation handling, and telemetry, works unchanged.

Good fits are problems with a repeatable automatic score and an evaluation that costs real time or money. Problems that need a human to judge each hypothesis are not a fit unless a model can act as the judge.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Example Task And Results

The bundled example task (`src/tasks/kernelTask.ts`) optimizes a Triton flash-attention forward kernel (B4 H32 N4096 D128 fp16). Hypotheses are kernel configurations rendered to real Triton source; the structural rewrites are block pointers, TMA tensor descriptors, warp specialization, and persistent scheduling.

Running the identical search (same seed, task, and evaluator) with and without Kev:

| | With Kev | Without Kev |
|---|---|---|
| Final score | same result | same result |
| Search cycles | 25 | 26 |
| Hypotheses proposed | 119 | 130 |
| Hypotheses discarded by Kev before evaluation | 73 | n/a |
| Evaluations run (compile + correctness) | 47 | 79 |
| Structural shifts | 3, chosen by Kev-27B | 4, fixed order |
| Estimated wall-clock (upper-end step costs) | about 1.9 h | about 2.3 h |

Kev cut evaluations by about 40% for the same final result. The example's evaluator is a performance model of the target hardware rather than physical runs, so treat its absolute scores as illustrative; the evaluation counts and Kev's decisions are what the comparison measures. The wall-clock estimate assumes 40 s to write a hypothesis, 35 s per evaluation, 20 s per full benchmark, and 1.3 s per Kev call; all four are editable in the web dashboard.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![TypeScript][TypeScript-badge]][TypeScript-url]
* [![Bun][Bun-badge]][Bun-url]
* [![Kev][Kev-badge]][Kev-url]
* [![Modal][Modal-badge]][Modal-url]
* [![Python][Python-badge]][Python-url]
* [![PyTorch][PyTorch-badge]][PyTorch-url]
* [![Triton][Triton-badge]][Triton-url]

| Layer | Tools |
|---|---|
| Research loop, agents, trajectory graph | TypeScript on Bun |
| Decision routing | Kev-9B and Kev-27B (TypeSafe SystemOne protocol), hosted gateway or self-hosted on Modal |
| Dashboards | Raw ANSI terminal UI; dependency-free HTML + SVG served by `Bun.serve` |
| Example task only | Triton, PyTorch, Python |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* [Bun](https://bun.sh) 1.4 or later
  ```sh
  npm install -g bun
  ```
* A Kev API key for the hosted gateway, or your own Kev endpoints (see the `kev-deploy` skill in `.claude/skills/`). Without a key the loop still runs offline, with heuristic triage and a fixed shift order
* Only for running the example task on real hardware: Python 3.10+, PyTorch with CUDA, Triton 3.4+, and an NVIDIA GPU

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/teddycitrus/Kev-olution.git
   cd Kev-olution
   ```

2. Install dependencies
   ```sh
   bun install
   ```

3. Create a `.env` file in the project root (Bun loads it automatically)
   ```env
   # Hosted Kev gateway
   KEV_API_KEY="your_kev_key"

   # Optional: self-hosted endpoints, one per model (both URL and key are required)
   KEV_9B_BASE_URL="https://<workspace>--kev-9b-api.modal.run"
   KEV_9B_API_KEY="your_9b_key"
   KEV_27B_BASE_URL="https://<workspace>--kev-27b-api.modal.run"
   KEV_27B_API_KEY="your_27b_key"
   ```

4. Run the checks
   ```sh
   bun run typecheck
   bun test
   ```

5. Run the example task
   ```sh
   bun run evolve
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->
## Usage

**Running your own task**

```ts
import { evolve } from "./src/harness/runner";
import { InMemoryGraphStore } from "./src/harness/inMemoryGraphStore";
import { KevClient } from "./src/harness/kevClient";
import { ParadigmShiftEngine, PlateauDetector } from "./src/harness/plateauDetector";
import { Reflector } from "./src/agents/reflector";

const task = myTask; // implements ResearchTask
const store = new InMemoryGraphStore();
const kev = KevClient.fromEnv(); // or null for offline triage

await evolve(
  {
    task,
    store,
    kev,
    reflector: new Reflector({ describe: task.describe }),
    plateau: new PlateauDetector({ window: 3, minRelativeGain: 0.01 }),
    shiftEngine: new ParadigmShiftEngine({ store, task, kev }),
  },
  { maxCycles: 40, onEvent: (e) => console.log(e.type) },
);
```

Every step emits a typed `EvolveEvent` (hypotheses proposed, Kev decisions, evaluations, plateaus, shifts), so you can log, store, or visualize a run however you like.

**Tuning the loop**

| Option | Default | What it controls |
|---|---|---|
| `candidatesPerCycle` | 6 | Hypotheses proposed per cycle |
| `keepTopK` | 3 | Max hypotheses per cycle that reach evaluation |
| `minKevScore` | 0.35 | Kev-9B confidence below which a hypothesis is discarded |
| `promotionRatio` | 0.95 | Quick-tier score needed (vs current best) to reach the full tier |
| `PlateauDetector.window` | 3 | Stalled cycles before a structural shift |
| `PlateauDetector.minRelativeGain` | 0.01 | Gain that counts as progress |

**CLI and terminal dashboard**

`bun run evolve` runs the loop with a live terminal dashboard (in a terminal at least 120 columns wide) showing the search graph, score curve, Kev telemetry, evaluation cost, and an event log. Piped output, or `--plain`, falls back to a plain event log.

| Flag | What it does |
|---|---|
| `--cycles N` | Max search cycles (default 40) |
| `--duration S` | Wall-clock budget in seconds |
| `--seed N` | Seed for hypothesis generation (default 0) |
| `--demo` | Replay Kev from the recorded cache, paced to a 45 s presentation |
| `--pace MS` | Slow evaluation down for live viewing |
| `--offline` | No Kev: heuristic triage and a fixed shift order |
| `--warmup` | Wake cold self-hosted Kev containers before the run |
| `--plain` | Plain text log instead of the dashboard |

**Web dashboard**

```sh
bun run web
```

Open [http://localhost:4173](http://localhost:4173) and pick a mode in the sidebar: replay from cache, live Kev calls, or no Kev. When a Kev run finishes, the "Time saved by Kev" panel compares it with the identical search without Kev: cycles, hypotheses, evaluations, and estimated wall-clock, with editable cost assumptions.

**Presentation demo**

```sh
bun run demo           # checks Kev readiness, then the 45 s paced run
bun run demo:offline   # same, with no Kev and no network
```

The first networked run records every Kev answer. Later runs with the same seed replay those answers from cache in under 5 ms each.

**Keeping self-hosted Kev warm**

```sh
bun run keepalive --every 180 --minutes 60
```

Self-hosted Modal containers sleep after 5 idle minutes and can take several minutes to cold start. This pings them with a free `GET /v1/models` and stops on its own.

**Project layout**

```
src/
  tasks/       ResearchTask interface and the example kernel task
  harness/     research loop, trajectory graph, Kev client, cache, quota, stagnation detector, evaluator
  agents/      mutator, reflector, kernel template (example task)
  prompts/     Kev question builders and LLM mutation templates
  ui/          terminal dashboard
  web/         localhost dashboard
benchmarks/    example task assets
scripts/       demo runner, keepalive
tests/         bun test suites
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Event-sourced trajectory graph with lineage, purging, and compressed failure notes
- [x] Kev client with per-model routing, quota tracking, response cache, and self-hosted endpoints
- [x] Kev-9B hypothesis triage and Kev-27B stagnation-triggered exploration
- [x] Tiered evaluation with early exit
- [x] `ResearchTask` interface; the research loop and shift engine are task-agnostic
- [x] Example task: GPU kernel optimization, with a with/without-Kev comparison
- [x] Terminal and web dashboards with deterministic replay
- [ ] Task selection from the CLI and web dashboard (`--task`), so new tasks need no wiring code
- [ ] More example tasks: hyperparameter search, heuristic discovery, prompt optimization
- [ ] Parallel evaluation workers for tasks with long evaluations
- [ ] LLM hypothesis generator wired to a frontier model for open-ended proposals
- [ ] Container sandbox for tasks that execute untrusted code

See the [open issues](https://github.com/teddycitrus/Kev-olution/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". New `ResearchTask` implementations are especially welcome.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Project Link: [https://github.com/teddycitrus/Kev-olution](https://github.com/teddycitrus/Kev-olution)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & BADGES -->
[contributors-shield]: https://img.shields.io/github/contributors/teddycitrus/Kev-olution.svg?style=for-the-badge&cacheSeconds=60
[contributors-url]: https://github.com/teddycitrus/Kev-olution/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/teddycitrus/Kev-olution.svg?style=for-the-badge&cacheSeconds=60
[forks-url]: https://github.com/teddycitrus/Kev-olution/network/members
[stars-shield]: https://img.shields.io/github/stars/teddycitrus/Kev-olution.svg?style=for-the-badge&cacheSeconds=60
[stars-url]: https://github.com/teddycitrus/Kev-olution/stargazers
[issues-shield]: https://img.shields.io/github/issues/teddycitrus/Kev-olution.svg?style=for-the-badge&cacheSeconds=60
[issues-url]: https://github.com/teddycitrus/Kev-olution/issues
[license-shield]: https://img.shields.io/github/license/teddycitrus/Kev-olution.svg?style=for-the-badge&cacheSeconds=60
[license-url]: https://github.com/teddycitrus/Kev-olution/blob/main/LICENSE

[TypeScript-badge]: https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white
[TypeScript-url]: https://www.typescriptlang.org/
[Bun-badge]: https://img.shields.io/badge/Bun-000000?style=for-the-badge&logo=bun&logoColor=white
[Bun-url]: https://bun.sh/
[Kev-badge]: https://img.shields.io/badge/Kev-4F8CFF?style=for-the-badge&logoColor=white
[Kev-url]: https://github.com/jaredpalmer/kev
[Modal-badge]: https://img.shields.io/badge/Modal-7FEE64?style=for-the-badge&logoColor=black
[Modal-url]: https://modal.com/
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch-badge]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[Triton-badge]: https://img.shields.io/badge/Triton-2C3E50?style=for-the-badge&logoColor=white
[Triton-url]: https://triton-lang.org/

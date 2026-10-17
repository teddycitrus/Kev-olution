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
    Evolutionary GPU kernel optimization, routed by Kev
    <br />
    <a href="https://github.com/teddycitrus/kev-olution"><strong>Explore the docs &raquo;</strong></a>
    <br />
    <br />
    <a href="https://github.com/teddycitrus/kev-olution/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/teddycitrus/kev-olution/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
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

Kev-olution is an autonomous research harness that evolves GPU kernels. It starts from a naive Triton flash-attention kernel, proposes mutations, tests them in a tiered sandbox, and keeps whatever runs faster without changing the output. It is built around two problems that sink most coding agents on long optimization runs: they get stuck in local maxima, and they fill their context with failed traces until their reasoning degrades.

Kev-olution handles both with Kev, a family of small decision models that answer typed questions about a state instead of generating text. Kev-9B scores every batch of candidate kernels in one call and prunes the weak ones before they reach the sandbox. When throughput stops improving for three cycles, Kev-27B picks a structural rewrite (TMA tensor descriptors, warp specialization, persistent scheduling) and the search jumps to a new basin. Search history lives in a DAG with failures compressed to one-line summaries, so no prompt ever carries a raw log.

<br>

Key capabilities:
* **Kev routing**: kev-9b prunes candidates before the sandbox; kev-27b chooses the Paradigm Shift when the Plateau Detector fires. Every decision reports its source (network, cache, or heuristic fallback)
* **DAG trajectory memory**: an event-sourced graph of every candidate, its lineage, metrics, and a failure summary under 280 characters. Stalled branches are purged on a shift; the best kernel and its ancestors are always kept
* **3-tier evaluator**: Tier 1 static checks (imports, launch, blocked dynamic code), Tier 2 compile and correctness on a mini workload, Tier 3 full benchmark. Weak candidates exit early so they never pay for Tier 3
* **Kernel templates**: mutations edit a typed kernel config and re-render real Triton source, so every candidate satisfies the harness contract. An optional LLM generator uses the same prompt templates
* **Live dashboards**: a terminal dashboard for the 45 second pitch and a localhost web dashboard with a search DAG, throughput chart, Kev telemetry, and a "time saved by Kev" breakdown
* **Deterministic demos**: Kev answers are cached to disk, so a rehearsed run replays exactly, with no network and no quota

<br>

**Results from the demo benchmark** (flash-attention forward, B4 H32 N4096 D128 fp16):

| | With Kev | Without Kev |
|---|---|---|
| Final throughput | 176.1 TFLOPS (4.27x baseline) | 176.1 TFLOPS (4.27x baseline) |
| Search cycles | 25 | 26 |
| Candidate kernels written | 119 | 130 |
| Compiles + correctness checks | 47 | 79 |
| Paradigm shifts | 3, chosen by kev-27b | 4, fixed order |
| Estimated wall-clock (upper-end step costs) | about 1.9 h | about 2.3 h |

Throughput numbers come from a simulated H100 performance model, not a real GPU, and are not calibrated to production H100 kernels. Kev's decisions are real: recorded from the self-hosted kev-9b and kev-27b endpoints and replayed from cache. The wall-clock estimate assumes 40 s per LLM-written kernel, 35 s per compile and check, 20 s per benchmark, and 1.3 s per Kev call. All four are editable in the web dashboard.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![TypeScript][TypeScript-badge]][TypeScript-url]
* [![Bun][Bun-badge]][Bun-url]
* [![Triton][Triton-badge]][Triton-url]
* [![PyTorch][PyTorch-badge]][PyTorch-url]
* [![Python][Python-badge]][Python-url]
* [![Modal][Modal-badge]][Modal-url]
* [![Kev][Kev-badge]][Kev-url]

| Layer | Tools |
|---|---|
| Search loop, agents, DAG store | TypeScript on Bun |
| Kernels and hardware harness | Triton, PyTorch, Python |
| Decision routing | Kev-9B and Kev-27B (TypeSafe SystemOne protocol), self-hosted on Modal |
| Dashboards | Raw ANSI terminal UI; dependency-free HTML + SVG served by `Bun.serve` |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* [Bun](https://bun.sh) 1.4 or later
  ```sh
  npm install -g bun
  ```
* A Kev API key for the hosted gateway, or your own Kev endpoints (see the `kev-deploy` skill in `.claude/skills/`). Without a key everything still runs offline with heuristic routing
* For real hardware runs only: Python 3.10+, PyTorch with CUDA, Triton 3.4+, and an NVIDIA GPU (Hopper for the TMA and warp-specialization variants)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/teddycitrus/kev-olution.git
   cd kev-olution
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

5. Run a search
   ```sh
   bun run evolve --benchmark triton_flash_attention
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->
## Usage

**Terminal dashboard**

`bun run evolve` runs the full loop. In an interactive terminal at least 120 columns wide it draws the live dashboard: search DAG, throughput chart with shift markers, Kev telemetry, compute cost, and an event log. Piped output, or `--plain`, falls back to a plain event log.

| Flag | What it does |
|---|---|
| `--benchmark NAME` | Benchmark folder under `benchmarks/` (default `triton_flash_attention`) |
| `--cycles N` | Max search cycles (default 40) |
| `--duration S` | Wall-clock budget in seconds |
| `--seed N` | Seed for the mutator and evaluator noise (default 0) |
| `--demo` | Kev response cache from `.kev/demo-cache.json`, 45 s pitch pacing, final frame held 7 s |
| `--pace MS` | Milliseconds slept per simulated GPU-second |
| `--offline` | No Kev calls: heuristic routing and the fixed shift order |
| `--gpu` | Use the real Triton runner when torch, Triton, and CUDA are available |
| `--warmup` | Wake cold Kev containers before the run |
| `--plain` | Plain text log instead of the dashboard |

**45 second pitch demo**

```sh
bun run demo           # checks Kev readiness, then the paced run with the dashboard
bun run evolve --demo  # same run without the readiness check
bun run demo:offline   # fallback with no Kev and no network
```

The first networked run records every Kev answer. Later runs with the same seed replay those answers from cache in under 5 ms each, so the demo is identical every time.

**Web dashboard**

```sh
bun run web
```

Open [http://localhost:4173](http://localhost:4173). The page starts empty; choose a run mode in the sidebar:

* **Run demo**: Kev answers replayed from the recorded cache
* **Run live**: every decision goes to Kev over the network (saved to its own cache file, so the demo replay never changes)
* **Run offline**: no Kev, for comparison

When a Kev run finishes, the "Time saved by Kev" modal compares it with the identical search without Kev: cycles, candidates, compiles, benchmarks, simulated GPU time, and estimated wall-clock.

**Keeping self-hosted Kev warm**

Self-hosted Modal containers sleep after 5 idle minutes, and cold starts take up to about 6.5 minutes. Before a live session:

```sh
bun run keepalive --every 180 --minutes 60
```

It pings `GET /v1/models` (free, no quota) and stops on its own. The containers bill while they are up.

**Project layout**

```
src/
  agents/      mutator, kernel template, reflector
  harness/     evolve loop, evaluator, kernel runners, DAG store, Kev client, cache, quota, plateau detector
  prompts/     LLM mutation templates and Kev question builders
  ui/          terminal dashboard (state, render, shell)
  web/         localhost dashboard (server + page)
benchmarks/
  triton_flash_attention/   baseline kernel, PyTorch reference, hardware harness
scripts/       demo runner, keepalive
tests/         bun test suites
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Event-sourced DAG trajectory store with replay, purging, and compressed failure summaries
- [x] Kev client with per-model routing, daily quota tracking, speculative cache, and self-hosted endpoints
- [x] 3-tier evaluator with a simulated H100 model and a pluggable real-GPU Triton runner
- [x] Mutator, reflector, Plateau Detector, and kev-27b Paradigm Shift engine
- [x] Terminal dashboard and deterministic 45 second demo
- [x] Web dashboard with a measured no-Kev comparison
- [ ] Validate the `--gpu` path end to end on a real H100 and replace simulated numbers with measured ones
- [ ] Wire `LlmGenerator` to a frontier model for open-ended kernel rewrites
- [ ] Run candidates inside a container sandbox instead of process isolation
- [ ] More benchmarks: causal attention, GEMM, layernorm, MoE routing

See the [open issues](https://github.com/teddycitrus/kev-olution/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

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

Project Link: [https://github.com/teddycitrus/kev-olution](https://github.com/teddycitrus/kev-olution)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & BADGES -->
[contributors-shield]: https://img.shields.io/github/contributors/teddycitrus/kev-olution.svg?style=for-the-badge&cacheSeconds=60
[contributors-url]: https://github.com/teddycitrus/kev-olution/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/teddycitrus/kev-olution.svg?style=for-the-badge&cacheSeconds=60
[forks-url]: https://github.com/teddycitrus/kev-olution/network/members
[stars-shield]: https://img.shields.io/github/stars/teddycitrus/kev-olution.svg?style=for-the-badge&cacheSeconds=60
[stars-url]: https://github.com/teddycitrus/kev-olution/stargazers
[issues-shield]: https://img.shields.io/github/issues/teddycitrus/kev-olution.svg?style=for-the-badge&cacheSeconds=60
[issues-url]: https://github.com/teddycitrus/kev-olution/issues
[license-shield]: https://img.shields.io/github/license/teddycitrus/kev-olution.svg?style=for-the-badge&cacheSeconds=60
[license-url]: https://github.com/teddycitrus/kev-olution/blob/main/LICENSE

[TypeScript-badge]: https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white
[TypeScript-url]: https://www.typescriptlang.org/
[Bun-badge]: https://img.shields.io/badge/Bun-000000?style=for-the-badge&logo=bun&logoColor=white
[Bun-url]: https://bun.sh/
[Triton-badge]: https://img.shields.io/badge/Triton-2C3E50?style=for-the-badge&logoColor=white
[Triton-url]: https://triton-lang.org/
[PyTorch-badge]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Modal-badge]: https://img.shields.io/badge/Modal-7FEE64?style=for-the-badge&logoColor=black
[Modal-url]: https://modal.com/
[Kev-badge]: https://img.shields.io/badge/Kev-4F8CFF?style=for-the-badge&logoColor=white
[Kev-url]: https://github.com/jaredpalmer/kev

# Algorithms and modeling limits

The optimizer proposes a grid, compiles the circuit on that grid, and uses the
resulting score to decide what to try next. The default score is **space–time
volume**: rectangular area in tiles × completed compiler time slices. Lower is
better. See [concepts and terminology](concepts.md) for units and a worked example.

## Geometry and objective

A search state contains the rectangular text grid and the coordinates assigned
to each logical qubit. Search can resize the grid, move data patches and change
initial qubit assignments. Data patches remain static, single-cell `Q` sites in
the grid interior, and routing cells must remain connected. Passing these checks
does not guarantee compilation: the native C++ compiler makes that determination.

There is no hard area cap. An optional **augmented Chebyshev objective** instead
scores weighted, normalized area and duration to explore different tradeoff
preferences. The saved **observed Pareto frontier** collects evaluated layouts
that are not outperformed in both quantities. It is not a complete frontier of
all possible designs. See the [API guide](api.md) for objective settings.

## Supported optimization API

An API is the Python interface through which another program invokes the search.
The `method` strings below are also command-line options; the prose names identify
the algorithms they select.

| Algorithm | Method identifier | Behavior |
|---|---|---|
| Simulated annealing | `sa` | Follows one sequence of candidate layouts; gradually reduces willingness to accept worse scores |
| Independent-restart simulated annealing | `restart_sa` | Runs independent searches sharing a fixed total request budget |
| Parallel tempering | `parallel_tempering` | Runs several searches at different temperatures and proposes exchanges of their layouts |

**Temperature** is an algorithm setting controlling acceptance of worse layouts,
not a physical device temperature. Calibration estimates a useful scale from
trial proposals. A **chain** is a sequence of search states; a **replica** is one
such chain in parallel tempering. Hotter replicas explore more freely, while
colder ones favor lower scores. Exchanges can transfer discoveries between them.

Proposals can relocate one patch, reassign several qubits together, insert or
delete routing rows and columns, and optionally relocate `M` supply cells.
An invalid proposal consumes a budget slot and leaves the state unchanged (a
**self-loop**). It is not silently replaced with another attempt. The uniform
proposal rules account for differences between forward and reverse proposal
probabilities. These searches are optimization heuristics; unlimited grid area
precludes a claim here of a normalizable Boltzmann equilibrium distribution.

Budgets count score requests, including the initial layout, temperature-calibration
probes, failures, duplicates and cached results. A cached result avoids a new
compiler invocation but still consumes a request. The public API makes one
additional fresh compilation of the exported winner (**replay**) outside that budget.

## Experimental reinforcement-learning interfaces

**Reinforcement learning (RL)** trains a policy to choose actions using reward
feedback. In `geometry_rl`, a **convolutional neural network (CNN)** processes
variable-size grids. Its **actor** scores actions; its **critic** estimates future
reward. Training uses **proximal policy optimization (PPO)**.

The policy scores individual actions and multi-step action templates (compound
moves and resizing). Some destinations within those templates are drawn randomly;
the network does not choose every endpoint. Successful proposals earn normalized
improvement in the best volume found so far; failures receive -0.2. Combining
learned proposals with annealed acceptance is an optimization heuristic, not an
exact equilibrium sampler.

The `neural`, `ppo` and older `search` modules include experiments that only
reassign qubits to fixed sites using precomputed score tables. Those workflows do
not change grid geometry. Their historical scores are excluded from this site's
benchmark. The supported `optimize_layout` function does not expose PPO training.

**Hyperparameter optimization (HPO)** searches over algorithm settings. Optuna is
the project's tool for declared tuning studies; MLflow records experiment settings
and outputs. The [20-circuit benchmark](local-results.md) uses simulated annealing
without hyperparameter tuning. It does not compare reinforcement learning against
annealing under the local-compilation protocol.

## Magic-state sources and local execution

`M` cells are abstract deterministic magic-state sources, not fully modeled
physical factories. Their count stays fixed within a search; an outer experiment
runs separate searches for different counts. The benchmark fixes the compiler's
`distillation_time` to 10 timesteps. Physical factory footprints, production failure
probabilities, code distances and error-correction/decoding cycles are not modeled.

`--local` requests explicit local instruction decomposition. `-P wave` selects the
scheduler. These are separate settings. The benchmark uses both; the distinct
`-P edpc` scheduling pipeline is not measured. See the
[integration contract](integration.md) for the exact replay command.

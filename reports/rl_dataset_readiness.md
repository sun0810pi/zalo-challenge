# Phase 4 Audit: Reinterpreting Challenge Data as an RL Experience Dataset

**Repository**: `c:\Users\phuon\zalo-challenge`  
**Dataset & Environment Assets**:
- `train/train.jsonl` (1,545 demonstration episodes)
- `train/train_environment/<id>/state.json` (1,545 initial state snapshots)
- `train/zaic_tools/` (`ToolEnv`, `Session`, `canon`, `state_hash`)  
**Audit Status**: CONFIRMED via mathematical formulation, state replaying, and evaluation metric verification.

---

## 1. Executive Summary

This audit determines whether the original challenge dataset (`train/train.jsonl`) can serve as an offline experience dataset for Reinforcement Learning (RL), preference optimization, and rejection sampling.

### Primary Conclusions
1. **The Dataset is a Set of Deterministic Expert Demonstrations, Not an Offline RL Buffer**:
   - The dataset contains **only positive, optimal trajectories** ($\mathcal{D} = \{\tau^*\}$).
   - It contains zero exploration rollouts, zero negative actions, zero counterfactual branches, and zero failed trajectories.
   - Standard offline RL algorithms (e.g. CQL, IQL, TD3-BC) require multi-modal data covering diverse state-action distributions with variable rewards. Applying offline RL directly to `train.jsonl` degenerates to Behavior Cloning (SFT).
2. **The Dataset + Simulator is Ready for Rejection Sampling Fine-Tuning (RSFT)**:
   - Because `train_environment/<id>/state.json` provides the exact initial state $s_0$, and because `ToolEnv` executes deterministically in **~15.5 ms**, any policy can generate $K$ candidate rollouts, execute them in `ToolEnv`, and verify success using the exact competition metric ($R = 1.0$).
3. **Preference Pair Generation (DPO) is Immediately Feasible**:
   - By running the current baseline model (LB ~30) on `train/`, we obtain hundreds of naturally failing trajectories ($\tau_{loser}$). Pairing each failure with the ground-truth demonstration ($\tau_{winner} = \tau^*$) creates an immediate, high-fidelity DPO dataset $(x, y_w, y_l)$.
4. **Reward Verifiability: The State-Hash Trap in Read-Only Tasks**:
   - For mutating write tasks (1,172 samples), comparing `state_hash(s_T) == gt_hash` provides 100% reliable ground-truth verification.
   - However, for read-only tasks (262 samples) and zero-call cancellation tasks (111 samples), `state.json` remains completely unchanged! A policy that emits zero actions would trivially achieve a perfect state-hash match. Therefore, rewards **must** combine state-hash accuracy with tool-call F1, exactly mirroring the official competition metric.

---

## 2. Formal MDP Formulation of Challenge Trajectories

Each example in `train/train.jsonl` maps to a finite-horizon Markov Decision Process (MDP):

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, \mathcal{T} \rangle$$

### 2.1 State Space $\mathcal{S}$
A state $s_t \in \mathcal{S}$ consists of:
- **Static Context** $c$: Utterance timestamp $t_0$, GPS location, user profile (`home_address`, `city`), and installed app catalog.
- **Goal Observation** $x$: User request transcript (`audio_text`) + previous dialogue history.
- **Environment State** $\mathcal{E}_t$: Live application data in memory (contacts, alarms, calendar events, smart home status, bank accounts) loaded from `state.json` and mutated by tools.
- **Dialogue Context** $h_t$: The sequence of prior agent actions and simulator observations $(a_0, o_0, a_1, o_1, \dots, a_{t-1}, o_{t-1})$.

### 2.2 Action Space $\mathcal{A}(s_t)$
- **Discrete Action**: Selecting an installed tool $t \in \text{installed\_apps} \cup \{\text{sys.ask\_missing\_info}\}$.
- **Continuous/Structured Arguments**: Generating schema-valid JSON parameters:
  $$a_t = \langle \text{tool\_name}, \text{arguments} \rangle$$
- **Termination Action**: Emitting natural text without `<tool_call>` (stopping).

### 2.3 Transition Dynamics $\mathcal{P}(s_{t+1} \mid s_t, a_t)$
- **100% Deterministic**: Given current memory state $\mathcal{E}_t$ and action $a_t$, `ToolEnv.call(name, args)` returns observation $o_t$ and transitions environment memory to $\mathcal{E}_{t+1}$.
- If $a_t$ is invalid or a read tool, $\mathcal{E}_{t+1} = \mathcal{E}_t$.

---

## 3. Method Suitability Assessment for Existing Challenge Data

We evaluate which learning paradigms can be implemented immediately with existing project data:

| Paradigm | Feasible Immediately? | Data Requirements | Required Infrastructure | Assessment & Value |
| :--- | :---: | :--- | :--- | :--- |
| **Behavior Cloning (SFT)** | **YES** | Demonstrations only ($\tau^*$). | Existing `train_sft.py`. | **Current baseline**. Baseline warm-start only; fails on error recovery. |
| **Offline RL (CQL / IQL)** | **NO** | Multi-modal rollouts with negative actions and reward labels. | Offline Q-function & Actor network. | **Unsuitable**. `train.jsonl` contains zero suboptimal actions to penalize. |
| **Rejection Sampling (RSFT)** | **YES** | Prompts + initial state snapshots (`state.json`). | vLLM rollout generator + `ToolEnv` scorer. | **Highest Value**. Sample $K=4$ trajectories per prompt, filter for $R=1.0$, fine-tune on verified successes. |
| **Direct Preference Opt (DPO)** | **YES** | Paired preferences $(x, y_w, y_l)$. | Baseline model rollouts paired with GT demonstrations. | **High Value**. Penalizes specific failure modes (e.g. wrong IDs, uncalled read tools). |
| **Online RL (GRPO / PPO)** | **Feasible in Phase P4** | Environment simulator with interactive step API. | Policy engine, rollout buffer, reference model, GPU cluster. | **High Complexity**. Requires dedicated online rollout pipeline. PPO infeasible on 15GB L4; GRPO possible. |
| **Neural Reward Model** | **NO / Unnecessary** | Ranked preference pairs. | Preference dataset + Reward Model Trainer. | **Not Recommended**. Simulator provides ground-truth deterministic reward; neural RM introduces reward hacking. |

---

## 4. Ground-Truth Reward Construction

### 4.1 Ground-Truth Verifier (Zero Reward Hacking)
Unlike open-ended creative tasks that require subjective human preferences, tool execution on LifeMate is **verifiable against formal simulator state and tool schemas**.

The terminal reward function should directly align with the competition scoring metric in [`evaluate.py`](file:///c:/Users/phuon/zalo-challenge/evaluate.py#L177):

$$R_{\text{terminal}}(\tau) = 0.5 \cdot \mathbb{I}\left(\text{state\_hash}(\mathcal{E}_T) == \text{gt\_hash}\right) + 0.5 \cdot F_1\left(\text{calls}_{\text{pred}}, \text{calls}_{\text{gt}}\right)$$

Where:
- $\text{gt\_hash}$ is computed by replaying `label.tool_calls` in a fresh `ToolEnv`.
- $F_1$ is the harmonic mean of precision and recall over tool names and arguments, utilizing the competition's fuzzy matching rules for free-text keys (`FREE_TEXT_KEYS`).

### 4.2 Dense Intermediate Step Penalties (Shaping Reward)
To prevent infinite looping and encourage minimal sufficient action sequences:

$$r_t(s_t, a_t) = \begin{cases}
-0.05 & \text{Step latency penalty (discourages unnecessary calls)} \\
-0.20 & \text{if } a_t \text{ violates schema } (env.validate() \text{ returns } ok=\text{false}) \\
-0.20 & \text{if } a_t \text{ uses an ungrounded write ID } (state\_tracker \text{ violation}) \\
0.00 & \text{Valid intermediate tool call}
\end{cases}$$

Total Episode Return:

$$R_{\text{total}} = R_{\text{terminal}} + \sum_{t=1}^{T} r_t$$

### 4.3 Why State-Hash Comparison Alone is Insufficient
We identified four critical task archetypes where state-hash comparison alone yields false positives or false negatives:

```text
Archetype 1: Read-Only Query (262 samples)
Example: "Xem giúp mai có mưa không"
State at start: S0 | State at end: S0 (No mutation!)
A model that does NOTHING emits 0 calls -> state_hash matches gt_hash!
-> F1 component enforces that weather.get_forecast was actually called.

Archetype 2: Full Cancellation (111 samples)
Example: "Tắt đèn phòng khách đi... à mà thôi, đừng tắt."
State at start: S0 | State at end: S0
-> Both state_hash and calls=[] must match.

Archetype 3: Disambiguation (217 samples)
Example: "Gọi cho Minh" (Two contacts named Minh exist)
State at start: S0 | State at end: S0
-> Model must call sys.ask_missing_info. State is unchanged, but F1 requires the correct system call.

Archetype 4: Failed Mutation Attempt
Example: Attempting to pay an electricity bill with insufficient wallet balance.
-> If the model attempts payment, the call succeeds with done=false, but state is unchanged.
-> Verification must check both state_hash and business logic return codes.
```

---

## 5. Summary of RL Dataset Readiness

| Component | Status | Source in Repository | Action Required |
| :--- | :---: | :--- | :--- |
| **Initial States ($s_0$)** | **Ready** | `train/train_environment/<id>/state.json` | None. 1,545 verified initial states exist. |
| **Optimal Actions ($a^*$)** | **Ready** | `train/train.jsonl` (`label.tool_calls`) | None. Verified 100% positionally aligned. |
| **Simulator Dynamics ($\mathcal{P}$)**| **Ready** | `train/zaic_tools/executor.py` | Fast (15.5 ms/sample) deterministic execution. |
| **Terminal Verifier ($R$)** | **Ready** | `evaluate.py` (`combined_score`) | Formal ground truth scorer already implemented. |
| **Exploration / Negative Rollouts** | **MISSING** | *Not in repository* | Must be generated by sampling baseline model. |
| **Preference Pairs $(y_w, y_l)$** | **MISSING** | *Not in repository* | Can be built in ~30 mins by pairing baseline errors with GT. |
| **Gymnasium RL Wrapper** | **MISSING** | *Not in repository* | Must wrap `ToolEnv` with `step()` and `reset()`. |

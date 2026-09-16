# The scorecard records the operating decision.

The evaluation status is one complete cached evaluation for each artifact.
The research status must state that a threshold selected from cached MVTec test predictions is retrospective.
This selection swept 242 cached test thresholds, so it is retrospective research and not a validated operating threshold.
The commercial status must remain unqualified until independent customer acceptance succeeds.
The commercial status is unqualified; no independent customer validation has run.

The following table compares the selected operating points.

| Decision | Precision | Provider and placement | Threshold | Recall | False reject rate | F1 | Expected dollars/shift | Clears sustained budget |
|---|---|---|---:|---:|---:|---:|---:|---|
| The cost rule selects this point. | INT8 | cpu, CPUExecutionProvider | 2.089220 | 0.984127 | 0.150000 | 0.968750 | 27155.66 | yes, worst sustained minute 29.778 ms |
| The F1 rule selects this point. | FP32 | comparison_only_not_a_provider_choice | 2.089352 | 0.984127 | 0.150000 | 0.968750 | 27155.66 | feasible |

The chosen artifact SHA256 is `8e78b0855d659be2473f55d69eb91eb063b0ace18f57d9bb5c25984bc34fed9a`.
The chosen configuration digest is `d77f4647b628e15d1474af04b3c291cfad8ced7f3c1c749ea5ff25fb0496dbab`.
The maximum-F1 comparison uses the same prevalence and disposition policy.
The maximum-F1 point is feasible.
The expected difference in dollars per shift is 0.00 USD.
The expected difference per month is 0.00 USD across 40 shifts.
These quantities describe modeled quality cost rather than observed commercial savings.

The following table records the economic assumptions.

| Input | Configured value | Source of validation |
|---|---|---|
| The model uses this defect prevalence. | 0.01 | illustrative_research_only; not measured on a customer line |
| The shift produces this many parts. | 21600 | illustrative_research_only |
| A false accept incurs this warranty cost. | 250.0 USD | illustrative_research_only |
| A false reject incurs this scrap cost. | 8.0 USD | illustrative_research_only |
| A true reject incurs this rework cost. | 3.0 USD | illustrative_research_only |
| The month contains this many shifts. | 40 | illustrative_research_only |

The cost curve appears at artifacts/figures/cost-curve.png.
The latency figure appears at artifacts/figures/latency-budget.png.
The curve marks both comparison points and displays prevalence sensitivity.
The chosen threshold satisfies the configured recall floor of 0.95.
The selected provider satisfied the measured budget in every sustained minute.
The frozen validation threshold produced recall 0.904762, which does not satisfy that floor.
The measured acquisition, transport, and actuation overhead is TBD.
The confidence interval for customer recall is TBD.
The source of that independent customer sample is TBD.

The evidence hashes for evaluation, timing, thermal samples, costs, and the environment follow.

| Evidence | SHA256 |
|---|---|
| artifacts/environment.json | `f1ae9bca21c06dda41634c5cf1a1fe35562e7b457c9fc3a48826ab5785acd87a` |
| config/bench.yaml | `a79c748c5a1075d9fbdce2a75725441fb84f1bc71f6d5f591055cb41236bb4cc` |
| config/costs.yaml | `1d528f89d8cd74f894c58d26958793f8f39d3307e62e9590478acf832bf1b139` |
| gate1.json | `ca05b817f94994153f27d8fa27a4b2d8691fbe09be9fdbfdab30228e2b6f704e` |
| gate2.json | `6b98e9985d02872e3b704c83cec6ffd170e3009f021c93bbe1451c521f0f3abc` |
| gate3.json | `15ea8d3dd530f371bd80620b9422839df568c91578ceb0b4495aeee0e404dab6` |
| gate4.json | `9f022ce12cb84b74246f69b5844f160635160601b74369081f4c2acab3cd6ff9` |
| selected artifact | `8e78b0855d659be2473f55d69eb91eb063b0ace18f57d9bb5c25984bc34fed9a` |
| thermal summary | `1f769c6cfff31b4e10dbc69d8a4383b5b74111327867576835f949b8a1940f6d` |

The LLM did not run, so its model and prompt versions are unavailable.
Every number in this scorecard comes from recorded evidence, and no LLM may rewrite a numeric field.
The unresolved commercial acceptance conditions are customer data rights, independent validation, signed release, and shift soak acceptance.

# The implementation plan reserves two weekends for research.

The first weekend implements dataset verification, deterministic partitions, and the chosen model wrapper.
The engineer records the teacher provenance and completes the ADR checks before training.
The engineer then creates FP32, freezes the initial threshold, and implements the locked evaluation ledger.
The engineer completes Gates 0 and 1 before creating reduced-precision artifacts.

The second weekend implements FP16, static INT8, parity, measured timing, and the cost reports.
The engineer must first test whether every requested provider accepts every artifact on the actual Mac.
A failed provider prevents a complete gate result and requires a visible compatibility decision.
The sustained matrix alone requires at least six times thirty minutes when all six pairs work.
Cooldowns, compilation, telemetry setup, and reruns require additional time.
The engineer should reserve a separate measurement block before the weekend ends.

The weekend scope leaves the LLM and production modules as explicit stubs if the core measurements consume the available time.
The first LLM implementation can explain existing evidence before the agent gains permission to execute a gate.
The next increment adds a bounded planner, policy checks, checkpoint recovery, and adversarial tests.
The project must not trade measurement integrity for a more elaborate agent demonstration.

The following table assigns the initial evidence to its gate.

| Gate | The implementation must produce this evidence. | The implementation must refuse this failure. |
|---|---|---|
| 0 | The verifier records hashes and validates all categories. | The verifier refuses unapproved or inconsistent data. |
| 1 | The trainer records FP32 and its sole evaluation. | The evaluator refuses an absent ADR or quality below the floor. |
| 2 | The exporter records all precisions, parity, and cached accuracy. | The exporter refuses missing operators or excess numerical error. |
| 3 | The runner records every provider pair and writes latency.md. | The runner refuses absent timestamps or unsupported required pairs. |
| 4 | The runner records thirty sustained minutes for each pair. | The runner refuses incomplete windows or hidden deadline misses. |
| 5 | The cost module writes the curve and scorecard. | The selector refuses stale evidence or an empty feasible set. |

The file stubs define the intended implementation without pretending that the gates already work.
The test skeletons intentionally fail until their behavioral assertions replace pytest.fail.
The scaffold's syntax checks do not establish ML accuracy, provider support, measured latency, or production readiness.

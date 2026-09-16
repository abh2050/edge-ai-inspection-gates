# Gate 4 records sustained paced latency.

Gate 4 passed its sustained checks.
Each series paced 45 parts per minute for 30 minutes against absolute monotonic deadlines.
Latency runs from each scheduled slot start to synchronous postprocessing completion.
A busy slot counts as missed, and the runner never queues catch-up frames.
The exact research cycle allowance is 1333.333333 milliseconds.

| Provider | Precision | Role | Minute 1 p99 ms (n) | Minute 25 p99 ms (n) | Worst-minute p99 ms (minute, n) | Achieved parts/min | Missed slots | Deadline misses | Throttling attribution |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| cpu | FP32 | required | 25.885 (45) | 30.712 (45) | 31.073 (21, 45) | 45.000 | 0 | 0 | no_throttling_signal_observed |
| cpu | FP16 | required | 30.774 (45) | 29.605 (45) | 32.889 (16, 45) | 45.000 | 0 | 0 | no_throttling_signal_observed |
| cpu | INT8 | required | 27.803 (45) | 28.615 (45) | 29.778 (29, 45) | 45.000 | 0 | 0 | no_throttling_signal_observed |

Each minute holds about 45 samples, so its empirical p99 lies near the largest observation and is a noisy tail estimate rather than a stable guarantee.
A later-minute increase alone does not establish thermal throttling.
Throttling attribution comes only from pmset thermal telemetry, and unavailable telemetry stays unknown.
The workstation did not record ambient temperature or a representative enclosure.
Release feasibility remains unestablished because acquisition, transport, and actuation overhead remain unmeasured.

# ADR 0008 adds one read-only evidence dashboard.

Status: accepted by the project owner on 2026-09-16.

The original contract refused a frontend because a console invites operator input, queued work, and hand-edited results.
The measured evidence now lives in JSON records that only a reader can interpret through six separate documents.
A reviewer who wants the result must open docs/latency.md, docs/sustained.md, docs/scorecard.md, and three gate records.

The repository therefore builds one dashboard that renders recorded evidence and nothing else.
The dashboard reads a generated data file that scripts/build_dashboard.py extracts from artifacts and configuration.
The build step copies recorded values and never recomputes a quantile, interpolates a curve, or derives a new number.
Each figure on the page names its source gate, and the page names the host that produced the measurements.

The dashboard accepts no operator input and writes nothing back.
The dashboard never triggers a gate, calls an LLM, or reaches any network destination.
The page serves from the local host for review and carries no credential.
The page presents the failed Neural Engine placement, the diagnostic CoreML rows, the recall shortfall, and the unqualified release with the same prominence as the passed gates.

A dashboard is a reading surface and never acceptance evidence.
The gate records remain the only evidence, and a reviewer who doubts a figure reads the recorded file the page cites.
This decision does not permit a review console, a labeling interface, a queue, or any control that mutates a measurement.

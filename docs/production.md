# The production plan sells a qualified inspection deployment.

The first buyer should own a repetitive inspection step with measurable escape and scrap costs.
A quality manager supplies defect definitions and acceptance criteria.
An automation engineer supplies the camera, trigger, controller, and fault-handling requirements.
An operations owner supplies volume, defect prevalence, and disposition costs.
The product qualifies one defined family of parts on one defined station before expanding to other sites.
The customer purchases a measured business outcome only after the pilot establishes it.

The first paid engagement should deliver a feasibility report using commercially authorized customer data.
The next engagement should deliver a signed offline inspection bundle and a tested station adapter.
The continuing service should cover requalification, controlled updates, and incident support.
A commercial offer should separate commissioning charges from recurring support and software charges.
The owner must enter proposed prices and measured delivery costs in config/costs.yaml before building a financial case.
The scaffold does not claim that a buyer will pay an unvalidated price.

The following table describes the work that makes the product purchasable.

| Buyer concern | The product must provide this capability. | The acceptance process must produce this evidence. |
|---|---|---|
| The buyer needs lawful inputs. | The team records rights for data, teacher weights, dependencies, and outputs. | The release includes approved commercial rights records. |
| The buyer needs defect detection on its parts. | The team creates lot-separated training, labeled validation, and untouched acceptance sets. | The test reports defect-wise recall and confidence intervals at the frozen threshold. |
| The buyer needs the promised line rate. | The station measures capture through result delivery under representative load. | The test reports sustained tails, deadline misses, and actual completed throughput. |
| The buyer needs stable behavior after heating. | The team runs repeated full shifts in the intended enclosure and ambient conditions. | The report preserves per-minute timing and thermal telemetry where available. |
| The buyer needs safe recovery. | The station detects stale frames, camera loss, process crashes, disk pressure, and power recovery. | Fault tests confirm the agreed station response and recovery limits. |
| The buyer needs controlled changes. | The team signs model bundles and freezes preprocessing and thresholds. | A release check verifies signatures, provenance, acceptance, and rollback. |
| The buyer needs continuity without the cloud. | The station completes inspection locally during network and LLM outages. | An outage test records uninterrupted local inference and the configured fault policy. |
| The buyer needs accountable explanations. | The agent cites approved procedures and measured evidence. | Adversarial tests reject invented claims, prompt injection, and unauthorized tools. |
| The buyer needs support. | The team defines ownership, escalation, retention, and update windows. | A customer exercise confirms the incident runbook and support agreement. |

The two-weekend research build does not satisfy these commercial obligations.
The first customer pilot may require several further iterations as camera geometry, lighting, part motion, and defect prevalence become known.
The team should commit dates only after it has access to the station and representative parts.
The acceptance criteria in config/production.yaml must become concrete before the pilot begins.
Null criteria must block release.

The pilot first runs in shadow mode and records decisions without controlling rejection hardware.
The customer compares those decisions with trusted outcomes from its existing process.
The team freezes its threshold on labeled validation data before the acceptance phase.
The acceptance phase measures both defect escapes and rejection of good parts on new lots.
The customer must choose the sample size from the required uncertainty and defect risks.
A high point estimate from a handful of defects does not establish the agreed recall.

The station adapter must associate each frame with the correct physical part and result deadline.
The adapter must reject duplicated or stale frame identifiers.
The station contract must define camera loss, timing faults, and result-handshake behavior with the customer's controls engineer.
The runtime returns a deterministic inspection result to that adapter.
The LLM never drives rejection hardware or changes a deployed threshold.
The repository still needs no review console, queue, labeling interface, or web framework.

The release process must preserve data lineage, seeds, preprocessing, calibration IDs, dependency locks, and artifact hashes.
The team must produce a software bill of materials and review dependencies before shipment.
The owner must also choose the code license and define the customer contract for updates, support, and permitted use.
The team must verify the release on the exact target OS and accelerator configuration.
A customer using a different industrial computer needs fresh measurements on that computer.
Apple Neural Engine results cannot substitute for those measurements.

The telemetry system must track score distributions, image quality, faults, missed deadlines, and available device health.
A score shift should trigger investigation because it does not prove a recall loss by itself.
The customer must periodically supply verified outcomes for performance review.
The team must requalify changes to cameras, lighting, optics, parts, preprocessing, models, and thresholds.
The runtime must enforce tenant boundaries and retention rules for any stored observations.

The LLM adds value when it reduces the time needed to interpret failed experiments and assemble evidence.
The team should compare a deterministic report with a cited LLM report using held-out operator questions.
The evaluation should measure numeric faithfulness, valid citations, task completion, engineering time, and cost per completed workflow.
The agent should earn permission to run experiments only after it passes policy and recovery tests.
The product can remain valuable when the customer disables the LLM.

The economic pilot must compare observed quality cost against the customer's baseline under comparable production conditions.
The business case must include commissioning, software, support, downtime, and human investigation costs.
The team must distinguish modeled avoided returns from returns that the customer actually observed.
A renewal proposal should show whether the measured benefit exceeds the customer's complete cost of ownership.

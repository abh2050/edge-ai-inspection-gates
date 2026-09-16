# The runbook records station recovery responsibilities.

The customer must assign a named owner before release.
The owner is TBD.
The support contact and response agreement are TBD.
The station identifier and authorized release digest are TBD.

| Observed condition | The runtime must take this action. | The owner must preserve this evidence. |
|---|---|---|
| A frame arrives late or repeats an identifier. | The runtime returns the configured station fault. | The owner preserves frame IDs and scheduled deadlines. |
| Inference exceeds the result deadline. | The runtime records the miss and invokes the agreed fault contract. | The owner preserves latency samples and station state. |
| The model fails integrity verification. | The runtime refuses activation. | The owner preserves the failed digest and signature check. |
| The LLM becomes unavailable. | The station continues local inspection and records the optional workflow failure. | The owner preserves provider errors without exposing credentials. |
| A monitored score distribution changes. | The system requests a documented investigation. | The owner preserves permitted observations and capture conditions. |
| A release produces an acceptance regression. | The authorized owner verifies and activates the approved rollback bundle. | The owner preserves the current and prior release records. |

The customer must approve the controller response to a station fault before live operation.
The team must rehearse camera loss, disk exhaustion, process termination, and power recovery on a test station.
The recovery test must check whether every physical part receives the intended disposition after restart.
The agent must not resolve these incidents by changing production thresholds or activating a release.

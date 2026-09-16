# ADR 0003 freezes inference results and separates research from acceptance.

Gate 1 trains the model and creates the FP32 export before it evaluates that export.
Gate 2 reuses the existing FP32 artifact and evaluates each new reduced-precision artifact once.
The evaluator keys its ledger by artifact SHA256 and records data, preprocessing, and evaluator hashes within that entry.
Each completed evaluation stores enough predictions to compute later threshold metrics without rerunning inference.
The report can reuse those metrics only when all provenance checks succeed.

The first threshold comes from the held-out normal training partition.
The research cost curve later sweeps the cached MVTec test scores.
The scorecard must call this a retrospective selection because the test labels influence the chosen threshold.
The chosen point does not establish out-of-sample performance.
A commercial release instead chooses the threshold on labeled customer validation data and freezes it before independent acceptance testing.
Customer splits must separate production lots or capture periods to reduce near-duplicate leakage.

The report defines an anomaly as the positive class.
A false negative accepts a defective part and creates an expected warranty return.
A false positive rejects a good part and creates scrap.
A true positive rejects a defective part and creates rework under the initial cost model.
A true negative creates no incremental quality cost under that model.
These outcomes remain mutually exclusive.
A customer must change the configured outcome policy if its actual disposition process differs.

The cost function uses the following equation.

\[
C(t)=N\left[\pi\,\mathrm{FNR}(t)C_w+(1-\pi)\,\mathrm{FPR}(t)C_s+\pi\,\mathrm{TPR}(t)C_r\right].
\]

The configuration supplies the shift volume N, production defect prevalence pi, and each monetary coefficient.
The curve must not substitute the dataset's test defect fraction for the factory's prevalence.
The selection minimizes expected cost across precision, provider, and threshold after enforcing recall and sustained timing.
The comparison point maximizes F1 because AUROC does not select a threshold.
The report must identify the F1 point as a surrogate accuracy optimum and report its feasibility.
The report must compare both costs under identical prevalence and disposition assumptions.

The curve describes incremental quality cost rather than complete ownership cost.
The commercial case must also measure integration expense, support expense, downtime, and any LLM expense.
The configuration reserves these values without fabricating customer economics.
A commercial case remains incomplete while its required values are null.

# ADR 0004 records execution placement separately from requested precision.

The default matrix requests FP32, FP16, and INT8 on CPU and on CoreML with CPUAndNeuralEngine.
CoreML permits CPU work under that compute-unit setting.
The report must label actual placement and retain evidence for any claimed Neural Engine execution.
The provider option alone does not establish ANE execution.
The adapter must also record operators that ORT places outside CoreML.
The default policy rejects ORT fallback while permitting explicitly reported CPU work inside CoreML.

The official documentation describes compute-unit options and provider configuration in the [CoreML execution provider reference](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html).
The implementation must verify behavior against its locked ORT version and installed macOS version.
An FP32 or INT8 ONNX artifact may fail a required provider or may execute with different internal arithmetic.
The report must distinguish artifact precision from execution precision.
An unsupported required pair must remain a failed gate with diagnostic evidence.
The project must not claim that all six pairs work before measuring them.

Placement inspection may require an Apple profiling trace that the Python package does not supply.
The operator must attach that trace when the automatic adapter cannot establish placement.
A compute-plan estimate may support a placement investigation, but it must never supply a latency value.
The benchmark must derive every latency statistic from actual timed completions.

A 45-part minute contains roughly 45 samples.
Its empirical p99 lies near its largest observation and has substantial sampling uncertainty.
The report must preserve that count and avoid calling the estimate a stable tail guarantee.
Gate 4 measures behavior under the stated target workload even when the hardware never throttles.
Thermal telemetry must support a separate claim that throttling caused a slowdown.
Commercial qualification adds repeated full-shift runs under representative enclosure and ambient conditions.
ADR 0006 removes the ANE requirement for the current model and keeps this fallback policy.

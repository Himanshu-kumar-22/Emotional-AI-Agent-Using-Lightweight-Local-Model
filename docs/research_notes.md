
## MPS Latency Finding (Phase 1 Testing)

DistilBERT emotion inference on Apple Silicon M3 (MPS backend):
  - Average latency: ~12ms per inference
  - Report baseline (CPU): ~148ms
  - MPS speedup: ~12x over CPU

This exceeds the performance targets stated in the report objectives.
The 12ms latency means emotion detection adds negligible overhead
to the conversational response time even on repeated rapid messages.

Note for report: The 148ms figure in Table 2 reflects CPU-only hardware
(the 4GB RAM low-end profile). On mid-to-high-end hardware with
GPU support, latency drops dramatically.

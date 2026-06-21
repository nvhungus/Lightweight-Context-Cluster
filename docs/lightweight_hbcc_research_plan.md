# Research Plan: Lightweight HBCC for Context Cluster

## 1. Goals and Context

This document sets the research and implementation direction for optimizing the Context Cluster architecture toward **lightweight but practically efficient**. The goal is not just to reduce parameter count, but to find one or more configurations on the Pareto frontier across:

- `Accuracy`: classification accuracy, prioritizing `Top-1`.
- `Params`: trainable and total parameter count.
- `FLOPs/MACs`: real arithmetic cost of float/int operators.
- `BOPs`: theoretical cost of binary operators, only meaningful for inference if a real bitwise kernel exists.
- `Latency`: single-batch inference time, especially important at batch 1.
- `Throughput`: images/second at larger batches, e.g. batch 16, 64, 128.
- `Memory`: peak GPU memory, activation memory, and checkpoint/model export footprint.

The current problem is that HBCC has reduced parameter count compared to ResNet-18, but throughput has dropped sharply. This is not surprising given the execution path: CoC/HBCC involves many tensor reshapes, permutes, scatters, reductions, sigmoids, small matmuls, small branches, and small kernel launches. These operations have few parameters but can be slower than large convolutions that cuDNN optimizes well.

So the research direction needs to shift from:

```text
Reducing params/FLOPs on paper
```

to:

```text
Optimizing the Pareto frontier: accuracy / params / latency / memory on real hardware
```

## 2. Research Principles

### 2.1. Do not equate params with speed

A model with fewer parameters can be slower than one with more parameters if it:

- Uses many small kernels.
- Has many small branches, concats, splits, shuffles.
- Has many `permute`, `reshape`, `contiguous`, `scatter_`, `topk`, `argmax` calls.
- Creates many intermediate activations.
- Uses operators that are not backend-optimized.

In HBCC, the parts most likely to cause latency are:

- Multi-head reshape inside the cluster.
- Similarity matrix between centers and points.
- Hard assignment via `max` + `scatter_`.
- Aggregate/dispatch via broadcast + `sum`.
- Channel split/fusion and channel shuffle.
- PointShrink/LBPConv if they create many intermediate maps.

### 2.2. BOPs are only an advantage if inference is truly bitwise

Hamming similarity can replace cosine similarity in binary space:

```math
x_b, y_b \in \{-1, +1\}^D
```

```math
x_b^\top y_b = D - 2 \cdot HD(x_b, y_b)
```

And with a bit-packed representation:

```math
x_b^\top y_b \propto \text{popcount}(\text{XNOR}(x_b, y_b))
```

However, if the implementation still uses `torch.sign()` and `torch.matmul()` on float tensors, the bitwise advantage does not yet exist at runtime. In that case BOPs is only a theoretical metric and should not be used to conclude that a model is faster.

### 2.3. Token count matters more than channel count in CoC

Context Cluster's cost depends strongly on the number of points `N`, the number of centers `M`, and the feature dimension `D`:

```math
O(N \cdot M \cdot D)
```

Therefore, for CIFAR-10 images of `32x32`, to be genuinely lightweight:

- The first stage should not run global clustering on `32x32` without a very strong reason.
- The point count must be reduced early: `32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2`.
- Global clustering should appear at a later stage, once `N` is already small.
- Early stages should use a lightweight local operation or region-local clustering.

### 2.4. Every technique must pass an independent ablation

The following techniques should not be enabled simultaneously and then evaluated as a whole:

- Hamming similarity.
- PointShrink or point reducer.
- LBPConv branch.
- Linear bottleneck.
- Channel shuffle.
- Structured pruning.
- Distillation.

Each technique needs its own ablation, measured under the same protocol, and is only kept if it:

- Improves accuracy at the same cost.
- Reduces latency/memory at equivalent accuracy.
- Or creates a clear trade-off on the Pareto frontier.

## 3. Foundations from the Original Context Cluster Paper

The paper `Image as Set of Points` treats an image as a set of points rather than a convolution grid or a sequence of attention patches. Key components:

### 3.1. Coordinate augmentation

The input image:

```math
I \in \mathbb{R}^{3 \times H \times W}
```

is augmented with 2D coordinates for each pixel:

```math
\left[\frac{i}{H} - 0.5, \frac{j}{W} - 0.5\right]
```

Each point initially has 5 dimensions:

```math
p_i = [R_i, G_i, B_i, x_i, y_i]
```

This is important because CoC wants to treat the image as a set of points. Without coordinates, the model loses an explicit positional signal; the original paper also shows positional information is an important component.

### 3.2. Points reducer

The paper reduces point count by placing anchors evenly in space, gathering the `k` nearest points around each anchor, concatenating features, and using a linear projection to produce new points.

In a grid-ordered image, this reducer can be implemented with equivalent convolution/pooling. What matters is not the module name, but that:

- The point count must actually decrease.
- The spatial size must shrink across stages.
- The reducer must produce a new representation, not just preprocess local features while keeping `H x W` unchanged.

For CIFAR-10, a reasonable reduction schedule is:

```text
Input:  32 x 32
S1:     16 x 16
S2:      8 x 8
S3:      4 x 4
S4:      2 x 2
```

### 3.3. Context cluster operation

A context cluster block consists of:

1. Project points to similarity space.
2. Propose centers evenly distributed in space.
3. Compute cosine similarity between centers and points.
4. Hard-assign each point to its nearest center.
5. Aggregate value features within each cluster.
6. Dispatch the cluster feature back to each point.
7. Multi-head, concat, projection.
8. MLP/channel communication.

Aggregate formula:

```math
g =
\frac{
  v_c + \sum_{i=1}^{m} \sigma(\alpha s_i + \beta) v_i
}{
  1 + \sum_{i=1}^{m} \sigma(\alpha s_i + \beta)
}
```

Where:

- `v_c`: the center in value space, giving an empty cluster a signal.
- `s_i`: similarity between a point and the center.
- `alpha`, `beta`: learnable parameters.
- `sigmoid`: normalizes the score to `(0, 1)`, not softmax.

### 3.4. Why region partition exists

If clustering happens over the whole image at an early stage, cost and memory grow quickly with `N`. Region partition divides the image into small regions and clusters independently within each region to reduce cost. It sacrifices some global receptive field but keeps training/inference feasible.

Therefore, removing region partition entirely should only be considered reasonable when:

- `N` is already small, e.g. at a later stage.
- Or the inference path already has a real bitwise kernel.
- Or benchmarks prove memory/latency are still good.

## 4. Diagnosis of the Current Implementation

The following deviations should be recorded before optimizing:

| Current issue | Why it is a problem | Proposed fix | Metric to verify |
|---|---|---|---|
| Input is still RGB 3-channel | Deviates from the CoC paper, which uses RGB + XY | Add optional coordinate augmentation | Accuracy, convergence, ablation with/without XY |
| HBCC's first stage runs at `32x32` | Token count is too large in the first cluster block | Stem/reducer reduces early to `16x16` | Latency batch 1/128, FLOPs/MACs, accuracy |
| `PointShrink` currently uses stride 1 | Does not reduce point count, only adds local preprocessing | Clearly separate `LocalPreprocess` from `PointReducer` | Shape trace across stages, latency |
| Hamming uses float matmul | No real XNOR/popcount advantage yet | Treat as an accuracy-only ablation for now; BitEdge comes later | Operator profile, latency |
| Profiler only counts Conv/Linear | Missing matmul/scatter/reduction/reshape | Add an operator-aware profiler or torch profiler summary | More complete FLOPs/MACs report |
| LBPConv/pruning not enabled in the main config | Proposal not yet verified in the main pipeline | Enable via a separate ablation, not by default | Accuracy/latency per ablation |
| Pruning mask does not yet export a real smaller model | Soft mask does not speed up inference | Add export/prune/fine-tune | Latency of the exported model |
| Benchmark only at a single batch size of 1 | Does not reflect the paper and is insufficient for deployment | Measure batch 1/16/64/128 | Latency + throughput matrix |

## 5. Evaluation of Current Proposals

### 5.1. Binarized Global Clustering

#### Idea

Replace cosine similarity with Hamming similarity in binary space. With a bit-packed representation, similarity can use XNOR + popcount.

#### What is correct

- The relationship between cosine and Hamming in `{ -1, +1 }` space is well-defined.
- It can reduce the theoretical cost of the similarity path.
- It can reduce the activation memory of similarity if bits are packed correctly.

#### Risks

- With PyTorch float matmul, there is no significant speedup.
- Binarization can lose fine-grained information, especially at early stages.
- Removing region partition from the first stage can increase memory/latency.
- Hamming only replaces the similarity path; aggregate/dispatch still involve broadcast, scatter, sum.

#### Adjusted proposal

Do not call this `Binarized Global Clustering` and apply it by default across the whole network. Instead split it into:

- `Binarized Similarity`: replace cosine with Hamming in the similarity path.
- `Late Global Cluster`: only go global at a later stage when `N` is small.
- `BitEdge Inference`: a real bitwise path, developed after the float/latency model is stable.

#### Criteria to keep

Keep Hamming if at least one condition holds:

- Accuracy drop is negligible and latency is no worse than cosine.
- Accuracy is better than cosine in the same architecture.
- Once a bitwise kernel exists, latency/memory clearly decrease.

### 5.2. Straight-Through Estimator

#### Idea

Since `sign()` has a gradient of nearly zero, use STE to binarize in the forward pass while passing an approximate gradient backward.

#### What is correct

- Necessary for training BNNs.
- Hard-tanh STE in the range `[-1, 1]` is a common choice.

#### Risks

- STE does not guarantee accuracy.
- Early binarization can make optimization harder.
- A scaling factor is needed to compensate for lost amplitude.

#### Adjusted proposal

Add the following mechanisms to the roadmap:

- Learnable scale per head or per channel:

  ```math
  \hat{x} = \gamma \cdot \text{sign}(x)
  ```

- Gradual binarization: warm up training with cosine/full precision before enabling Hamming.
- Distillation from a full-precision teacher.
- Monitor activation distribution before and after sign.

#### Criteria to keep

The STE path is considered stable if:

- Training is stable, without collapse.
- Accuracy does not drop beyond a threshold relative to cosine.
- Binary activations have sufficiently high entropy, not all `+1` or all `-1`.

### 5.3. Hybrid LBPConv Branch

#### Idea

Half the channels go through global/binarized clustering, the other half go through a local branch using fixed LBP filters to preserve high-frequency detail.

#### What is correct

- There is a clear rationale: binarization and global clustering can lose edge/texture detail.
- Fixed LBP filters reduce parameter count.
- The local branch adds inductive bias helpful for CIFAR and small images.

#### Risks

- LBPConv may reduce params but not necessarily latency.
- The intermediate `8C` output can increase memory.
- Sigmoid/QReLU over many maps can create additional small kernels.
- DWConv + PWConv may be faster on GPU due to better-optimized backends.

#### Adjusted proposal

Do not make LBPConv the default. Compare three local branches:

1. `Identity`.
2. `DWConv 3x3 + PWConv`.
3. `LBPConv fixed filters + activation + PWConv`.

#### Criteria to keep

LBPConv should only go into the main model if:

- Accuracy clearly improves over DWConv at equivalent cost.
- Or latency does not increase significantly and params decrease.
- Or on a specific edge target, fixed sparse filters have an optimized kernel.

### 5.4. Learnable Structured Pruning

#### Idea

Learn a channel/center mask to remove unimportant parts. Structured pruning is better than unstructured pruning because it can be exported into a genuinely smaller model.

#### What is correct

- A suitable direction for lightweight design.
- Channel-level pruning can genuinely reduce compute if the model is materialized.
- Can be applied after the architecture has trained stably.

#### Risks

- A soft mask during training does not speed up inference unless exported.
- Crispness loss can push the mask toward 0/1 without guaranteeing the desired sparsity.
- Pruning too early can collapse accuracy.
- Pruning centers in a cluster is more complex than pruning channels because it affects assignment.

#### Adjusted proposal

The pruning workflow should consist of:

1. Train a stable base student.
2. Enable the mask with sparsity regularization.
3. Choose a threshold.
4. Export a model with channels/centers genuinely removed.
5. Fine-tune the exported model.
6. Benchmark the exported model again, not the soft-mask model.

#### Criteria to keep

Pruning is considered successful if the exported model:

- Has fewer params.
- Has lower latency.
- Recovers accuracy after fine-tuning.

## 6. Proposed Architecture Strategy

### 6.1. Two research tracks

#### Track A: HBCC-Latency

Goal: genuinely fast on PyTorch/GPU or a typical edge runtime.

Characteristics:

- Use backend-friendly operators.
- Reduce token count early.
- Reduce the number of small branches.
- Limit unnecessary scatter/topk/permute.
- Hamming is only an ablation until a bitwise kernel exists.
- Prioritize batch-1 latency and batch-128 throughput.

#### Track B: HBCC-BitEdge

Goal: real binary inference on hardware that supports bit operations.

Characteristics:

- Bit-pack the activation similarity path.
- XNOR + popcount kernel.
- Has its own dedicated inference export.
- BOPs is reported alongside real latency.
- Not used to conclude speed if only running PyTorch float.

Track A must be completed before Track B. If Track A has not produced a good Pareto architecture, Track B risks optimizing a kernel for an architecture that is not yet good.

### 6.2. v1 Architecture for CIFAR-10

The goal of v1 is to fix the largest deviations while keeping a reasonable scope.

```text
Input: RGB, optional XY coordinate augmentation

Stem / Point Reducer:
  32x32 -> 16x16
  channels: 3 or 5 -> 32

Stage 1:
  size: 16x16
  op: local lightweight op or region-local cluster
  goal: preserve local detail, avoid expensive global cluster

Stage 2:
  size: 8x8
  op: small HBCC block
  goal: mix local + limited contextual information

Stage 3:
  size: 4x4
  op: global or semi-global HBCC
  goal: context aggregation when token count is small

Stage 4:
  size: 2x2
  op: global cluster or lightweight MLP/conv block
  goal: final semantic feature

Head:
  global average pooling
  layer norm or batch norm depending on best ablation
  linear classifier
```

### 6.3. Initial parameters

Proposed config for `HBCC-Latency-Tiny`:

```python
embed_dims = [32, 48, 96, 160]
depths = [1, 1, 2, 1]
mlp_ratio = 2.0
heads = [1, 2, 2, 4]
head_dim = [16, 16, 16, 16]
centers = [(2, 2), (2, 2), (2, 2), (2, 2)]
```

Proposed config for `HBCC-Latency-Small`:

```python
embed_dims = [32, 64, 128, 192]
depths = [1, 2, 2, 1]
mlp_ratio = 3.0
heads = [1, 2, 4, 4]
head_dim = [16, 16, 16, 16]
centers = [(2, 2), (2, 2), (2, 2), (2, 2)]
```

Rules:

- Do not use one fixed `heads/head_dim` setting for every stage if it wastes compute.
- Do not raise `mlp_ratio` to 4 by default if the goal is lightweight.
- Do not enable a bottleneck/local/global split unless an ablation has proven its benefit.

### 6.4. Local branch policy

Early stages should favor the local branch because:

- The image still has many points.
- Local edge/texture detail is important.
- Early global clustering is expensive and prone to noise.

Order to try:

1. `DWConv 3x3 + BN + GELU + PWConv`.
2. `PointShrink/PointReducer` with stride 2 when spatial reduction is needed.
3. `LBPConv` to verify fixed binary filters.

### 6.5. Similarity path policy

Order to try:

1. Cosine full precision baseline.
2. Hamming simulated via sign + matmul, to measure accuracy.
3. Hamming with learnable scale.
4. Bit-packed Hamming only in Track B.

Do not use BOPs to claim speed at steps 2 and 3.

### 6.6. Distillation policy

Teacher:

- Prefer `ResNet-18 CIFAR` if accuracy is high and stable.
- `CoC baseline full precision` can be used if transferring closer behavior is desired.

Student:

- `HBCC-Latency-Tiny`.
- `HBCC-Latency-Small`.

Default loss:

```math
L = L_{CE}(y, p_s) + \lambda_{KD} T^2 KL(p_t^T || p_s^T)
```

Where:

- `T`: temperature, try `T = 4` by default.
- `lambda_KD`: try `0.5` by default.

Feature distillation should only be enabled if:

- CE + KL still loses too much accuracy.
- The teacher/student have compatible stage feature maps.

## 7. Implementation Roadmap

### Phase 0: Measurement and a correct profiler

Goal: every later conclusion is based on trustworthy measurements.

Work to do:

- Benchmark batch `1, 16, 64, 128`.
- Measure latency and throughput separately.
- Use `torch.profiler` to obtain top operators.
- Report kernel/operator counts where possible.
- Separate CPU/GPU results.
- Use the default benchmark environment: `ImgCaption`.

Deliverables:

- A unified benchmark script.
- A baseline results table for the current state.
- An operator profile for ResNet-18, the CoC baseline, and the current HBCC.

Acceptance:

- Benchmark can be rerun with one command.
- Results record device, torch version, batch size, warmup/runs.

### Phase 1: Baseline matrix

Goal: compare HBCC against the correct baselines.

Minimum models:

- `ResNet-18`: accuracy reference.
- `MobileNetV2` or `ShuffleNetV2`: lightweight latency reference.
- `CoC baseline`: architecture reference.
- `HBCC current`: current proposal reference.

Metrics:

- Top-1 accuracy.
- Params.
- Operator-aware FLOPs/MACs.
- Batch-1 latency.
- Batch 16/64/128 throughput.
- Peak memory.

Acceptance:

- A baseline matrix table exists.
- Every baseline uses the same dataset, transforms, device, and benchmark protocol.

### Phase 2: Fix the CoC/HBCC foundation

Goal: bring the implementation closer to the paper's foundation before optimizing further.

Work to do:

- Add optional coordinate augmentation.
- Add a reducer that shrinks early from `32x32 -> 16x16`.
- Ensure the shape progression across stages is `16 -> 8 -> 4 -> 2`.
- Clearly separate local preprocessing from point reduction.
- Ensure each change can be toggled on/off via config.

Acceptance:

- Shape trace is correct.
- Accuracy does not collapse.
- The new HBCC's latency is better than the current HBCC at batch 1 or batch 128.

### Phase 3: Lightweight Pareto architecture search

Goal: find the best width/depth/head/MLP configuration within a small budget.

Initial grid:

```text
embed_dims:
  A: [24, 48, 96, 160]
  B: [32, 48, 96, 160]
  C: [32, 64, 128, 192]

depths:
  A: [1, 1, 1, 1]
  B: [1, 1, 2, 1]
  C: [1, 2, 2, 1]

mlp_ratio:
  2.0, 3.0

heads/head_dim:
  small per-stage settings, avoid over-wide early stages
```

Do not run full 200-epoch training for every initial grid point. Use a short training proxy:

- 20-50 epochs to filter out weak configurations.
- Full training only for top candidates.

Acceptance:

- At least 3 candidates lie on the Pareto frontier.
- Slow/low-accuracy configurations are eliminated using clear criteria.

### Phase 4: Per-technique ablations

Goal: verify each proposal independently.

Ablation groups:

| Group | Variants | Question |
|---|---|---|
| Coordinate | RGB vs RGB+XY | Does XY help CoC/HBCC? |
| Reducer | stride1 vs stride2 | How does reducing tokens early affect accuracy/latency? |
| Similarity | cosine vs Hamming simulated | Does Hamming preserve accuracy? |
| Local branch | identity vs DWConv vs LBPConv | Which local branch is best on the Pareto frontier? |
| Bottleneck | off vs on | Is the bottleneck worth the latency? |
| Channel shuffle | off vs on | Does shuffle improve accuracy enough to keep? |
| Pruning mask | off vs soft mask | Does the mask learn useful sparsity? |

Acceptance:

- Each technique has a `keep`, `drop`, or `defer` conclusion.
- No technique is kept purely because it is theoretically reasonable.

### Phase 5: Distillation and fine-tuning

Goal: recover accuracy for small students.

Procedure:

1. Train the teacher or use the best available teacher checkpoint.
2. Train the student with CE + KD.
3. Compare against the CE-only student.
4. Try a small temperature and lambda.
5. Fine-tune the best student with a few CE-only epochs if needed.

Acceptance:

- KD improves student accuracy or yields more stable convergence.
- KD does not change latency.

### Phase 6: Structured pruning export

Goal: turn learned sparsity into a genuinely smaller and faster model.

Procedure:

1. Choose the best student.
2. Add a pruning mask for channels or centers.
3. Train with sparsity/crispness regularization.
4. Choose a threshold.
5. Export a model with parts genuinely removed.
6. Fine-tune the exported model.
7. Benchmark the exported model again.

Acceptance:

- The exported model has fewer params.
- The exported model has lower latency than the soft-mask model.
- Accuracy recovers within an acceptable margin.

### Phase 7: HBCC-BitEdge

Only carried out once Track A has a good architecture.

Goal:

- Implement bit-packed Hamming inference.
- Separate the training graph from the inference graph.
- Measure real latency on the target hardware or via a CUDA extension.

Acceptance:

- The similarity path no longer uses float matmul.
- The BOPs report is accompanied by real latency.
- A fallback path exists to verify correctness against simulated Hamming.

## 8. Experiment Design

### 8.1. Required metrics

Every model must report:

| Metric | Meaning | Notes |
|---|---|---|
| `Top-1 Accuracy` | Model effectiveness | CIFAR-10 validation/test |
| `Params` | Model size | Total/trainable |
| `FLOPs/MACs` | Float/int arithmetic cost | Should be operator-aware |
| `BOPs` | Theoretical binary cost | Only used for Hamming/bit path |
| `Latency b1` | Response time | Important for real-time use |
| `Throughput b16/b64/b128` | Batch performance | Compared to paper and deployment batch size |
| `Peak Memory` | Deploy/train feasibility | GPU memory or activation estimate |
| `Operator Profile` | Runtime bottleneck | Top ops from the profiler |

### 8.2. Benchmark protocol

Default:

```text
device: cuda if available
env: ImgCaption
input: 3x32x32 or 5x32x32 if coordinate augmentation is enabled
warmup: >= 30 iterations
runs: large enough to be stable
batch sizes: 1, 16, 64, 128
model.eval()
torch.no_grad()
synchronize before and after timing
```

Do not synchronize after every iteration unless measuring latency very conservatively, since this can penalize small kernels and make large-batch throughput less representative. There should be two modes:

- `latency_strict`: synchronize every iteration.
- `throughput_stream`: synchronize after the whole loop.

### 8.3. Logging format

Each experiment should produce a record like:

```json
{
  "model_name": "hbcc_latency_tiny",
  "config_id": "dims32_48_96_160_depth1121_mlp2",
  "dataset": "cifar10",
  "device": "cuda",
  "torch_version": "...",
  "params": 0,
  "flops": 0,
  "bops": 0,
  "acc1": 0.0,
  "latency_ms_b1": 0.0,
  "throughput_b16": 0.0,
  "throughput_b64": 0.0,
  "throughput_b128": 0.0,
  "peak_memory_mb": 0.0,
  "notes": ""
}
```

### 8.4. Rules for drawing conclusions

A model must not be concluded as better if:

- It only has fewer params but much worse latency.
- It only has lower FLOPs but the profiler shows high operator overhead.
- It only has lower BOPs but inference still uses float matmul.
- It has higher accuracy but params/latency far exceed the lightweight target.

A model is considered Pareto-good if no other model simultaneously has:

- Accuracy higher or equal.
- Params lower or equal.
- Latency lower or equal.
- Memory lower or equal.

and at least one criterion that is clearly better.

## 9. Overall Acceptance Criteria

### 9.1. Research success criteria

At minimum:

- At least one model is clearly smaller than ResNet-18.
- Accuracy only drops within an acceptable threshold.
- Latency is no worse than lightweight baselines such as MobileNetV2/ShuffleNetV2.
- The new HBCC-Latency is faster than the current HBCC at batch 1 and batch 128.

Initial proposed thresholds:

```text
Params: <= 40% of ResNet-18
Accuracy drop: <= 2% absolute vs. the ResNet-18 CIFAR baseline
Latency b1: <= current HBCC, and as close to the lightweight conv baseline as possible
Throughput b128: >= current HBCC
```

These thresholds can be adjusted once the full baseline matrix is available.

### 9.2. Engineering acceptance criteria

- A reproducible benchmark script exists.
- Each ablation has a clear config.
- A standardized results table exists.
- A profiler report exists for the main candidate.
- A shape trace exists for the new architecture.
- A checkpoint and config mapping exists for the best model.

### 9.3. Scientific acceptance criteria

- Every major claim is backed by a measurement.
- BOPs is not used as a substitute for latency.
- Hamming, LBPConv, and pruning each have their own ablation.
- If a technique is dropped, the reason is recorded with a metric.

## 10. Risk Register

| Risk | Severity | Signal | Mitigation |
|---|---:|---|---|
| Hamming reduces accuracy | High | Large accuracy drop when sign is enabled | Distillation, learnable scale, only use at later stages |
| LBPConv is slower than DWConv | Medium | Latency increases despite fewer params | Keep DWConv as default, LBP as an ablation |
| Pruning mask does not speed things up | High | Soft-mask model's params/runtime unchanged | Export a genuinely pruned model |
| Coordinate augmentation complicates the stem | Low | Shape/input mismatch | `use_coord` flag, shape tests |
| Profiler misses operator cost | High | Low FLOPs but high latency | Use torch profiler/operator-aware summary |
| Search too broad, too compute-heavy | Medium | Too many configs fully trained | Short proxy training first |
| BitEdge scope too large | High | Requires custom kernel/export | Defer to Phase 7 |

## 11. Implementation Checklist

### Measurement

- [ ] Benchmark batch `1, 16, 64, 128`.
- [ ] Record device, environment, torch version.
- [ ] Add profiler top operators.
- [ ] Separate latency strict and throughput stream.
- [ ] Save results in a standardized JSON/CSV format.

### Baselines

- [ ] ResNet-18 CIFAR.
- [ ] MobileNetV2 or ShuffleNetV2.
- [ ] CoC baseline.
- [ ] HBCC current.

### Architecture

- [ ] Optional coordinate augmentation.
- [ ] Reducer that shrinks `32x32 -> 16x16`.
- [ ] Stage sizes `16 -> 8 -> 4 -> 2`.
- [ ] Per-stage dims/depths/heads config.
- [ ] Local branch variants.
- [ ] Similarity variants.

### Training

- [ ] Teacher checkpoint.
- [ ] CE-only student baseline.
- [ ] KD student.
- [ ] Optional feature distillation.

### Pruning

- [ ] Train the mask.
- [ ] Choose a threshold.
- [ ] Export the pruned model.
- [ ] Fine-tune the exported model.
- [ ] Benchmark the exported model.

### Reporting

- [ ] Pareto table.
- [ ] Current issue -> fix -> metric table.
- [ ] Ablation summary.
- [ ] Final recommendation.

## 12. Proposed Priority Order

The order of work should be:

1. Measurement/profiler.
2. Baseline matrix.
3. Correct stage size/reducer.
4. Coordinate augmentation.
5. Tiny/small architecture sweep.
6. Distillation.
7. Hamming ablation.
8. Local branch ablation, including LBPConv.
9. Structured pruning export.
10. BitEdge kernel, if still needed.

Do not start with the bitwise kernel or pruning before having a good Pareto architecture. If the underlying architecture is not good, every later optimization may just improve a design that was pointed in the wrong direction.

## 13. Assumptions

- The primary initial dataset is CIFAR-10.
- The preferred benchmark environment is `ImgCaption`.
- ResNet-18 is the accuracy reference, not the lightweight latency baseline.
- The lightweight latency baseline should also include MobileNetV2 or ShuffleNetV2.
- The early phases do not implement a bitwise kernel.
- BOPs in the early phases is only a theoretical metric.
- The main goal is to find a good Pareto model, not to preserve every original idea.
- This file is a research and implementation planning document, not a final result.

## 14. Conclusion

The Lightweight HBCC direction has potential, but needs a shift in focus. Proposals such as Hamming, LBPConv, and structured pruning all have a rationale, but only become a real advantage once correctly implemented and measured.

The research plan should start from the most certain issues:

- Measure latency and operator cost correctly.
- Reduce token count early.
- Stay close to the important components of the original CoC, such as coordinate augmentation and point reduction.
- Find a small Pareto architecture first.
- Only then add Hamming, LBPConv, pruning, and bitwise inference.

A good lightweight model is not the one with the fewest parameters, but the one that achieves the best trade-off between accuracy, latency, memory, and real deployability.
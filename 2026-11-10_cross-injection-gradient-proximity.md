# FMNIST split cross-injection — Gradient proximity (corrected sampling, ckpt 46k)

## Abstract

We study whether **gradient neighborhoods** (peer rankings from random-projected mid-block sensitivities) stay aligned when denoising continues with a **class-split Model B** instead of Model A, starting from the same intermediate state `x_τ^A`. On FashionMNIST split diffusion (checkpoint 46k), we fix 100 noise anchors and inject at τ ∈ {800, 600, 400, 200}. **One-Step A→A** rank NDCG@10 ≈ 1.0; **A→B One-Step** is lower with a clear minimum at **τ = 600** (μ = 0.684), reproduced under two projection seeds. **Full t = 0** probes show strong neighborhood **reorganization even without switching** (A→A Full drops to ≈ 0.26–0.57); switching adds a smaller endpoint gap. We separate **Δ_switch** (model swap) from **Δ_proj** (projection seed disagreement). Effects of trajectory change and Model B's distinct gradient geometry are **not isolated** in this design—disentangling them is future work.

**Scope:** Model A/B class-split unconditional diffusion; shared `x_τ^A` from Model A; injection τ ∈ {800, 600, 400, 200}; mid-block random-projected gradient neighborhoods (N = 100 noise IDs 0–99).  
**Primary source:** `experiments/fmnist_split_cross_injection/full_ckpt46000_phase4_proj2_20261110/merged_report.json` (`validation: ok`). Set 0 (seed 20261109) matches Corrected Phase 3.

**Notation:** `x_τ^A` = Model-A latent immediately **before** the DDIM update at injection τ; τ′ = next scheduler index (799, 599, 399, 199). Display math uses `$$ … $$`.

---

## 1. Overview

We ask whether **gradient neighborhoods**—peer rankings induced by projected mid-block sensitivities—stay aligned with a reference computed at `x_τ^A` under Model A when subsequent denoising uses Model B.

| Case | Model at probe | Gradient timestep |
| --- | --- | --- |
| Reference | A | τ |
| A→A One-Step | A | τ′ |
| A→B One-Step | B | τ′ |
| A→A Full | A | **t = 0** on final image |
| A→B Full | B | **t = 0** on final image |

Phase 4 repeats all gradient metrics under **two projection sets** (seeds 20261109 / 20261110, K = 32 each) with shared DDIM latents (`sampling_shared/`). Cross-set agreement is reported as **exploratory reproducibility across two projection seeds** (n = 2); it is not a general stability certificate.

---

## 2. Experimental Setup

| Setting | Value |
| -------- | ----- |
| Data | FashionMNIST 32×32 grayscale, unconditional |
| Model A classes | 0, 2, 3, 4, 6 |
| Model B classes | 1, 5, 7, 8, 9 |
| Split size | 30,000 images per model |
| Init / training | Same initial weights; **trained independently from identical initial weights** to **EMA step 46000** |
| Sampling | DDIM, η = 0, 1000 train timesteps |
| Noise anchors | Fixed IDs **0–99** (`fixed_noise_1000.pt`) |
| Injection τ | 800, 600, 400, 200 |
| Projections | **Set 0:** seed 20261109; **Set 1:** seed 20261110; K = 32 directions each |
| Gradients | **U-Net mid block only**; weights frozen except mid block for `autograd.grad` |
| Metrics | Rank- and cosine-grade NDCG@5/@10, top-k overlap, random baseline (50 permutations, seed 20261109) |
| Classifier | ONNX on **full** outputs only (predicted labels; auxiliary) |

**Sampling validation (post-fix):** Independent DDIM trajectory max diff = **0**; τ-wise step counts match; A→A and A→B One-Step share identical `x_τ^A`; model `state_dict` unchanged across cases.

---

## 3. Gradient Proximity

Let ε<sub>θ</sub>(x, t) be the U-Net noise prediction. Output dimension 1024 (= 1×32×32 flattened). For each noise anchor *i*, we build a reference neighbor ranking from Model A gradients at **(x_τ^A)<sub>i</sub>, t = τ**, then compare test cases built from the same anchor. **Neighborhood metrics (NDCG, overlap, gaps)** are defined in §5 beside each Figure; this section is the shared **feature pipeline**.

**1. U-Net output flattening**

$$
z_\theta(x,t)=\mathrm{vec}(\epsilon_\theta(x,t))
$$

| Symbol | Meaning |
| --- | --- |
| x | Input latent image at probe timestep (32×32 FMNIST, per anchor). |
| t | Diffusion timestep index for the gradient probe (e.g. τ, τ′, or 0). |
| ε<sub>θ</sub>(x,t) | U-Net noise prediction; shape 1×32×32 before flattening. |
| vec(·) | Column-major flatten to ℝ<sup>1024</sup> (output dimension **1,024**). |
| z<sub>θ</sub>(x,t) | Flattened prediction vector used for random projections. |

**2. Random projection**

$$
s_k=r_k^\top z_\theta(x,t)
$$

| Symbol | Meaning |
| --- | --- |
| k | Projection index, k = 1,…,32 (**32** unit vectors per projection set). |
| r<sub>k</sub> | Fixed random unit vector in ℝ<sup>1024</sup> from set seed (20261109 or 20261110). |
| s<sub>k</sub> | Scalar sensitivity score before backprop (one per projection direction). |

Each r<sub>k</sub> ∈ ℝ<sup>1024</sup> is a unit vector from the projection seed (K = 32 per set; seeds 20261109 / 20261110 in Phase 4).

**3. Mid-block gradient**

$$
g_k=\nabla_{\theta_{\mathrm{mid}}}s_k
$$

| Symbol | Meaning |
| --- | --- |
| g<sub>k</sub> | Gradient of s<sub>k</sub> w.r.t. mid-block weights only (vector length **2,757,632** per k). |
| θ<sub>mid</sub> | U-Net **mid block** parameters (only subset with `requires_grad=True`). |
| ∇<sub>θ mid</sub> | Partial derivatives via `torch.autograd.grad`; weights are not updated. |

**4. Gradient feature concatenation**

$$
G_i=\mathrm{concat}(g_1,\ldots,g_{32})
$$

| Symbol | Meaning |
| --- | --- |
| i | Noise **anchor** index (i ∈ {0,…,99}, **100** fixed anchors). |
| G<sub>i</sub> | Concatenated mid-block gradient feature, dimension **88,244,224** (32 × 2,757,632). |
| concat(·) | Stack flattened g<sub>k</sub> vectors in fixed projection order k = 1,…,32. |

**5. Gradient cosine similarity**

$$
S_{ij}=\frac{G_i^\top G_j}{\|G_i\|_2\|G_j\|_2}
$$

| Symbol | Meaning |
| --- | --- |
| i,j | Anchor indices among the **100** noise IDs (same case and τ). |
| S<sub>ij</sub> | Cosine similarity between gradient features G<sub>i</sub> and G<sub>j</sub> (diagonal S<sub>ii</sub> = 1). |
| ‖·‖₂ | Euclidean norm of the concatenated feature vector. |

Within each case, neighbors of anchor *i* are ranked by test-case S<sub>ij</sub> (self excluded). **Implementation (`gradient_proximity.py`):** only mid-block parameters use `requires_grad`; `torch.autograd.grad` does not update weights. **|G<sub>i</sub>| = 88,244,224**.

---

## 4. Experimental Design

![Experiment flow](visualizations/2026-11-10_cross-injection-gradient-proximity/fig0_experiment_flow.png)

**Figure 0.** Shared pipeline: fixed ε<sub>i</sub> → Model A DDIM to pre-step state at τ → `x_τ^A` → reference / One-Step / Full branches.

Phase 4: one `sampling_shared/` tree per τ; gradient passes differ only by projection bank.

---

## 5. Results

All plotted values are means over **N = 100** noise anchors (IDs 0–99). Phase 4 uses **two projection sets** (seeds 20261109 / 20261110) with shared DDIM latents. Complete Case × τ tables are in **Appendix A**.

### 5.1 Summary — rank NDCG@10 (μ ± *s* across projection sets)

For each scalar metric, **x<sub>0</sub>** and **x<sub>1</sub>** are the Set 0 and Set 1 values (same case, same τ):

$$
\mu=\frac{x_0+x_1}{2}
$$

| Symbol | Meaning |
| --- | --- |
| x<sub>0</sub> | Metric value from **Projection Set 0** (seed 20261109). |
| x<sub>1</sub> | Metric value from **Projection Set 1** (seed 20261110). |
| μ | Arithmetic mean of the two sets for the same case, τ, and metric. |

$$
s=\sqrt{\frac{(x_0-\mu)^2+(x_1-\mu)^2}{n-1}}, \quad n=2
$$

| Symbol | Meaning |
| --- | --- |
| s | Sample standard deviation across the two projection banks (ddof = 1). |
| n | Number of projection sets, **n = 2** (not a hypothesis test). |
| x<sub>0</sub>, x<sub>1</sub>, μ | Same as in the mean formula above. |

This *s* is a **sample** standard deviation across two projection banks only; it is **not** a significance test. Values match `tau_report.json`; rank/cosine NDCG@10 and top-10 overlap μ, *s* also match `merged_report.json` (see audit scripts in footer).

| τ | A→A One μ ± *s* | A→B One μ ± *s* | A→A Full μ ± *s* | A→B Full μ ± *s* |
| ---: | ---: | ---: | ---: | ---: |
| 800 | 0.9961 ± 0.0003 | 0.7377 ± 0.0095 | 0.2634 ± 0.0168 | 0.2632 ± 0.0083 |
| 600 | 0.9989 ± 0.0005 | **0.6839 ± 0.0120** | 0.3957 ± 0.0001 | 0.3461 ± 0.0118 |
| 400 | 0.9990 ± 0.0002 | 0.7686 ± 0.0040 | 0.4968 ± 0.0183 | 0.4611 ± 0.0101 |
| 200 | 0.9991 ± 0.0003 | 0.7832 ± 0.0118 | 0.5687 ± 0.0078 | 0.5489 ± 0.0073 |

### 5.2 Figure 1 — Rank NDCG@10

**Metric.** Rank-grade neighborhood preservation (primary switching metric).

$$
\mathrm{DCG@}k=\sum_{r=1}^{k}\frac{\mathrm{rel}(j_r)}{\log_2(r+1)}
$$

| Symbol | Meaning |
| --- | --- |
| k | Neighborhood size evaluated (we use **k=10** for reported NDCG@10). |
| r | Rank position in the **case** neighbor list (r = 1,…,k). |
| j<sub>r</sub> | Neighbor index at rank r in the evaluated (test) case for anchor i. |
| rel(j<sub>r</sub>) | Relevance of j<sub>r</sub> from the **reference** ranking (rank-grade; see below). |
| log₂(r+1) | Logarithmic rank discount (earlier ranks in the list contribute more). |
| Σ (r = 1…k) | Sum of discounted relevance over the top-k positions in the case ranking. |

$$
\mathrm{NDCG@}k=\frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}
$$

| Symbol | Meaning |
| --- | --- |
| DCG@k | Discounted cumulative gain of the **case** neighbor order at cut-off k. |
| IDCG@k | Ideal DCG if the case order matched the reference ideal top-k order. |
| NDCG@k | Normalized score in [0,1]; mean over **100** anchors per (case, τ, set). |

**Relevance (rank-grade).** From the **reference** similarity row for anchor *i*: top-k neighbors get relevances **k, k−1, …, 1**; all other peers get **0**. For each anchor, rank test-case neighbors by case-specific S<sub>ij</sub>, compute DCG@k; divide by IDCG from the reference ideal order. Average NDCG@10 over 100 anchors per (case, τ, projection set); Figure 1 plots **(Set 0 + Set 1) / 2** for each of four cases vs τ. **Random baseline** (rank NDCG@10 ≈ **0.083–0.084**): permuted test order with 50 draws (seed 20261109), reported in each `tau_report.json`.

![Figure 1](visualizations/2026-11-10_cross-injection-gradient-proximity/fig1_rank_ndcg10.png)

**Key observations.** **A→A One-Step** ≈ **1.0** at all τ. **A→B One-Step** is lowest at **τ = 600** (μ = **0.6839**). **Full** curves sit well below One-Step A→A (see §5.7).

### 5.3 Figure 2 — Cosine NDCG@10

**Metric.** Same DCG / NDCG formulas as §5.2, with cosine-based relevance:

$$
\mathrm{rel}_{\mathrm{cos}}(j)=\frac{S_{ij}^{\mathrm{ref}}+1}{2}
$$

| Symbol | Meaning |
| --- | --- |
| i | Anchor noise index (among **100** anchors). |
| j | Neighbor index (j ≠ i) in the reference similarity row. |
| S<sub>ij</sub><sup>ref</sup> | Reference **gradient cosine** S<sub>ij</sub> at (x<sub>τ</sub><sup>A</sup>, t = τ) from Model A—not a raw cross-model vector comparison. |
| rel<sub>cos</sub>(j) | Soft relevance in [0,1] fed into the same DCG/NDCG formulas as §5.2. |

**Calculation.** Relevance uses the **reference** cosine S<sub>ij</sub><sup>ref</sup> (not rank grades). Every non-self peer can have **positive** relevance, so NDCG stays high even when **ranks** reshuffle. **Rank-grade** (Figure 1) is stricter: only the reference top-k receive positive relevance. High cosine NDCG does **not** mean Model A/B raw gradient **directions** are preserved—it reflects the softer relevance definition. Figure 2: mean of Set 0 and Set 1 per case.

![Figure 2](visualizations/2026-11-10_cross-injection-gradient-proximity/fig2_cosine_ndcg10.png)

**Key observations.** A→A and A→B One-Step cosine NDCG both stay **> 0.98**; switching effects visible under **rank** NDCG are compressed here. Use Figure 1 for switching claims.

### 5.4 Figure 3 — Top-10 neighborhood overlap

$$
\mathrm{Overlap@}k(i)=\frac{\left|N_k^{\mathrm{ref}}(i)\cap N_k^{\mathrm{case}}(i)\right|}{k}
$$

| Symbol | Meaning |
| --- | --- |
| i | Anchor sample (noise ID among **100** anchors). |
| k | Top-k neighborhood size (**k=10** in Figure 3). |
| N<sub>k</sub><sup>ref</sup>(i) | Set of **k** neighbor indices with highest reference S<sub>ij</sub> (self excluded). |
| N<sub>k</sub><sup>case</sup>(i) | Set of **k** neighbor indices with highest case S<sub>ij</sub>. |
| ∩ | Set intersection: neighbors appearing in both top-k lists. |
| &#124;·&#124; | Cardinality (count of elements in a set). |

**Variables.** N<sub>k</sub><sup>ref</sup>(i) and N<sub>k</sub><sup>case</sup>(i) are the top-k neighbor **sets** of anchor *i* (by reference vs case similarity); self excluded; **k = 10**. Example: 7 of the reference top-10 appear in the case top-10 → Overlap@10 = **0.7**. **Order within the top-k is ignored.** Mean over 100 anchors; Figure uses Set 0 / Set 1 average per case.

![Figure 3](visualizations/2026-11-10_cross-injection-gradient-proximity/fig3_top10_overlap.png)

**Key observations.** A→A One-Step overlap ≈ **0.98–1.0**; A→B One-Step drops with rank NDCG (e.g. ≈ **0.61** at τ = 600). Full paths are lower still.

### 5.5 Figure 4 — Projection sensitivity

$$
\Delta_{\mathrm{proj}}=\left|\mathrm{NDCG}_{\mathrm{Set0}}-\mathrm{NDCG}_{\mathrm{Set1}}\right|
$$

| Symbol | Meaning |
| --- | --- |
| Δ<sub>proj</sub> | Absolute difference in the **same** scalar metric under two projection banks. |
| NDCG<sub>Set0</sub> | NDCG from Set 0 (seed 20261109, **32** directions). |
| NDCG<sub>Set1</sub> | NDCG from Set 1 (seed 20261110, **32** directions). |
| &#124;·&#124; | Absolute value (magnitude of set disagreement). |

**Calculation.** Same case, same τ, same 100 anchors; only the **projection seed** (32 directions) changes. Figure 4 uses **rank NDCG@10**. Small Δ<sub>proj</sub> ⇒ similar **reported** NDCG under two banks (**exploratory** n = 2); not neighborhood preservation, and similar NDCG does **not** imply identical neighbor identities. Maxima (rank @10): **A→B One 0.0169 @ τ = 600**; **all cases 0.0259 (A→A Full, τ = 400)**—see §6.1.

![Figure 4](visualizations/2026-11-10_cross-injection-gradient-proximity/fig4_projection_sensitivity.png)

**Key observations.** A→A One-Step bars are near zero; larger bars on A→B and Full paths. Do not confuse with **Δ<sub>switch</sub>** (Figure 5).

### 5.6 Figure 5 — Switching gap (One-Step)

$$
\Delta_{\mathrm{switch}}(\tau)=\mathrm{NDCG}_{A\rightarrow A}(\tau)-\mathrm{NDCG}_{A\rightarrow B}(\tau)
$$

| Symbol | Meaning |
| --- | --- |
| τ | Injection timestep (**800, 600, 400, 200** in this study). |
| Δ<sub>switch</sub>(τ) | Model-switch contrast at fixed τ (Figure 5: One-Step rank NDCG@10). |
| NDCG<sub>A→A</sub>(τ) | Rank NDCG@10 for **A→A One-Step** within one projection set. |
| NDCG<sub>A→B</sub>(τ) | Rank NDCG@10 for **A→B One-Step** within the same set. |

**Calculation.** **Same projection set**; **rank NDCG@10**; **One-Step** cases only. Positive gap ⇒ A→A preserves reference neighborhoods better than A→B at that τ. Bars show Set 0 and Set 1 separately; largest mean gap at **τ = 600 (μ ≈ 0.315)**. This gap mixes **DDIM trajectory change** and **Model B gradient geometry**—not a pure cross-injection update effect (see §7).

![Figure 5](visualizations/2026-11-10_cross-injection-gradient-proximity/fig5_switching_gap.png)

**Key observations.** One-Step switching gap peaks at **τ = 600**. Set-to-set variation in the gap (up to **≈ 0.017** at τ = 200) is tabulated in Appendix A (switching table).

### 5.7 Full vs One-Step — rank NDCG@10 (μ)

Reference: **(x_τ^A, t = τ)**. Full cases: **(x<sub>0</sub>, t = 0)** on completed images. Even **without** switching, **A→A Full** rank NDCG drops far below One-Step A→A:

| τ | A→A Full μ | A→B Full μ | μ difference (A→A − A→B) |
| ---: | ---: | ---: | ---: |
| 800 | 0.2634 | 0.2632 | 0.0002 |
| 600 | 0.3957 | 0.3461 | 0.0496 |
| 400 | 0.4968 | 0.4611 | 0.0357 |
| 200 | 0.5687 | 0.5489 | 0.0199 |

Full-denoising shows **substantial neighborhood reorganization without model switching**; the extra switching-related gap at the endpoint is **comparatively small** at several τ (e.g. **0.0002** at τ = 800). Full rank NDCG **rises** as injection moves toward completion (τ 800 → 200).

---

## 6. Projection reproducibility

| Set | Projection seed | Vectors |
| ---: | ---: | ---: |
| 0 | 20261109 | 32 |
| 1 | 20261110 | 32 |

Set 0 matches Corrected Phase 3 (same seed); merge reports `validation_warnings_phase3_parity: []`.

### 6.1 Maximum $\Delta_{\mathrm{proj}}$ (from Phase 4 JSON, recomputed)

| Quantity | Scope | Max value | τ | Case / metric |
| --- | --- | ---: | ---: | --- |
| Rank NDCG@10 | A→B One-Step only | **0.0169** | 600 | A→B One |
| Rank NDCG@10 | **All cases** | **0.0259** | 400 | A→A Full |
| Cosine NDCG@10 | All cases | 0.003870 | 400 | A→A Full |
| Top-10 overlap | All cases | 0.0260 | 400 | A→A Full |

**Do not** summarize projection sensitivity as a single “~0.013” without specifying metric and case. A→A One-Step rank @10 has $|\Delta_{\mathrm{proj}}|\lesssim 7\times 10^{-4}$ at all τ; larger rank $\Delta_{\mathrm{proj}}$ appears on **Full** paths and some **A→B** One-Step τ values.

Interpretation: **exploratory reproducibility across two projection seeds**—qualitative patterns align, but n = 2 does not establish general projection stability.

---

## 7. Discussion

### One-Step: what changes under A→B?

**Reference** gradients use Model A at **(x_τ^A, t = τ)**. **A→A One-Step** applies one DDIM step with A, then probes at **(x_{τ′}^{AA}, t = τ′)**—latent and timestep move, **same model**.

**A→B One-Step** applies one DDIM step with B, then probes at **(x_{τ′}^{AB}, t = τ′)**—latent, timestep, **and UNet** all differ from the reference construction path.

The observed A→B neighborhood change reflects **both** the altered denoising trajectory and the use of **Model B's gradient geometry**. These effects are **not isolated** in the current experiment; we cannot attribute the full rank NDCG drop to a single cross-injection DDIM update alone.

**Observations (rank NDCG@10):** A→A μ ≈ **0.996–0.999** at all τ. A→B μ ≈ **0.684–0.783**, well above random (~0.083). **τ = 600** is the **minimum** for A→B One-Step (μ = **0.6839**; both projection sets).

**τ = 600 dip — observation, not mechanism.** The largest immediate gap vs A→A appears at τ = 600, not at the noisiest (800) or cleanest (200) injection. We observe this in **both** projection sets. A plausible hypothesis is that split-specific gradient neighborhoods differ most in that noise regime, but we have **not** separated causes (τ / latent / model geometry) and treat the mechanism as an **open question** for presentation: *Why 600?* → consistent pattern, cause not yet identified.

### Full denoising: not “collapse from B alone”

Full-denoising results show **substantial neighborhood reorganization even without model switching** (A→A Full μ ≈ **0.26–0.57** vs One-Step A→A ≈ 1.0). The reference is at **t = τ** on x_τ^A; Full probes at **t = 0** on completed images—different **state and time index**.

The additional **switching-related** gap at the endpoint is **comparatively small** at several τ (e.g. τ = 800: A→A vs A→B Full μ differ by **0.0002**). Full rank NDCG **increases** as injection moves toward completion (τ 800 → 200). See §5.7.

### Projection reproducibility

Qualitative patterns appear in **both** seeds (**exploratory**, n = 2). **Rank** metrics show larger Δ_proj than **cosine** on some cells.

### Switching effect (Δ_switch vs Δ_proj)

One-Step rank Δ_switch (μ) peaks at **τ = 600** (0.315). Across projection sets, One-Step Δ_switch differs by up to **0.0172** (τ = 200). **Figure 5** = model swap within a seed; **Figure 4** = seed disagreement.

**Classifier (auxiliary):** Set 0 mean P(split-A group) on full outputs: A→A ≈ 67%; A→B ≈ 18% / 22% / 34% / 50% at τ 800 / 600 / 400 / 200.


## 8. Limitations

- **A→B One-Step confounding:** Trajectory change (x, t) and Model B's gradient map are coupled; no B-only baseline at the same (x_{τ′}, t) probe.
- **Projection sets n = 2** (exploratory reproducibility only).
- 100 fixed noise IDs; FashionMNIST 32×32; small unconditional U-Net.
- Mid-block gradient features only; Full probe at **t = 0**, not in-step DDIM Jacobians.
- Classifier uses predicted labels.
- Phase 4 ran with `--no-plots` in-run; report figures rebuilt from JSON via `build_corrected_phase4_report_figures.py`.

---

## 9. Conclusion

Same-model **One-Step** denoising almost perfectly preserves rank gradient neighborhoods. **Switching to Model B** yields substantial reorganization under rank-grade metrics, with the **largest One-Step gap at τ = 600**—an observed pattern whose mechanism remains open. **Full t = 0** probes show large realignment **even for A→A**, so endpoint metrics must not be read as switch-only collapse. Two projection seeds give **exploratory** agreement on the main pattern.

**Next step:** Separate the effect of **changing the model's gradient representation** from the effect of the **DDIM update and trajectory** (e.g. matched probes, B-only baselines at shared states).

---

## Appendix A — Complete metric tables

Tables list every Case × τ combination. *s* is the sample standard deviation with **n = 2** projection sets (ddof = 1). μ, *s* formulas: §5.1.

#### Rank NDCG@5

| Case | τ | Set 0 | Set 1 | μ | *s* |
| --- | ---: | ---: | ---: | ---: | ---: |
| A→A One | 800 | 0.9940 | 0.9909 | 0.9925 | 0.0022 |
| A→A One | 600 | 0.9990 | 0.9988 | 0.9989 | 0.0001 |
| A→A One | 400 | 0.9965 | 0.9983 | 0.9974 | 0.0013 |
| A→A One | 200 | 0.9985 | 0.9971 | 0.9978 | 0.0010 |
| A→B One | 800 | 0.6575 | 0.6479 | 0.6527 | 0.0068 |
| A→B One | 600 | 0.5966 | 0.5881 | 0.5923 | 0.0060 |
| A→B One | 400 | 0.6694 | 0.6892 | 0.6793 | 0.0140 |
| A→B One | 200 | 0.6742 | 0.6905 | 0.6824 | 0.0115 |
| A→A Full | 800 | 0.1524 | 0.1840 | 0.1682 | 0.0223 |
| A→A Full | 600 | 0.2749 | 0.2652 | 0.2701 | 0.0068 |
| A→A Full | 400 | 0.3585 | 0.3909 | 0.3747 | 0.0229 |
| A→A Full | 200 | 0.4564 | 0.4778 | 0.4671 | 0.0151 |
| A→B Full | 800 | 0.1789 | 0.1694 | 0.1741 | 0.0067 |
| A→B Full | 600 | 0.2462 | 0.2242 | 0.2352 | 0.0156 |
| A→B Full | 400 | 0.3422 | 0.3589 | 0.3506 | 0.0119 |
| A→B Full | 200 | 0.4549 | 0.4579 | 0.4564 | 0.0021 |

#### Rank NDCG@10

| Case | τ | Set 0 | Set 1 | μ | *s* |
| --- | ---: | ---: | ---: | ---: | ---: |
| A→A One | 800 | 0.9963 | 0.9960 | 0.9961 | 0.0003 |
| A→A One | 600 | 0.9993 | 0.9986 | 0.9989 | 0.0005 |
| A→A One | 400 | 0.9989 | 0.9992 | 0.9990 | 0.0002 |
| A→A One | 200 | 0.9993 | 0.9989 | 0.9991 | 0.0003 |
| A→B One | 800 | 0.7444 | 0.7311 | 0.7377 | 0.0095 |
| A→B One | 600 | 0.6924 | 0.6755 | 0.6839 | 0.0120 |
| A→B One | 400 | 0.7658 | 0.7714 | 0.7686 | 0.0040 |
| A→B One | 200 | 0.7748 | 0.7915 | 0.7832 | 0.0118 |
| A→A Full | 800 | 0.2515 | 0.2752 | 0.2634 | 0.0168 |
| A→A Full | 600 | 0.3957 | 0.3956 | 0.3957 | 0.0001 |
| A→A Full | 400 | 0.4839 | 0.5097 | 0.4968 | 0.0183 |
| A→A Full | 200 | 0.5632 | 0.5742 | 0.5687 | 0.0078 |
| A→B Full | 800 | 0.2691 | 0.2574 | 0.2632 | 0.0083 |
| A→B Full | 600 | 0.3545 | 0.3377 | 0.3461 | 0.0118 |
| A→B Full | 400 | 0.4682 | 0.4539 | 0.4611 | 0.0101 |
| A→B Full | 200 | 0.5540 | 0.5437 | 0.5489 | 0.0073 |

#### Cosine NDCG@5

| Case | τ | Set 0 | Set 1 | μ | *s* |
| --- | ---: | ---: | ---: | ---: | ---: |
| A→A One | 800 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| A→A One | 600 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| A→A One | 400 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| A→A One | 200 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| A→B One | 800 | 0.9904 | 0.9896 | 0.9900 | 0.0006 |
| A→B One | 600 | 0.9798 | 0.9793 | 0.9796 | 0.0004 |
| A→B One | 400 | 0.9848 | 0.9854 | 0.9851 | 0.0005 |
| A→B One | 200 | 0.9905 | 0.9904 | 0.9905 | 0.0000 |
| A→A Full | 800 | 0.9372 | 0.9376 | 0.9374 | 0.0003 |
| A→A Full | 600 | 0.9483 | 0.9502 | 0.9493 | 0.0013 |
| A→A Full | 400 | 0.9568 | 0.9623 | 0.9595 | 0.0039 |
| A→A Full | 200 | 0.9775 | 0.9794 | 0.9785 | 0.0014 |
| A→B Full | 800 | 0.9448 | 0.9425 | 0.9437 | 0.0016 |
| A→B Full | 600 | 0.9413 | 0.9385 | 0.9399 | 0.0020 |
| A→B Full | 400 | 0.9543 | 0.9547 | 0.9545 | 0.0003 |
| A→B Full | 200 | 0.9764 | 0.9757 | 0.9761 | 0.0005 |

#### Cosine NDCG@10

| Case | τ | Set 0 | Set 1 | μ | *s* |
| --- | ---: | ---: | ---: | ---: | ---: |
| A→A One | 800 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| A→A One | 600 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| A→A One | 400 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| A→A One | 200 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| A→B One | 800 | 0.9908 | 0.9900 | 0.9904 | 0.0005 |
| A→B One | 600 | 0.9821 | 0.9813 | 0.9817 | 0.0006 |
| A→B One | 400 | 0.9878 | 0.9881 | 0.9880 | 0.0003 |
| A→B One | 200 | 0.9924 | 0.9928 | 0.9926 | 0.0003 |
| A→A Full | 800 | 0.9420 | 0.9422 | 0.9421 | 0.0002 |
| A→A Full | 600 | 0.9527 | 0.9543 | 0.9535 | 0.0011 |
| A→A Full | 400 | 0.9633 | 0.9672 | 0.9652 | 0.0027 |
| A→A Full | 200 | 0.9813 | 0.9824 | 0.9818 | 0.0007 |
| A→B Full | 800 | 0.9476 | 0.9473 | 0.9474 | 0.0002 |
| A→B Full | 600 | 0.9455 | 0.9434 | 0.9445 | 0.0015 |
| A→B Full | 400 | 0.9616 | 0.9600 | 0.9608 | 0.0011 |
| A→B Full | 200 | 0.9802 | 0.9798 | 0.9800 | 0.0003 |

#### Top-5 overlap

| Case | τ | Set 0 | Set 1 | μ | *s* |
| --- | ---: | ---: | ---: | ---: | ---: |
| A→A One | 800 | 0.978 | 0.980 | 0.979 | 0.001 |
| A→A One | 600 | 0.998 | 1.000 | 0.999 | 0.001 |
| A→A One | 400 | 0.988 | 0.994 | 0.991 | 0.004 |
| A→A One | 200 | 0.996 | 0.992 | 0.994 | 0.003 |
| A→B One | 800 | 0.578 | 0.562 | 0.570 | 0.011 |
| A→B One | 600 | 0.538 | 0.520 | 0.529 | 0.013 |
| A→B One | 400 | 0.626 | 0.622 | 0.624 | 0.003 |
| A→B One | 200 | 0.612 | 0.618 | 0.615 | 0.004 |
| A→A Full | 800 | 0.174 | 0.188 | 0.181 | 0.010 |
| A→A Full | 600 | 0.292 | 0.294 | 0.293 | 0.001 |
| A→A Full | 400 | 0.336 | 0.394 | 0.365 | 0.041 |
| A→A Full | 200 | 0.418 | 0.448 | 0.433 | 0.021 |
| A→B Full | 800 | 0.188 | 0.172 | 0.180 | 0.011 |
| A→B Full | 600 | 0.262 | 0.242 | 0.252 | 0.014 |
| A→B Full | 400 | 0.334 | 0.354 | 0.344 | 0.014 |
| A→B Full | 200 | 0.408 | 0.396 | 0.402 | 0.008 |

#### Top-10 overlap

| Case | τ | Set 0 | Set 1 | μ | *s* |
| --- | ---: | ---: | ---: | ---: | ---: |
| A→A One | 800 | 0.980 | 0.985 | 0.983 | 0.004 |
| A→A One | 600 | 0.996 | 0.991 | 0.994 | 0.004 |
| A→A One | 400 | 0.995 | 0.996 | 0.996 | 0.001 |
| A→A One | 200 | 0.996 | 0.996 | 0.996 | 0.000 |
| A→B One | 800 | 0.652 | 0.650 | 0.651 | 0.001 |
| A→B One | 600 | 0.617 | 0.592 | 0.605 | 0.018 |
| A→B One | 400 | 0.680 | 0.683 | 0.681 | 0.002 |
| A→B One | 200 | 0.691 | 0.701 | 0.696 | 0.007 |
| A→A Full | 800 | 0.275 | 0.284 | 0.279 | 0.006 |
| A→A Full | 600 | 0.386 | 0.398 | 0.392 | 0.008 |
| A→A Full | 400 | 0.451 | 0.477 | 0.464 | 0.018 |
| A→A Full | 200 | 0.514 | 0.514 | 0.514 | 0.000 |
| A→B Full | 800 | 0.266 | 0.259 | 0.263 | 0.005 |
| A→B Full | 600 | 0.340 | 0.333 | 0.336 | 0.005 |
| A→B Full | 400 | 0.429 | 0.409 | 0.419 | 0.014 |
| A→B Full | 200 | 0.505 | 0.482 | 0.493 | 0.016 |

#### Projection sensitivity Δ_proj (|Set 0 − Set 1|)

Absolute difference between projection sets for the same case and τ. Columns: rank NDCG@10, cosine NDCG@10, top-10 overlap.

| Case | τ | Δ_proj rank@10 | Δ_proj cos@10 | Δ_proj top-10 |
| --- | ---: | ---: | ---: | ---: |
| A→A One | 800 | 0.0004 | 0.000013 | 0.0050 |
| A→A One | 600 | 0.0007 | 0.000001 | 0.0050 |
| A→A One | 400 | 0.0003 | 0.000000 | 0.0010 |
| A→A One | 200 | 0.0004 | 0.000000 | 0.0000 |
| A→B One | 800 | 0.0134 | 0.000762 | 0.0020 |
| A→B One | 600 | 0.0169 | 0.000867 | 0.0250 |
| A→B One | 400 | 0.0056 | 0.000375 | 0.0030 |
| A→B One | 200 | 0.0167 | 0.000404 | 0.0100 |
| A→A Full | 800 | 0.0237 | 0.000237 | 0.0090 |
| A→A Full | 600 | 0.0001 | 0.001625 | 0.0120 |
| A→A Full | 400 | 0.0259 | 0.003870 | 0.0260 |
| A→A Full | 200 | 0.0110 | 0.001033 | 0.0000 |
| A→B Full | 800 | 0.0117 | 0.000342 | 0.0070 |
| A→B Full | 600 | 0.0167 | 0.002066 | 0.0070 |
| A→B Full | 400 | 0.0143 | 0.001540 | 0.0200 |
| A→B Full | 200 | 0.0103 | 0.000408 | 0.0230 |

#### Switching gap Δ_switch (rank NDCG@10, A→A − A→B)

| τ | One-Step Set 0 | One-Step Set 1 | One-Step μ | Full Set 0 | Full Set 1 | Full μ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 800 | 0.2519 | 0.2649 | 0.2584 | -0.0176 | 0.0179 | 0.0001 |
| 600 | 0.3069 | 0.3231 | 0.3150 | 0.0413 | 0.0579 | 0.0496 |
| 400 | 0.2331 | 0.2278 | 0.2305 | 0.0156 | 0.0558 | 0.0357 |
| 200 | 0.2245 | 0.2073 | 0.2159 | 0.0092 | 0.0305 | 0.0199 |
---


---

*Raw assets:* `full_ckpt46000_phase4_proj2_20261110/` (`merged_report.json`, `tau_*/projection_set_*/tau_report.json`, `sampling_shared/`). Corrected Phase 3 parity for Set 0. No training reruns.*

*Figure build:* `build_corrected_phase4_report_figures.py`. *Audit (read-only):* `validate_phase4_report_data.py`, `audit_report_figures.py` — last run 2026-11-10, merged cross-check 0 failures, Fig 1–5 present and consistent with JSON.*

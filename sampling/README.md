# Activation Sampling (TongueX)

`pool_50000.txt` (uniform LHS over `[0,1]^11`) is **not usable as a training
distribution**. Diagnosis, then the replacement.

## 1. Why the LHS pool fails

Independent uniform draws on 11 axes concentrate: the sum of 11 iid U(0,1)
variables is `5.5 ± 0.96` by CLT. So *every* sample is a whole-tongue
co-contraction.

| diagnostic | LHS 50k | stratified v2 |
|---|---|---|
| effort (sum of 11 activations) | 5.50 ± 0.96 (never < 1.61) | 2.73 ± 1.80 (0 → 8.93) |
| muscles > 0.5 per sample | 5.50 | 2.19 |
| sparse samples (≤3 on, ≥8 off) | **0 / 50,000** | 33.6 % |
| near-rest samples (effort < 0.3) | **0 / 50,000** | 3.8 % |
| L2 gap to rest pose | 0.655 | **0.000** |
| worst gap to a single-muscle posture | 0.787 | **0.000** |
| median gap to literature articulatory anchors | 0.587 | 0.022 |

Three consequences:

1. **The counterfactual XAI claim (0714.md §13, §21) has no data support.**
   Muscle importance `I_i = L(M') − L(M)` under a single-muscle intervention
   requires the surrogate to have seen single-muscle responses `∂M/∂a_i`. The old
   pool contains literally zero such samples, and never comes within 0.56–0.79
   (L2) of one.
2. **The rest pose does not exist in the dataset.** Nothing to anchor
   displacement `U = V − V₀` against.
3. **Real rtMRI contours will be OOD.** Speech activation is sparse and
   low-dimensional; the pool occupies a thin dense-co-contraction shell that
   real articulation never visits.

## 2. Replacement design

Effort and direction are decoupled instead of sampling each muscle independently:

```
effort    E ~ stratified over [0.05, 5.5]
direction d ~ Dirichlet(α),  α ∈ {0.15 … 2.5}   # α ≪ 1 ⇒ sparse simplex
a = clip(E · d, 0, cap)
```

plus deterministic blocks that *guarantee* the primitives exist:

| block | share | purpose |
|---|---|---|
| `REST` | 1 | reference pose `M₀` |
| `SINGLE` | 132 | 11 muscles × 12 levels — **basis of the counterfactual claim** |
| `PAIR` | 880 | 55 pairs × 4×4 — 2nd-order co-contraction (nonlinear; not derivable from singles) |
| `TRIPLE` | 6 % | random sparse 3-muscle draws |
| `ANCHOR` | 14 % | literature-informed articulatory synergies + heavy perturbation |
| `EFFORT` | 62 % | effort/Dirichlet stratified core |
| `LHS` | 18 % | legacy uniform LHS, retained **as an OOD split**, not as training density |

The pool is shuffled, so **any prefix is a balanced subsample** — you can stop the
ArtiSynth run early, and the 10k/25k/50k dataset-utility ablation (§16.4) is just
`head -n`.

### On the `ANCHOR` block

The synergy table in `muscles.py` is approximate, distilled from EMG /
biomechanics literature (Baer 1988; Miyawaki 1975; Buchaillard–Perrier–Payan
2009; Gerard 2006). It is **not ground truth for this ArtiSynth model** and is
never used as supervision or as an evaluation target — it only biases *sampling
density* toward the region that plausibly matters. Perturbation is deliberately
generous (effort rescale 0.4–1.8×, per-muscle jitter, 15 % muscle dropout, 25 %
unexpected recruitment) so the support is not pinned to the prior. Edit the table
freely.

## 3. Sequences

The old sequence file drew 4 pool indices **uniformly at random** (2,796 were
degenerate A-B-A-B oscillations; only 32 % had 4 distinct keyframes). Transitions
between two arbitrary co-contraction blobs are not achievable control paths.

`build_sequences.py` implements what 0714.md §6.2 already specifies: restrict to
watchdog-VALID poses → build a kNN pose graph → walk it, biased to start and end
near neutral. Every hop is a graph edge, i.e. a bounded, achievable control step.

**Build the graph in shape space, not activation space** (`--descriptors`). Two
distant activations can produce the same tongue; what must be smooth is the
*shape* trajectory. Falls back to activation space if descriptors are absent.

## 4. Usage

```bash
# 1. static pool (do this now)
python sampling/build_pool.py --n 200000 --out datasets/pool_v2_200000.txt

# 2. audit / compare
python sampling/audit_pool.py \
    OLD=datasets/pool_50000.txt NEW=datasets/pool_v2_200000.txt

# 3. run the ArtiSynth static pass -> validity labels + shape descriptors

# 4. sequences (AFTER validity labels exist)
python sampling/build_sequences.py \
    --pool datasets/pool_v2_200000.txt \
    --validity datasets/validity.csv \
    --descriptors datasets/shape_desc.npy \
    --n-seq 200000 --seq-len 4 \
    --out datasets/sequences_v2.txt
```

## 5. What "coverage" may and may not claim

`N^(1/11)` for N = 200,000 is **3.0 grid levels per muscle**. No pool of any
realistic size fills `[0,1]^11`. **Do not report activation-space fill distance as
a coverage result** — a reviewer with a calculator will use it against you.

The defensible claim is in *shape* space, anchored to real data:

```
real midsagittal contour c_k   (GT_Segmentations, Subject 1–5, all frames)
        ↓ landmark-aligned (palate + jaw)
g_k = min over valid meshes M_i of  d_SDF( c_k , R_π(M_i) )
        ↓
C(ε) = fraction of real frames with g_k ≤ ε,  ε anchored to segmentation
       noise (≈ 1 px ≈ 1.5–3 mm)
```

Report `C(ε)` per subject and per phone class. Uncovered phone classes are both a
stated limitation **and** the target set for the next adaptive sampling round.

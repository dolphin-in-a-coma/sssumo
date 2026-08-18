# Improvement backlog

Findings from the reproduction study (`VALIDATION.md`) that are worth acting on,
with the evidence that motivated each. Ordered by value per unit of effort.

## Performance

**Organic evaluation re-reads every CSV three times.**
`evaluate_on_organic_data` constructs `OrganicDataset` *inside* the noise loop and
`OrganicDataset.__init__` does a full `pd.read_csv`, so a full evaluation performs
21 reads of files up to 567 MB where 7 would do. Stage-2 fine-tuning is dominated
by statistics extraction rather than training, so caching parsed trials is
probably the single largest available speedup.

## Model directions the code already anticipates

**Train with noiseless samples.** `snr_distribution: [10, 50]` means SNR ∞ is
always extrapolation — precisely where the reproduced pretrained checkpoint fails
on crank, via intermittent detection dropout on smooth signal. One config change,
one ~2 h run, and the validation harness already measures the outcome.

**The primitive shape is frozen and symmetric.** `BASE_RECONSTRUCTOR_PARAMS` pins
`beta_mean=(0.5, 0.0)` and `beta_precision=(6.0, 0.0)` with
`freeze_primitive_parameters=True`. The second element of each pair is a
*duration slope* the architecture supports and nothing uses; real minimum-jerk
submovements are slightly asymmetric.

**Reconstruction loss cannot reach the detection channel.**
`gradient_for_detection` defaults to `False`, so the straight-through mask is
detached and onsets are trained only by the BCE terms. A `'NegativeOnly'` mode is
implemented and never exercised; it would let reconstruction error suppress false
positives.

**The 4-channel reconstruction-mask path is dead.** Every config ends
`channels: [..., 3]`, yet `models.py` and the training loop both carry full
`shape[1] == 4` branches with their own loss term. Exercise it or delete it — an
unexercised branch inside a loss function is a liability.

## Evaluation rigour

**More seeds.** All variance estimates rest on n = 2 replicates per stage. Four or
five would let the seed-variance figures carry intervals of their own; ~4 h on two
L4s with no code changes.

**A canonical cluster bootstrap.** `hierarchical_bootstrap_metrics` resamples
participants into *row weights* and then draws rows i.i.d. with the row count
pinned — an approximation of a cluster bootstrap, not the textbook one, which
keeps every row of each selected participant. ~15 lines, no GPU (per-trial dumps
already exist), and worth checking whether the intervals move.

**Split allocation for the small datasets.** The alternating participant split
leaves pointing with 2 test participants and crank/object_moving with 5. A
stratified split, or leave-one-participant-out for those three, would use all
5/10/11 subjects instead of 2/5/5.

## Code health

**No tests.** A reconstructor round-trip (labels → signal → detect → labels) would
be a handful of lines and would guard the component everything else depends on.

**`Config` mutates itself during evaluation.** `evaluate_on_organic_data` assigns
`config.snr_distribution = noise_condition` and never restores it. Harmless in
stage 1; in stage 2 that object is shared with dataset construction.

**pandas chained assignment** at `utils.py:576` raises under pandas 3.

**`np.random.choice` on a possibly-empty array** at `utils.py:580`. Left as a
documented sharp edge rather than patched, since the original code survives its
own pipeline at realistic trial counts.

## Data

**Handwriting participant count disagrees with the article.** Table 1 reports 91
subjects; the shipped CSV contains **89**. Every other dataset matches exactly.
Worth resolving before publication — either two participants were dropped in
preprocessing, or the table counts something slightly different.

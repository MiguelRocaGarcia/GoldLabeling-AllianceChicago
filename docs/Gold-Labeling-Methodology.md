# Gold-Label Construction for Patient Entity Resolution

*Source material for the research paper. Written to be reformatted / trimmed into the
Methods section later. Numbers marked `<PLACEHOLDER>` still need to be pulled from the
AllianceChicago VM (see §9 for the snippet that produces them).*

---

## 1. Motivation

Evaluating an entity-resolution system on real patient data requires ground truth, and no
ground truth exists: the AllianceChicago MDM population has no curated set of known
same-patient record pairs. Hand-labeling the full candidate set is not feasible either —
blocking produces **204,805** candidate pairs, and adjudicating each one by eye at even a
few seconds per pair would take months of annotator time.

The strategy adopted here is a **weak-supervision-then-targeted-correction** design, in the
spirit of programmatic labeling (Snorkel-style) but with a human correction pass that is
*deliberately biased toward the pairs most likely to be mislabeled*:

1. **Silver labels.** Cheap, automatic labels for all 204,805 candidate pairs, produced by
   combining two independent weak labelers.
2. **Error-targeted human review.** Rather than sampling uniformly (which would spend most
   of the annotation budget confirming easy, obviously-correct labels), the candidate set is
   sliced with hand-designed *suspicion queries* — logical conditions under which a silver
   label is likely to be wrong. Each slice is then adjudicated by a human, individually or
   in bulk.
3. **Gold labels.** The final label for a pair is the human decision where one exists, and
   the silver label otherwise. Each reviewed pair additionally carries an **ambiguity flag**
   recording whether a human could be confident about the decision at all.

The result is a gold set that concentrates annotation effort where it changes the label,
while still assigning a defensible label to every candidate pair.

---

## 2. Inputs

| Artifact | Description |
|---|---|
| `MDM_Population_cleaned_v5_2026_06_21.parquet` | Per-record cleaned MDM population. Cleaning rules in `docs/Data-Cleaning-Guide.md` (raw + `*_clean` + derived `full_name_tokens`, `full_name_compact`, `Phones_set`; `valid_record` flag). |
| `candidate_pairs_*.parquet` | Output of the blocking stage: 204,805 candidate record pairs `(PATID_A, PATID_B)`. |
| `anymatch_predictions_full_v3.csv` | Binary prediction + `match_prob` for every candidate pair, from the AnyMatch (GPT-2) model fine-tuned on the synthetic AllianceChicago-shaped corpus. |
| `deterministic_rules_predictions_full_v1.csv` | `rule_pred ∈ {match, non_match, review}` plus a traceable `rule_id` / `rule_reason` for every candidate pair, from the hand-authored rule cascade (`deterministic_rules/RULES.md`). |

The two predictors are **methodologically independent**: one is a learned neural matcher
trained on synthetic pairs, the other is an explicit, hand-authored cascade of comparators.
They fail in different ways, which is what makes their disagreement a useful error signal
(§5.3).

Comparison fields used throughout (per side, suffixed `_A` / `_B`):
`FirstNM_clean`, `MiddleNM_clean`, `LastNM_clean`, `SuffixNM_clean`, `BirthDT_clean`,
`SSN_clean`, `last_4_SSN`, `AddressLine1_clean`, `AddressLine2_clean`, `CityNM_clean`,
`ZipCD_clean_base`, `ZipCD_clean_ext`, `StateCD_clean`, `Email_clean`,
`SexAtBirthDSC_clean`, `Phones_set`, `full_name_compact`.

---

## 3. Silver labels

Silver labels are the **disjunction (logical OR)** of the two weak labelers
(`data/silver_labels/get_silver_labels.ipynb`):

```
silver_label = (rule_pred == 'match') OR (anymatch_pred == 1)
```

Note that the rule engine's third outcome, `review`, is folded into the negative side of
the disjunction: a pair the rules defer on is silver-positive only if AnyMatch says match.

This union is **deliberately recall-biased ("aggressive")**. A pair is silver-positive if
*either* labeler calls it a match. The rationale is that the human pass that follows is far
better at *removing* false positives from an explicit, inspectable slice than at *finding*
false negatives hidden among ~200k mostly-negative pairs; so the automatic stage is tuned to
over-produce positives and let the human prune, rather than to be conservative and leave
misses undiscoverable.

The silver stage yields `silver_labels_v1_2026_06_21.csv`: one boolean per candidate pair.

---

## 4. Review tooling

Two mechanisms were built (`gold_labeling/`), both writing to the same sparse store.

**The label store.** `gold_labels_v1.csv` is *sparse*: it contains one row per pair a human
has actually touched, keyed by `(PATID_A, PATID_B)`, with columns

```
PATID_A, PATID_B, gold_label ∈ {match, no_match}, ambiguous_pair ∈ {True, False}, reviewed_at
```

Untouched pairs never appear. This makes the store append-only in spirit, restartable across
sessions, and trivially auditable — every human decision has a timestamp, and re-labeling a
pair overwrites its row rather than duplicating it.

**(a) Pair-by-pair review — `labeler.py`.** A small Flask app bound to `127.0.0.1` renders a
scrollable, colour-coded grid: the two records of a pair are stacked as adjacent rows and
each field cell is shaded **green** (both present and equal), **red** (both present and
different), **yellow** (present on one side only) or **grey** (missing on both sides). The
existing silver label is displayed so the reviewer sees what is being corrected. Match /
no-match and ambiguous / OK are single keystrokes (`1`, `2`, `3`), and every click writes
through to the CSV immediately, so a browser or kernel crash loses nothing.

The colour coding is the reason thousands of pairs could be adjudicated at reasonable speed:
the reviewer does not read fields, they read a pattern of colour, and only stop to read text
where the pattern is unusual.

*PHI note.* The rendered page contains real patient values in the browser DOM. The server
binds to loopback only, runs on the HIPAA-tier VM, and the store is excluded from version
control.

**(b) Bulk adjudication.** `add_gold_label_pairs(subset, is_match, is_ambiguous)` writes the
same decision for every pair in a slice at once (with an automatic dated backup of the store
before each write). `add_indv_gold_label(patid_a, patid_b, ...)` overrides a single pair.
Bulk adjudication was used only when a slice was inspected, found to be **homogeneous** —
i.e. a random scroll through it showed the same verdict again and again with no counter-
examples — and was too large to click through individually.

---

## 5. The review protocol: error-targeted slicing

The core of the method. Rather than sample pairs at random, the candidate set was queried
with conditions that make a *silver-label error likely*. Three families of query were used.

Two helper predicates make the queries precise about missingness, and are worth stating
explicitly because they define what "agrees" and "disagrees" mean throughout:

- `_equal(f)` — field `f` is **present on both sides and equal**. A missing side never counts
  as agreement.
- `_differ(f)` — field `f` is **present on both sides and different**. A missing side never
  counts as a contradiction.

so `~_equal(f)` covers *both* "different" and "missing", and the two predicates are not
complements. `_phone_overlap()` is set intersection over `Phones_set`, and
`_no_strong_match()` is true when *no* strong identifier agrees (SSN, last-4 SSN, DOB, email,
phone, full name).

### 5.1 Suspicious-negative queries (candidate false negatives)

Pairs the silver labels call **no-match** despite carrying evidence that they are the same
person. Examples actually run:

- equal full SSN, silver says no-match;
- equal email **and** equal DOB, silver no-match;
- equal `full_name_compact` + equal DOB + equal address, silver no-match;
- equal first + middle (both populated) + last + DOB, silver no-match;
- equal first + last + DOB and the **leading house number of the address** agrees, silver no-match;
- equal first name + DOB + phone overlap, silver no-match;
- phone overlap + equal DOB, silver no-match;
- equal first name + equal DOB + phone overlap, different last name, missing/differing address and email (the *surname-change* pattern);
- equal last name + DOB + address but different first name.

### 5.2 Suspicious-positive queries (candidate false positives)

Pairs the silver labels call **match** despite carrying a contradiction:

- `_differ('SSN_clean')` and silver says match;
- `_differ('BirthDT_clean')` and silver says match;
- `_no_strong_match()` and silver says match — i.e. the labelers matched two records on
  *nothing strong at all*;
- equal name + DOB but different sex at birth;
- equal DOB + last name, different SSN (potential twins).

### 5.3 Disagreement queries (the two labelers contradict each other)

Because the silver label is an OR, any pair where AnyMatch and the rules disagree is a pair
where exactly one of them determined the label — a natural high-yield error region. Both
directions were mined, and further sub-sliced by which evidence was present:

- `anymatch_pred == True` **and** `rule_pred == 'non_match'`, sub-sliced by: equal email /
  differing first *and* last name / equal address / no phone overlap / the remainder;
- `anymatch_pred == False` **and** `rule_pred == 'match'`, sub-sliced by: equal SSN / equal
  SSN with differing first name / agreeing leading house number / phone overlap / agreeing
  middle initial / no phone overlap / the remainder.

### 5.4 Ordering and exhaustion

Queries were run in a **cascade**, from the most specific and most decisive to the most
general. Every query after the first is conjoined with `gold_label.isna()` — "not already
adjudicated" — so each slice contains only pairs no earlier, sharper query had already
settled. This does three things: it prevents double-labeling, it makes the *residual* of a
family explicit (e.g. after the strong-evidence sub-slices of "equal first + DOB, silver
no-match" were resolved, the remaining 841 pairs were what was left with no
corroborating evidence, and were labeled as an ambiguous match), and it means the reviewer
always sees, at each step, the hardest remaining version of the problem.

Slices were consumed in one of three ways:

| Disposition | When | Example from the notebook |
|---|---|---|
| **Individual review** in the labeler UI | slice small, or heterogeneous (mixed verdicts) | "same last name + DOB + address, different first name" (1,230 pairs); "surname-change" candidates (298); the SSN-collision and disagreement residuals |
| **Bulk label** | slice large **and** homogeneous on inspection | equal first + last + DOB + phone overlap → 15,302 pairs, all match, not ambiguous |
| **Bulk label as ambiguous match** | slice large and homogeneous *in the sense that the human is uniformly unsure* | first + last + DOB agree, address/email/phone absent or disagreeing, no SSN/middle/suffix on at least one side → 20,717 pairs, labeled match **and** ambiguous |

### 5.5 Easy cases are not reviewed

Pairs that no suspicion query selects are, by construction, pairs on which both weak labelers
agreed and no contradiction pattern fires. These **retain their silver label** and were never
shown to a human. This is the explicit budget decision of the method: annotation effort is
spent only where it can change a label.

---

## 6. The ambiguity flag

Human review of real FQHC data made it clear that a binary match/no-match target is a
misrepresentation of the problem. There is a substantial band of pairs on which **an expert
human cannot be certain either way, and no additional evidence in the record can resolve it**.
The canonical case: two records agreeing on first name, last name and date of birth, with
every distinguishing field (SSN, middle name, suffix, address, phone, email) missing on at
least one side. These are indistinguishable from a common-name/shared-DOB coincidence — and
from a household of same-named relatives — using the information available.

Every reviewed pair therefore carries a **second, orthogonal annotation**: `ambiguous_pair`.
It records not the class of the pair but the **confidence of the annotator** in the class
assigned. A pair can be a confident match, an ambiguous match, or a confident no-match. The
label and the confidence are stored in separate columns, and can therefore be consumed
separately downstream.

The flag is not a scientific nicety — it maps directly onto the deliverable. The production
system exposes a **three-way outcome**, `match / no-match / to_review`, with the third routing
to a human queue at AllianceChicago. `ambiguous_pair = True` is precisely the training and
evaluation target for that third class: a model that confidently auto-merges the pairs a human
annotator could not confidently merge is a model that will silently corrupt the master patient
index.

Conventions applied when setting the flag (derived during review, applied consistently):

- **Match, not ambiguous** — a *typo-shaped* contradiction on an otherwise strongly corroborated
  pair: a transposed or single-digit-off SSN, a day/month-swapped DOB, a misspelled name, a
  present-vs-missing suffix, a differing sex-at-birth with everything else agreeing. These are
  clerical artifacts, not evidence of a different person.
- **Match, ambiguous** — the pair agrees on first + last + DOB (or first + DOB) but *nothing
  else corroborates*, either because the corroborating fields are missing or because they
  disagree. Also: completely different SSNs with an agreeing address and phone.
- **No-match** — a contradiction that a clerical error does not plausibly explain.

---

## 7. Composing the final gold label

```python
final_gold_label = (gold_label == 'match')  if pair was reviewed
                   else  silver_label
```

so:

| Column | Meaning |
|---|---|
| `PATID_A`, `PATID_B` | the candidate pair |
| `final_gold_label` | boolean; human decision where one exists, silver label otherwise |
| `ambiguous_pair` | boolean; `True` only where a human reviewed the pair and was not confident |

Written to `final_gold_labels_v1_2026_07_05.csv`, one row for each of the 204,805 candidate
pairs. Note that `ambiguous_pair = False` is therefore *two different things*: "a human looked
and was confident" and "no human looked". Any analysis of the ambiguous class must condition on
the pair being present in the sparse `gold_labels_v1.csv` store.

---

## 8. Descriptive statistics

| Quantity | Value |
|---|---|
| Candidate pairs (post-blocking) | 204,805 |
| Silver positives | 51,067 (24.9% of candidates) |
| Pairs touched by a human (rows in `gold_labels_v1.csv`) | 52,138 (25.5% of candidates) |
| — of which individually reviewed in the labeler UI | 2,856 |
| — of which bulk-labeled | 49,282 |
| Reviewed → match | 51,554 |
| Reviewed → no-match | 584 |
| Reviewed → flagged ambiguous | 28,050 (53.8% of reviewed; 13.7% of all candidates) |
| Silver labels **overturned** by review | 12,111 (23.2% of reviewed) — 11,789 silver false negatives, 322 silver false positives |
| Final gold positives | 62,537 (30.5% of candidates) |

Three things in this table matter for the paper.

**The error-targeted slicing worked.** 23.2% of everything a human looked at had its silver
label overturned. This is the measured yield of the method: under uniform sampling the expected
overturn rate is the silver error rate over the whole population — which, given that review
found ~12k errors in ~205k pairs, is on the order of a few percent at most. Targeting raised the
per-pair annotation yield by roughly an order of magnitude, which is the entire reason a set of
this size was labelable by one annotator.

**The silver labels erred overwhelmingly in one direction: they missed matches.** 11,789 false
negatives against 322 false positives, a 37:1 ratio. This is *not* what the recall-biased OR
construction (§3) was designed to produce — a disjunction of two labelers should over-call
positives, not under-call them. What it says is that both weak labelers are individually far too
conservative on real FQHC data: even taking their union, they miss matches at scale. The bulk of
the misses are the patterns catalogued in §10 — misspelled surnames with an agreeing first name
and DOB, typo'd SSNs, addresses that differ in formatting but agree on the house number. The
human pass moved the positive rate from 24.9% to 30.5% of candidate pairs, a **22% relative
increase in the number of matches**.

**Over half of the reviewed pairs are ambiguous.** 28,050 of 52,138 — and 13.7% of the entire
candidate set — are pairs a human could label but not label *confidently*. That is the single
strongest empirical argument for the three-way `match / no-match / to_review` deliverable (§6,
§11): a binary system deployed on this population would be forced to guess on one candidate pair
in seven.

Note also the shape of the reviewed set: 51,554 matches against 584 no-matches. The review pass
was, in effect, almost entirely a hunt for missed matches, because that is where the suspicion
queries pointed. The gold set's negatives are therefore mostly *inherited* from the silver
labels rather than human-confirmed — see §11's caveat and §12.

---

## 9. Reproducing the statistics

Run on the VM, with the notebook's `pairs_gold` in scope:

```python
gold  = pd.read_csv(GOLD_LABELS_CSV)         # sparse store: reviewed pairs only
final = pd.read_csv(FINAL_GOLD_LABELS_CSV)   # all 204,805 pairs

print('candidates          ', len(final))
print('silver positives    ', pairs_gold.silver_label.sum())
print('reviewed            ', len(gold))
print('  gold_label        ', gold.gold_label.value_counts().to_dict())
print('  ambiguous         ', gold.ambiguous_pair.sum())
print('final positives     ', final.final_gold_label.sum())
print('ambiguous overall   ', final.ambiguous_pair.sum())

# overturns: reviewed pairs whose human decision contradicts the silver label
rev = pairs_gold[pairs_gold.gold_label.notna()]
flips = rev[(rev.gold_label == 'match') != (rev.silver_label == True)]
print('overturned          ', len(flips))
print('  silver FN (no->yes)', ((flips.gold_label == 'match') & (~flips.silver_label)).sum())
print('  silver FP (yes->no)', ((flips.gold_label == 'no_match') & ( flips.silver_label)).sum())
```

Individual-vs-bulk cannot be recovered from the store directly (both write the same rows);
approximate it by the `reviewed_at` timestamps — bulk writes share a single timestamp across
many rows, individual clicks do not:

```python
ts = gold.reviewed_at.value_counts()
bulk_ts = ts[ts > 5].index          # a timestamp shared by >5 pairs == a bulk write
print('bulk      ', gold.reviewed_at.isin(bulk_ts).sum())
print('individual', (~gold.reviewed_at.isin(bulk_ts)).sum())
```

---

## 10. Empirical observations from the review

These came out of the manual pass and are findings in their own right — they characterize the
error modes of real FQHC registration data, and several of them directly motivated changes to
the deterministic rules and to the synthetic training corpus.

- **SSN typos are pervasive**, and they are recognisably *handwriting-transcription* errors:
  3↔8, 1↔7, and digit transpositions. An equal SSN is therefore strong evidence, but an
  *unequal* SSN is weak evidence of a non-match — which invalidates the naive rule "different
  SSN ⇒ different person".
- **DOB typos** occur too, most commonly month/day transposition.
- **Name misspellings are extremely common**, especially in first names, and coexist with
  legitimate variation (nicknames: *Nick* vs *Nickolas*; abbreviations; middle name written in
  full on one record and as an initial on the other).
- **The leading house number of the address is a surprisingly strong identifier.** Full address
  strings disagree constantly (formatting, apartment, abbreviation), but the street number
  agrees or disagrees cleanly. A slice defined by name + DOB + agreeing house number was
  uniformly matches (2,285 pairs) — enough to justify promoting it to a deterministic rule.
- **Sex-at-birth disagreement is frequently a data-entry error**, not evidence of a different
  person, when everything else agrees.
- **Some genuinely-same-looking people carry entirely different SSNs**, and it is not a typo —
  a reminder that SSN is not reliably one-per-person in this population.
- **The hardest region is: same last name, same DOB, same address, different first name.** It
  mixes misspellings, nickname variants, and genuinely different household members (siblings,
  parent/child with a shared DOB is rare but twins are not). Both weak labelers produce false
  positives and false negatives here at high rates; almost all of these pairs were reviewed
  individually.
- Conversely, **first name + DOB agreeing with a misspelled last name** was systematically
  *under*-called by the silver labels.

---

## 11. Downstream use

The gold set is used for **both** training and evaluation:

- **Training.** `final_gold_label` supervises a supervised matcher trained on real
  AllianceChicago pairs (as opposed to the synthetic-corpus fine-tune).
- **Evaluation.** The gold set is the common yardstick for every method in the study —
  the deterministic rule cascade, zero-shot AnyMatch, synthetic-fine-tuned AnyMatch, and the
  supervised model — so that they are comparable on the same pairs.
- **The `to_review` class.** `ambiguous_pair` supervises / evaluates the third outcome of the
  deployed system: pairs that must be routed to a human queue in the AllianceChicago dashboard
  rather than auto-merged.

*Caveat to carry into any evaluation:* the gold set is not an i.i.d. sample of the candidate
pairs. It is the candidate population with the *most error-prone regions* corrected. Metrics
computed on all 204,805 pairs therefore inherit whatever silver-label errors survive in the
un-reviewed easy region — they are an upper bound on the true agreement between a model and a
human. Metrics computed on the reviewed subset alone are, conversely, computed on a
deliberately adversarial sample and will *understate* real-world performance. Both should be
reported, and the difference between them is informative.

---

## 12. Limitations

- **Single annotator.** All labels were produced by one reviewer (the author). No second pass
  and no inter-annotator agreement statistic is available, so annotator-specific bias in the
  conventions of §6 cannot be quantified. The `ambiguous_pair` flag partially mitigates this by
  isolating the decisions that were subjective in the first place.
- **Recall-biased review.** Only pairs selected by a suspicion query were examined; silver
  false negatives in regions no query covers remain in the gold set.
- **The negative class is largely un-audited.** Of the 52,138 reviewed pairs only 584 were
  labeled no-match; the ~142k gold negatives are almost entirely inherited silver negatives.
  Given that review overturned 11,789 silver negatives in the slices it *did* examine, the
  surviving negative class should be assumed to still contain missed matches, and recall
  measured against this gold set is correspondingly an **over**-estimate.
- **Bulk labeling trades granularity for coverage.** Homogeneity of a bulk-labeled slice was
  established by inspection, not exhaustively, so a small number of within-slice exceptions may
  have been swept up in the majority verdict.

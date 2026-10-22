# COURO Two-Week Progress & Learnings Ledger

**Ledger ID:** `COURO-OPS-20261022-001`  
**Version:** `1.0`  
**Prepared (UTC):** `2026-10-22T02:11:22Z`  
**Window covered:** `2026-10-08T00:00:00Z` → `2026-10-22T02:11:22Z`  
**Prepared by:** Manus  
**Classification:** Internal, non-sensitive operating record  
**Status:** **Verified consolidation with explicit evidence gaps**

> **Boundary:** This ledger records product, research, workflow, and technical-validation progress only. It contains no credentials, personally identifiable information, raw movement/video material, customer data, health-adjacent personal data, diagnostic claims, treatment claims, injury predictions, readiness conclusions, or safety conclusions about a person.

## Executive readout

The two-week period produced a meaningful control-plane and verification advance, with the strongest evidence in the COURO Work OS project. The selected `Cam-Couro/couro-vision-validation` repository did not receive a main-branch commit during this window; its latest verified main-branch evidence remains the October 1 per-dataset validity analysis. This distinction is preserved rather than inferred away.

1. **Research Wing became a unified Work OS surface.** The Research Wing Pulse was mounted in the Command Center and Second Brain, exposing source-linked corpus state, review and Curator queues, evidence trails, sparse-coverage recommendations, and a bounded daily handoff timeline. A live Q&A extension was added using bounded metadata/excerpts, structured output, citation validation, and explicit unanswered-question and limitation fields.
2. **The Research Wing stayed inside a safe source-only boundary.** The automation tracker records corpus persistence, Curator/review controls, taxonomy-gap discovery, protocol export, and direct bridge-dispatch verification. The surface is decision support and review planning; it does not store raw source bodies, credentials, PII, raw movement/health-adjacent data, or unapproved customer material.
3. **Report Operations closed important synthetic-only safety gaps.** Work OS checkpoint `7f01fef671f944753195bee4284b20e3135dce5e` (2026-10-17) added explicit `speed_v1` field-shape validation, account-bound cancellation obligations across lost-create replay, retry-pending state before awaits, originating-account authorization before pending cancellation side effects, stale-render fences, and explicit `cancelled_before_submission` completion. The checkpoint reports 31 focused tests, TypeScript, production build, and diff checks passing; no live calls, media upload, processing, deployment, or access change occurred.
4. **The validation evidence still points to structural evaluation risks, not a simple data-volume story.** Repository commit `a988446` (2026-10-01) records that pooled CCC can be inflated by cohort clustering, that camera-side effects are consistent with a left/right handling defect, and that LoA is a more range-independent absolute-agreement measure for bounded claims. This is a prior evidence baseline, not new activity in the current window.
5. **No production, paid-provider, GPU, raw-media, or sensitive-data action was authorized by this consolidation.** Remaining runtime refresh, deployment, provider, data-governance, and external-commitment decisions stay gated.

## Material progress

| Lane | Verified progress | Evidence | Current state |
|---|---|---|---|
| Research Wing Pulse | Shared compact/expanded component, Command Center and Second Brain integration, corpus/review/Curator queues, sparse-coverage signals, and daily handoff timeline. | Work OS record `2026-10-15T0428Z--couro-research-wing-pulse--v1.md`; Work OS checkpoint `348d36b5` named by the record. | Code and focused verification complete. |
| Research Wing Q&A | Bounded retrieval over approved Vault metadata/excerpts; structured synopsis/answer; citation IDs validated against retrieved records; unanswered questions, sparse signals, limitations, loading/error/no-match states. | Same Research Wing record; TODO `TODO_MJJhHJrqbvwiBdS4t53bz6.md`. | Code complete; managed runtime refresh and real cited-answer confirmation remained pending in the source record. |
| Research automation and governance | Corpus sync reconciliation, explicit sync-lag/failed-heartbeat visibility, Curator/review pipeline, taxonomy coverage/gap synthesis, source-level review filters, protocol export, and controlled discovery dispatch. | `TODO_ILg6ziqmzt5wtEXkdCtAij.md`; `TODO_research-pulse-sync-repair-20261015.md`. | Tracker marks the listed implementation work complete; final deploy/refresh follow-through remains explicit rather than assumed. |
| Report Operations cancellation/auth safety | Added field-shape checks, durable account-bound cancellation obligations, retry-pending state, originating-account authorization, stale-render fences, and pre-submission cancellation terminal state. | Work OS commit `7f01fef671f944753195bee4284b20e3135dce5e`. | Verified checkpoint; no live processing or deployment. |
| Synthetic sparse-coverage control plane | Synthetic-only admission and bounded sparse-to-dense handoff; unknown, excluded, failed, and unobserved intervals remain explicit. | `docs/REPORT_RUNNER_SPARSE_COVERAGE.md`; `docs/REPORT_OPS_SYNTHETIC_BENCHMARK.md`. | Local synthetic baseline only; no real-media connection or GPU authorization. |
| Validation repository | Latest main evidence diagnoses pooled-CCC inflation, within-dataset spread limitations, camera-side reversal, and a need to fill observed range gaps rather than simply add more volume. | `Cam-Couro/couro-vision-validation` commit `a988446`; `harness/per_dataset_validity.py` referenced by that commit. | No main-branch commits in the requested window; open PR #2 predates the window. |

## What changed in the operating model

### 1. Source-linked research is now a visible control-plane capability

The Research Wing is no longer only a backend corpus or tracker. The shared Pulse gives the Founder a single read-model surface for corpus receipt state, review queues, evidence trails, sparse-coverage recommendations, and daily handoff timing. The Q&A path is intentionally bounded: it retrieves only approved metadata/excerpts, requires structured output, validates citations, and returns limitations rather than silently filling gaps.

### 2. Runtime state is separate from code-complete state

The October 15 record explicitly distinguishes corrected code/checkpoint state from the managed runtime still serving the prior permission boundary. The durable lesson is that a passing typecheck, test suite, and build establish implementation evidence, not live deployment evidence. Any future status should retain separate fields for code, checkpoint, managed runtime, and browser acceptance.

### 3. Cancellation is an ownership and race-safety problem, not only a UI problem

The October 17 checkpoint treats cancellation as an account-bound obligation that must survive lost-create replay, retry windows, and final awaits. The explicit retry-pending state prevents a cancel action from targeting the previously displayed job, while originating-account authorization prevents a pending side effect from crossing account boundaries. The pre-submission terminal state makes the final user-visible outcome explicit.

### 4. Validation claims must distinguish agreement, discrimination, and cohort range

The validation repository’s October 1 analysis establishes that pooled CCC can look strong when separated cohorts create artificial range, while within-dataset CCC can collapse when the true between-subject spread is smaller than error. LoA is identified as a range-independent absolute-agreement measure. Future summaries must state the metric, cohort scope, view/laterality mapping, and limitations instead of using an undifferentiated “validated slots” headline.

### 5. Synthetic-only boundaries remain a productive default

The Work OS’s sparse-coverage and visual-stage work demonstrates useful workflow behavior without accepting raw media, source paths, providers, GPUs, persistence, or production actions. Unknown and unobserved intervals remain visible; they are not converted into negative observations. This preserves a safe path for testing control-plane behavior while governance and Founder approvals remain open.

## Evidence index

| Evidence | Safe reference | What it establishes |
|---|---|---|
| Work OS Research Wing record | `docs/operating-records/2026-10-15T0428Z--couro-research-wing-pulse--v1.md` | Pulse delivery, bounded Q&A, tests/build, permission-boundary correction, and managed-runtime limitation. |
| Work OS Research Wing tracker | `TODO_MJJhHJrqbvwiBdS4t53bz6.md` | Pulse and Q&A workstream completion states and checkpoint reference. |
| Research automation tracker | `TODO_ILg6ziqmzt5wtEXkdCtAij.md` | Corpus, Curator, review, taxonomy, export, and dispatch workstream states. |
| Sync-repair tracker | `TODO_research-pulse-sync-repair-20261015.md` | Sync-lag and reconciliation actions; final deploy/refresh remained unchecked. |
| Work OS report-operation checkpoint | `7f01fef671f944753195bee4284b20e3135dce5e` | Cancellation/auth/race-safety changes; 31 focused tests, TypeScript, build, and diff checks. |
| Synthetic sparse-coverage contract | `docs/REPORT_RUNNER_SPARSE_COVERAGE.md` | Synthetic-only admission, explicit abstentions, bounded escalation, and prohibited real-media/provider paths. |
| Synthetic benchmark protocol | `docs/REPORT_OPS_SYNTHETIC_BENCHMARK.md` | CPU-only process baseline and explicit non-accuracy/non-GPU interpretation. |
| Validation repository baseline | `Cam-Couro/couro-vision-validation@a988446` | October 1 per-dataset validity analysis and metric interpretation. |
| Validation repository activity check | `git log --since=2026-10-07` and GitHub events/PR list | No main-branch commit during the current window; PR #2 remains open from October 1. |

## Decisions, constraints, and remaining gates

### Confirmed constraints

- COURO remains a non-clinical decision-support system. Nothing in this ledger is a diagnostic, treatment, injury-prediction, readiness, or medical claim.
- Use only synthetic, de-identified, or explicitly Founder-approved data. This package contains only safe metadata, summaries, commit identifiers, and file references.
- Do not treat Curator scores, review flags, sparse-coverage recommendations, or validation metrics as product approval, license clearance, clinical evidence, or permission to process new data.
- No credentials, tokens, raw media, source bodies, PII, health-adjacent personal data, or unapproved customer materials belong in this ledger, archive, or GitHub commit.
- Provider spend, GPU dispatch, production deployment, managed-runtime refresh, external communication, retention, consent, and sensitive-data governance remain Founder-controlled gates.

### Remaining gates / follow-through

1. **Managed runtime refresh and browser acceptance:** The October 15 record states that the code/checkpoint permission correction was not yet confirmed on the managed runtime. A separately authorized publish/refresh action is required before claiming end-to-end live Q&A acceptance.
2. **Research sync finalization:** The sync-repair tracker leaves final validation, checkpoint, and deploy unchecked. Reconcile those states before representing the repair as live operational completion.
3. **Validation repository freshness:** If additional work occurred outside `couro-vision-validation` after October 1, it needs a safe commit, receipt, or other evidence before being added to the validation progress narrative. This ledger does not manufacture recency.
4. **Data and retention governance:** Before any move beyond synthetic or explicitly approved inputs, record purpose, access path, consent/authority, retention period, deletion verification, report audience, and owner-scoped access.
5. **GPU/provider work:** The synthetic baseline does not authorize a provider, GPU, spend, raw-media intake, or production expansion. Any such step requires a new bounded Founder decision and evidence plan.

## Packaging receipt

- **Ledger document:** `docs/operating-records/2026-10-22T0211Z--couro-two-week-progress-ledger--v1.md`
- **Archive:** `docs/operating-records/2026-10-22T0211Z--couro-two-week-progress-ledger--v1.zip`
- **Archive contents:** this ledger and `SHA256SUMS.txt` only.
- **Filing locations:** selected GitHub repository and canonical COURO Work OS `docs/operating-records/` folder.
- **Sanitization:** confirmed by construction; no secrets, credentials, PII, raw media, source bodies, or unapproved customer material included.
- **GitHub upload:** pending final local verification, commit, and push in `Cam-Couro/couro-vision-validation`.
- **Work OS checkpoint:** pending after the canonical file is written and verified.

## Final state

**Current state:** `complete` for the safe consolidation and filing package; `awaiting Founder decision` for any managed-runtime refresh, production deployment, provider/GPU action, sensitive-data governance, retention policy, or external commitment.

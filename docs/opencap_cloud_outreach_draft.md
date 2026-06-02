# OpenCap Cloud / Stanford NMBL Outreach — Draft

**Prepared:** 2026-05-28
**For:** Cameron Van (cam@couro.io), Founder, Couro
**Purpose:** Commission a custom mocap collection with 2 rear-facing cameras to close Couro's rear-view validation gap.

---

## 1. Key finding: "OpenCap Cloud" is not a productized rental service

OpenCap itself is the free 2-phone cloud-processed app, available at app.opencap.ai for educational/research use only. The OpenCap web app is governed by an academic-use license and explicitly is **not** open to commercial use without a separate agreement.

The commercial spin-out is **Model Health** (modelhealth.io), founded late 2024 by Antoine Falisse (CEO), Scott Uhlrich (CSO), and Nicolas Bellemans (CPO/COO). Model Health is the entity that supports commercial coaches, clinicians, and sports teams. They raised a $1M pre-seed in 2025 led by APEX Capital. **This is who Cameron should be talking to for a commercial validation collection.**

There is also Stanford's **Human Performance Laboratory** within the Neuromuscular Biomechanics Lab (NMBL, Delp/Hicks group) at 341 Galvez St. — that's the wet lab with the 27-camera Motion Analysis Vicon-style system where the original OpenCap validation work was done. Uhlrich is Director of Research there. That's the facility most likely to be able to physically execute a custom 5-camera collection with 2 rear-facing cameras.

So the right play is: lead with Uhlrich (he sits at the intersection of NMBL lab + Model Health commercial), CC Falisse (commercial lead), and let them route.

---

## 2. Primary contact

**Scott Uhlrich, PhD**
Director of Research, Human Performance Laboratory, Stanford
Co-founder & CSO, Model Health
**Email: suhlrich@stanford.edu**
LinkedIn: https://www.linkedin.com/in/scott-uhlrich
Lab address: 341 Galvez St., Stanford, CA 94305
Lab phone: (650) 721-2547

**Why Uhlrich first:** He runs the physical lab where a custom Vicon + multi-camera collection would actually happen, and he's the commercial co-founder, so he can speak to both the science and the commercial engagement terms in one thread.

## 3. Suggested CC

**Antoine Falisse, PhD**
Co-founder & CEO, Model Health
**Email: afalisse@stanford.edu**
LinkedIn: https://www.linkedin.com/in/antoine-falisse

The OpenCap project explicitly directs commercial-use questions to both suhlrich@ and afalisse@. CC'ing Falisse signals this is a commercial conversation, not an academic one.

## 4. Backup / fallback contacts

If neither responds in ~7-10 days:

- **mobilize-center@stanford.edu** — general OpenCap / Mobilize Center inbox; routes to admin staff who can forward.
- **Łukasz Kidziński, PhD** — lukasz.kidzinski@stanford.edu — NMBL postdoc, computer vision / biomech crossover, co-founder of Saliency.ai. Reasonable warm path if Cameron wants a CV-focused conversation.
- **Jennifer Hicks, PhD** — Mobilize Center Executive Director, NMBL — last-resort routing if commercial conversation needs admin escalation. Search for current email on https://nmbl.stanford.edu/people/.
- **Model Health website contact form** — https://www.modelhealth.io/ (likely has a contact/demo form).

---

## 5. Pricing & turnaround — what's public

**Public pricing for custom mocap collection: none.** Neither OpenCap, Model Health, nor NMBL publishes rates for one-off commercial mocap rentals. Cameron should expect to negotiate.

For reference frame:
- The Stanford Human Performance Lab runs a 27-camera Motion Analysis Vicon-style system; a half-day to full-day session for 1-2 subjects in a comparable academic lab typically lands in the **$2-5K range** when sold to industry, plus data processing fees. $2-4K is in the right ballpark but on the low end — Cameron should be ready to flex up to ~$5K or offer a smaller subject count (1 subject instead of 2) if the price comes back higher.
- Turnaround for raw marker data is typically 1-2 weeks post-collection; for processed OpenSim IK results, 2-4 weeks is realistic. Cameron's 2-4 week timeline is tight but feasible if scope stays small.

**Ground truth they can provide (based on the OpenCap validation paper, Uhlrich et al. 2023 PLoS Comp Bio):**
- Raw Vicon marker trajectories (.c3d or .trc)
- Scaled OpenSim model + inverse kinematics joint angles (.mot)
- Likely force plate GRF data (the lab has them; ask)
- Synchronized video from all cameras

**Camera config — unknowns to ask about:**
- Their standard OpenCap validation used 2 iPhones at front-oblique angles (~30-60° off front-facing). Cameron's ask of 5 cameras with 2 rear-facing (~150° and ~210°) is **non-standard**. Confirmable by asking — they have plenty of capture-volume coverage with the 27-cam Vicon, but the markerless-camera placement is the variable.
- They have published a study on camera-configuration effects on accuracy (Applied Sciences 2025, doi 10.3390/app16041842) — worth Cameron reading before the call.

---

## 6. Draft email (copy-paste ready)

**To:** suhlrich@stanford.edu
**CC:** afalisse@stanford.edu
**Subject:** Commercial mocap collection — rear-view camera validation for sports biomech product

Scott,

I'm Cameron Van, founder of Couro (couro.io). We build single-camera markerless biomechanics for sports — pose + joint-angle estimation validated against Vicon, currently in pilot with NHL and college softball programs.

We have five camera angles validated against marker-based ground truth (front, side, and front-oblique). We have zero rear-view validation, and rear-view is the dominant broadcast and dugout-camera angle in softball. I'd like to commission a small custom collection with your team to close that gap.

Scope I have in mind:
- 1-2 subjects
- 5 synchronized cameras: 3 in your standard OpenCap front/side/oblique positions, plus **2 rear-oblique cameras at roughly 150° and 210° relative to the subject's facing direction**
- Motions: athletic drop jumps (DVJ) at minimum; softball pitching motions ideal if your capture volume and pitching surface accommodate it
- Ground truth: Vicon marker trajectories (.trc/.c3d) and OpenSim IK joint angles preferred; force plate GRFs welcome if available
- Budget: ~$2-4K
- Timeline: collection + delivery in 2-4 weeks if feasible

This is for commercial product validation, not academic dual-use — happy to sign whatever data-use or commercial-engagement agreement you typically run. CC'ing Antoine in case this is better routed through Model Health.

A few questions:
1. Is the 2 rear-camera config workable in your standard capture volume?
2. What's your typical commercial day-rate and turnaround for a 1-2 subject collection?
3. What GT deliverable formats can you provide?

Happy to jump on a 20-minute call to scope. Calendar is open most of next week.

Thanks,

Cameron Van
Founder, Couro
cam@couro.io
couro.io

---

## 7. Flags Cameron should know before sending

1. **Don't lead with "OpenCap Cloud."** That phrase isn't how they refer to it internally — the productized version is Model Health. The free app is just "OpenCap." Using "OpenCap Cloud" signals Cameron hasn't done his homework. The draft above avoids the term.
2. **Commercial vs academic framing matters.** Saying "for commercial product validation" up front is correct and respectful — it routes the conversation to the Model Health side and avoids the awkward "wait, you can't use OpenCap for commercial purposes" detour.
3. **Uhlrich and Falisse are competitors-adjacent.** Model Health does smartphone-video biomech for sports performance — overlapping market with Couro. Be aware that any technical specifics Cameron shares could be read competitively. The current draft only describes the validation gap, not Couro's pipeline. Keep it that way unless/until an NDA is in place.
4. **AUSL / UC Berkeley funding angle — leave out for now.** Mentioning a specific funder this early can either (a) make Couro look better-resourced (positive) or (b) make Model Health raise the price. Hold that card for the follow-up call.
5. **Budget is on the low end.** Stanford lab rates for industry usually start around $3-5K/day. Be prepared for a "we can do 1 subject for $3K" counter, or to drop scope to drop jumps only.
6. **2-4 week turnaround is tight.** Academic labs run on the academic calendar. If this lands during finals/summer transitions, expect 4-6 weeks minimum. The draft says "if feasible" — keep that hedge.
7. **Rear-camera config research exists.** Worth reading Uhlrich's group's own work on camera-configuration effects (Applied Sciences 2025, 10.3390/app16041842) before any call — shows Cameron is up to speed.

---

## 8. URLs to review before sending

- https://www.opencap.ai/ — landing page
- https://app.opencap.ai/ — the free web app (academic-use license terms here)
- https://www.modelhealth.io/ — Model Health (the commercial entity)
- https://www.modelhealth.io/resources/model-health-pre-seed — Model Health $1M raise announcement
- https://nmbl.stanford.edu/people/scott-uhlrich/ — Uhlrich profile
- https://nmbl.stanford.edu/human-performance-laboratory/ — Stanford lab page
- https://mobilize.stanford.edu/software/opencap/ — Mobilize Center / OpenCap overview
- https://nmbl.stanford.edu/wp-content/uploads/OpenCapManuscript_higherRes.pdf — original Uhlrich et al. 2023 PLoS Comp Bio paper
- https://doi.org/10.3390/app16041842 — camera-configuration accuracy paper from the same group
- https://github.com/stanfordnmbl/opencap-core — main processing pipeline repo (LICENSE.md confirms academic-only for the free app)

---

## 9. If Uhlrich/Falisse decline or refer elsewhere

Plausible alternative paths for rear-view-inclusive Vicon collections:

- **UC Berkeley Human Performance Lab** — Cameron is already in conversation with Berkeley re: AUSL; could fold this into that relationship.
- **University of Delaware Department of Kinesiology and Applied Physiology** — published rear-view markerless work, open to industry collaborations.
- **Auburn University Sports Medicine / Wendi Weimar's lab** — softball-specific biomechanics, friendly to industry pilots.
- **Driveline Baseball R&D** — commercial, has Vicon + multi-camera markerless, set up for pitching, fast-turn — but expensive and conflicted (they have their own analytics product).

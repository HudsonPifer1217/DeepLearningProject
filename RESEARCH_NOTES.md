# Research Notes: Lead I or Twelve?

*Measuring how much heart-disease signal survives a wearable-style single-lead ECG.*
Compiled 2026-10-22 for the project check-in. Numbers were checked against the source named next to them. Anything marked **(unverified)** could not be confirmed from a primary source during this session.

---

## 1. Talking points for the check-in

1. **The PTB-XL 12-lead deep-learning benchmark is about 0.93 macro AUROC on the 5 superclasses.** In Strodthoff et al. (IEEE JBHI), resnet1d_wang scored 0.930, xresnet1d101 0.928, inception1d 0.921 and an ensemble 0.934. A wavelet-feature baseline scored 0.874. [Strodthoff 2021]
2. **Our task is not the same as theirs, so the numbers aren't directly comparable.** Their benchmark averages AUROC over the 5 multi-label superclasses. Ours is binary normal vs abnormal. Our 12-lead gradient-boosted trees (GBT) result of 0.927 is in the same range, but it is not a like-for-like comparison.
3. **The closest prior study is Saglietto et al. 2024.** They used PTB-XL, 20 diagnostic classes and a small 1D CNN. Average AUC was 93.2% with 12 leads and 85.1% with Lead I alone, an 8.1-point drop. Adding lead II recovered most of it (90.6%). Our Lead I drop (0.927 → 0.845, 8.2 points) closely matches theirs. [Saglietto 2024]
4. **The Lead I losses in that paper concentrate in specific diagnoses, not whole superclasses.** The biggest losses were inferior MI (95.2 → 71.6), incomplete RBBB (96.2 → 73.9), inferior ischemia (93.9 → 72.0) and left fascicular block (98.2 → 79.0). Complete LBBB, complete RBBB and lateral MI lost almost nothing. [Saglietto 2024]
5. **This explains why CD dropped more than we expected.** PTB-XL's "CD" is not just a timing problem. It includes incomplete RBBB, which is defined by an rSR′ pattern in V1, and left anterior fascicular block, which is defined by the frontal axis and needs II/III/aVF. Lead I alone cannot see either. [LITFL; Saglietto 2024]
6. **Our missed inferior MI is the textbook failure case.** Inferior MI shows in II, III and aVF. Lead I is at most a reciprocal lead. Saglietto's largest single-lead drop was exactly IMI. [LITFL inferior STEMI; Saglietto 2024]
7. **HYP losing less than expected is plausible.** ECG voltage criteria for LVH are insensitive even with 12 leads (low ceiling). LVH from Lead I alone still reached an AUC of about 88% in Saglietto. In a binary task, HYP records are often caught through co-occurring abnormalities. [LITFL LVH; Saglietto 2024]
8. **The 2021 PhysioNet/CinC Challenge ("Will Two Do?") found that the lead set mattered less than the data source.** Across 12, 6, 4, 3 and 2 leads, the mean change in score was about 2.6%. Performance swung far more between hospitals. But its smallest set was 2 leads (I + II), never Lead I alone. [Reyna et al. 2022]
9. **The Apple Watch ECG is FDA-authorized only for AF vs sinus rhythm.** It is described as "similar to a Lead I ECG", and the FDA summary says "It cannot detect heart attacks". About 1 in 8 readings were inconclusive. [FDA DEN180044]
10. **Single-lead AI can still detect some structural disease.** Mayo's Apple Watch study detected EF ≤ 40% with an AUC of 0.885 (16 positives among 421 patients). The authors retrained on Lead I extracted from 12-lead ECGs and band-passed it to 0.2–25 Hz to mimic the watch. [Attia 2022, Nat Med]
11. **CNNs and ResNets still dominate ECG work.** About 90% of the official CinC 2021 entries were deep learning, mostly CNN/ResNet. Strodthoff found CNNs beat feature-based methods by a large margin. Foundation models now match but do not clearly beat them on 12-lead PTB-XL. [Reyna 2022; Strodthoff 2021; Coppola 2026]
12. **Electrode-motion noise is the hardest type.** The MIT-BIH Noise Stress Test DB provides baseline wander, muscle and electrode-motion noise at 24 to −6 dB SNR. The database page calls electrode motion "generally considered the most troublesome", which makes it the right stress test for dry-electrode wearables. [PhysioNet NSTDB]

---

## 2. How our project fits

### What we reproduce
- **12-lead performance in the expected range.** Our GBT on hand-crafted features scored 0.927 (validation, binary). Deep models in Strodthoff score 0.92–0.93 on the multi-label superdiagnostic task. These are different tasks (see Talking point 2), so the useful check is that the 1D ResNet beats our GBT baseline at 12 leads. Strodthoff found feature-based models clearly behind CNNs (0.874 for Wavelet+NN).
- **Size of the 12 → Lead I drop.** Ours is 0.927 → 0.845. Saglietto's is 93.2 → 85.1 (average over 20 classes, CNN). These are very close.
- **Limb leads recover most of the loss.** Our 6-limb-lead result is 0.904. Saglietto found that Lead I + lead II recovers to 90.6, which fits ours. Because of Einthoven's law (I + III = II) and the augmented leads, the 6 limb leads hold roughly 2 independent signals. [Reyna 2022]

### What is new or different
- **A per-superclass loss analysis at a clinically meaningful operating point** (recall at 95% specificity), with bootstrap CIs. Saglietto reports per-class AUCs but not a fixed-specificity analysis or a superclass-level hypothesis test.
- **Three configurations, including limb-only.** This separates "lost because no chest leads" from "lost because only one frontal-plane lead". This is the key design strength. MI/HYP losses should mostly appear at A → B (chest leads gone). Axis-dependent CD (LAFB) and inferior MI losses should mostly appear at B → C.
- **Noise-robustness on Lead I with NSTDB noise.** Not in Saglietto or Strodthoff.
- **Hand-crafted features vs CNN on the same lead sets.** Tests whether a CNN extracts more from a single lead than clinical features do.

### Are our early results consistent with the literature?
- **12 → Lead I drop: yes.** It matches Saglietto closely.
- **MI dropping most (≈22 points): yes.** It is driven by inferior MI (and anterior MI). Lateral MI should survive. Check by MI subtype (IMI vs AMI/ASMI vs LMI) using the PTB-XL `scp_codes`.
- **CD dropping more than predicted: consistent with the literature, and the hypothesis needs refining.** In Saglietto, complete BBBs barely changed, but IRBBB (−22 points) and LAFB/LPFB (−19 points) collapsed. These are V1-morphology and frontal-axis findings, not pure timing. Prediction: in the limb-lead config (B), LAFB should mostly recover (D1 + D2 reached 97.6 in Saglietto), and IRBBB should not.
- **Feature artefact to check.** QRS width measured on a single lead can underestimate global QRS duration, because part of the QRS can be isoelectric in that lead **(unverified as a quantified effect, but a standard measurement caveat)**. This would hurt the hand-crafted CD features specifically.
- **HYP dropping less than predicted: plausible, for three reasons.**
  1. Lead I still carries LVH information (88% AUC in Saglietto, vs 94% with 12 leads).
  2. Even 12-lead voltage criteria are insensitive, so there is less to lose.
  3. In a binary task, HYP records are often flagged via co-labels (for example ST/T "strain").
  HYP is also the smallest class (2,649 records in all of PTB-XL, about 1/10 of that in one fold), so a 12-point recall drop at 95% specificity likely has wide CIs.
- **Label-policy warning.** Some PTB-XL records carry both NORM and an abnormal code. Strodthoff showed that IVCD co-labelled with NORM was much harder (AUC 0.59 vs 0.91). How we define "abnormal" will affect CD numbers. State the rule explicitly.

---

## 3. Topic sections

### 3.1 PTB-XL and its benchmarks

**Dataset facts (PhysioNet v1.0.3)**
- Size: 21,799 10-second 12-lead ECGs from 18,869 patients.
- Sampling: 500 Hz and a downsampled 100 Hz version.
- Superclass record counts (multi-label, so they overlap): NORM 9,514; MI 5,469; STTC 5,235; CD 4,898; HYP 2,649.
- Recommended split (`strat_fold`): folds 1–8 train, 9 validation, 10 test. Patients don't cross folds. Folds 9 and 10 were checked by at least one human cardiologist, so their labels are higher quality.
- Source: https://physionet.org/content/ptb-xl/1.0.3/

**Strodthoff et al. (IEEE JBHI), superdiagnostic macro AUROC, 12 leads, 100 Hz, test fold 10 (Table II):**

| Model | Macro AUROC |
|---|---|
| resnet1d_wang | 0.930 ± 0.006 |
| xresnet1d101 | 0.928 ± 0.006 |
| lstm | 0.927 ± 0.005 |
| fcn_wang | 0.925 ± 0.006 |
| inception1d | 0.921 ± 0.006 |
| lstm_bidir | 0.921 ± 0.006 |
| Wavelet + NN (feature baseline) | 0.874 ± 0.008 |
| Ensemble | 0.934 ± 0.005 |

The ± is the maximum deviation from the 95% bootstrap interval.

**Other useful findings from Strodthoff**
- Moving from 100 Hz to 500 Hz gave "no significant gain" for diagnostic tasks with time-domain CNNs. This justifies our 100 Hz choice.
- Hidden stratification: IVCD has AUC 0.74 overall, but 0.59 when it is co-labelled with NORM and 0.91 otherwise. Non-specific labels (NST, IVCD, LAO/LAE) are weak spots.
- The paper does not analyze lead subsets.

**PTB-XL with reduced leads**
- **Saglietto et al. 2024** (Front. Cardiovasc. Med.) is the most directly relevant. See 3.2.
- **LAEF (Coppola et al., arXiv Aug 2026 preprint)** benchmarks ECG foundation models on PTB-XL Super with randomly chosen leads. 12 leads: ECGFounder 93.5, LAEF 92.3. One random lead: most foundation models drop to 50–62, LAEF 72.3. With fixed Lead I, LAEF scores 77.8.
  - Lesson: models trained on 12 leads and fed a zero-padded single lead fail. You must train on the configuration you test (we do).
  - This is a preprint and was not peer reviewed at time of writing.
- **González-Cabeza et al. 2025** (3 leads vs 12 leads, PTB-XL 5 classes) reports micro-F1 77% vs 78% **(unverified, from a search snippet only)**.

### 3.2 Reduced-lead and single-lead ECG classification

**PhysioNet/CinC Challenge 2021, "Will Two Do? Varying Dimensions in Electrocardiography"**
- **Lead sets:** 12; 6 (I, II, III, aVR, aVL, aVF); 4 (I, II, III, V2); 3 (I, II, V2); 2 (I, II).
  - Training data was 12-lead only. Hidden validation and test data were the same recordings with leads removed.
- **Data and task:** 131,149 recordings from 10 international sources (88,253 public). There were 30 scored conditions, many of them rhythm-type. PTB-XL was part of the training data.
- **Rationale (from the paper):** Leads III, aVR, aVL and aVF are mathematically redundant with I and II. The authors note that "precordial leads 'image' the ventricles far better than the limb leads". They swapped in V2 for the septum and anterior wall.
- **Main result (Reyna et al. 2022, Physiol. Meas.):** "The performance differences across the different lead combinations were smaller than the performance differences across the different test databases." The mean change across lead sets was about 2.6% of the Challenge metric. A voting model of the top 10 entries scored 0.60 on the 2-lead test set vs 0.58 for the best single algorithm.
- **Winner:** ISIBrno-AIMT ranked 1st in every lead category (official results page).
- **Caveats for us:**
  - The metric is a custom weighted score that gives partial credit for clinically similar mistakes, not AUROC.
  - The class mix is rhythm-heavy, so it favours lead sets that keep rhythm information.
  - No single-lead configuration was tested.
  - So "Will Two Do?" says little directly about Lead I alone for MI/HYP.
- Sources: https://moody-challenge.physionet.org/2021/ ; https://moody-challenge.physionet.org/2021/results/ ; https://physionet.org/files/challenge-2021/1.0.3/papers/2021ChallengePaperPMEA.pdf

**Saglietto et al. 2024**, "CNN-enabled ECG analysis: a comparison between standard twelve-lead and single-lead setups" (Front. Cardiovasc. Med.)

Setup:
- PTB-XL, 21,008 ECGs after filtering, 20 diagnostic classes.
- Folds 1–8 train, 9 validation, 10 test.
- Lightweight 1D CNN (about 36k parameters for 1 lead, 96k for 12 leads), 100 Hz input, 3.4-s windows.

| Class | 12-lead AUC | Lead I (D1) | Best D1 + one lead |
|---|---|---|---|
| NORM | 96.1 | 89.8 | D1+D2 94.0 |
| IMI (inferior MI) | 95.2 | **71.6** | D1+D2 93.3 |
| AMI (anterior MI) | 96.7 | **83.2** | D1+V2 94.9 |
| LMI (lateral MI) | 99.1 | 98.8 | D1+D2 99.6 |
| ISCI (inferior ischemia) | 93.9 | **72.0** | D1+D2 91.7 |
| ISCA (anterior ischemia) | 92.0 | 87.9 | D1+V4 89.7 |
| STTC | 90.6 | 80.7 | D1+V5 88.3 |
| CLBBB | 99.5 | 99.6 | D1+V2 99.8 |
| CRBBB | 99.6 | 98.9 | D1+V1 99.7 |
| IRBBB | 96.2 | **73.9** | D1+V1 96.2 |
| LAFB/LPFB | 98.2 | **79.0** | D1+D2 97.6 |
| IVCD | 76.4 | 68.4 | D1+D2 73.9 |
| AV block | 93.7 | 92.5 | D1+D2 95.2 |
| WPW | 93.4 | 86.6 | D1+V1 90.7 |
| LVH | 94.2 | 88.0 | D1+V6 94.1 |
| LAO/LAE | 82.4 | 80.6 | D1+D2 86.9 |
| RAO/RAE | 92.6 | 84.9 | D1+D2 97.1 |
| **Average (20 classes)** | **93.2** | **85.1** | D1+D2 90.6 |

The table is a subset of the 20 classes.

- **Takeaway:** The best second lead depends on the diagnosis. II for inferior and axis findings, V1 for IRBBB, V2 for anterior MI, V5/V6 for LVH and lateral ST. This strongly matches our clinical reasoning.
- Source: https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2024.1327179/full

**Other relevant work**
- **Lead I MI detection on PTB-XL** (medRxiv 2024, "Detection of Acute Myocardial Infarction Using Deep Learning on Lead-I ECG Data"). Found, but the full text was blocked (HTTP 403). Results are **unverified**.
- **Bergquist et al., CinC 2023** (Utah): single-lead networks predicting low LVEF scored above 90% average AUC for leads I, V5 and V6 **(from a search snippet; unverified in full text)**.

### 3.3 Clinical background: which leads show which conditions

**Lead geometry in plain language**
- The **6 limb leads** (I, II, III, aVR, aVL, aVF) look at the heart in the *frontal plane*: up/down and left/right, like looking at the person face-on.
  - Lead I looks from the left side (left arm vs right arm).
  - II, III and aVF look from below (the "inferior" leads).
  - Only 2 of the 6 are independent (I + III = II, Einthoven's law).
- The **6 chest (precordial) leads** V1–V6 look in the *horizontal plane*: front-to-back, wrapping from the right of the sternum (V1) around to the left side (V6).
  - V1–V2 sit over the septum and right ventricle.
  - V3–V4 sit over the front wall.
  - V5–V6 sit over the left lateral wall.

**MI territories**

| Territory | Leads | Typical artery | Lead I? |
|---|---|---|---|
| Inferior | II, III, aVF (reciprocal changes in aVL; sometimes I) | RCA (or LCx) | No (at best an indirect reciprocal sign) |
| Septal | V1–V2 | LAD | No |
| Anterior | V3–V4 (often V1–V4) | LAD | No |
| Lateral | I, aVL, V5–V6 | LCx / diagonal | **Yes** |

- From LITFL's inferior STEMI page: aVL is "the only lead truly reciprocal to the inferior wall". Lead I ST depression hints at an RCA culprit.
- In plain terms, an MI leaves a "scar signature" (Q waves, ST/T changes) only in the leads that look at the damaged wall. Lead I looks at the lateral wall, so it sees lateral MIs and misses inferior and anterior ones. This is exactly our missed example.

**LVH voltage criteria (LITFL)**
- **Sokolow-Lyon:** S in V1 + tallest R in V5 or V6 ≥ 35 mm. This uses chest leads only.
- **Cornell:** R in aVL + S in V3 > 28 mm (men), > 20 mm (women). This uses one limb lead plus one chest lead.
  - Cornell product: (RaVL + SV3, + 8 mm if female) × QRS duration > 2440 mm·ms.
- LITFL says voltage criteria have "high specificity but low sensitivity".
- Limb-lead-only criteria exist (for example R in aVL > 11 mm; R in I + S in III > 25 mm, the Gubner criterion) **(unverified in this session; LITFL page did not list them)**.
- In plain terms: a thick heart muscle makes bigger electrical signals, which are best measured by chest leads close to the left ventricle. Lead I sees some of the voltage and the lateral "strain" ST/T pattern, but not the main criteria.

**Bundle branch blocks and fascicular blocks (LITFL)**
- **RBBB:** QRS > 120 ms, an rSR′ "M-shaped" pattern in V1–V3, and a wide, slurred S wave in the lateral leads **I**, aVL, V5–V6. The QRS width and slurred S in Lead I are visible, which explains why complete RBBB survives single-lead (98.9 AUC).
- **Incomplete RBBB:** an rSR′ pattern in V1–V3 with QRS < 120 ms. This is defined almost entirely by V1, so Lead I cannot see it (73.9 AUC). LITFL notes it is often a normal variant.
- **LBBB:** QRS ≥ 120 ms, a dominant S in V1, and a broad monophasic R with no Q in **I**, aVL and V6. Lead I shows it directly (99.6 AUC).
- **LAFB:** left axis deviation (−45° to −90°), qR in I/aVL, rS in II/III/aVF. Axis needs at least 2 frontal leads, so Lead I alone cannot set the axis. Lead I + II restores it (97.6).
- **AV block (1st degree):** PR > 200 ms. This is a timing feature visible in any lead with a clear P wave. It survives (92.5).
- In plain terms: a "block" means the electrical signal takes a detour through the ventricles. Complete blocks make the QRS wide and oddly shaped everywhere, so any lead notices. Incomplete blocks and fascicular blocks are subtler and show up as a specific shape in V1 or as a change in the heart's electrical direction (axis). One lead can't measure either.

**ST/T changes (STTC)**
- ST depression or elevation and T-wave inversion reflect ischemia, strain or electrolyte and drug effects.
- They are territorial like MI. Inferior ischemia is invisible in I (72.0). Lateral and "non-specific" changes are partly visible.
- Non-specific ST changes (NST) are a weak class even with 12 leads (88.2).

### 3.4 Wearable / smartwatch ECG evidence

**Apple Watch ECG app (FDA De Novo DEN180044, 2018)**
- Indication: "a single channel electrocardiogram (ECG) similar to a Lead I ECG". It "determines the presence of atrial fibrillation (AFib) or sinus rhythm on a classifiable waveform."
- Over the counter. Not for users under 22.
- Limitations: "not intended to detect any other type of arrhythmia. It cannot detect heart attacks."
- Clinical study: 602 subjects. The sensitivity and specificity goals were 90% and 92%.
  - The summary reports false-positive and false-negative rates of 0.4% and 1.7% when a classification was given. That implies roughly 99.6% specificity and 98.3% sensitivity on classifiable strips.
  - "approximately 1 in 8 readings were inconclusive".
- Rates > 120 or < 50 bpm are "Unclassified".
- Source: https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180044.pdf

**Apple Heart Study (Perez et al., NEJM 2019)**
- 419,297 participants. 0.52% received an irregular-pulse notification.
- 34% of notified participants who wore a follow-up ECG patch had AF on it.
- Positive predictive value of a notification for concurrent AF: 0.84.
- **Important:** this study used the optical (PPG) pulse sensor, not the ECG electrodes. Don't cite it as single-lead ECG evidence.

**KardiaMobile (AliveCor)**
- The single-lead device received FDA clearance for AF, bradycardia (40–50 bpm) and tachycardia (100–140 bpm) determinations in 2019 (press coverage).
- KardiaMobile 6L was the first FDA-cleared 6-lead personal ECG (2019). It produces the 6 limb leads, which is exactly our configuration B.
- Additional determinations (PVCs, wide QRS, SVE) are offered through a subscription (AliveCor store page).

**AI for structural disease from a single lead**
- **Attia et al., Nat Med 2022 (Mayo, Apple Watch):**
  - 2,454 patients sent 125,610 watch ECGs.
  - 421 had a watch ECG within 30 days of an echo; 16 (3.8%) had EF ≤ 40%.
  - AUC 0.885 (95% CI 0.823–0.946) using the mean prediction over 30 days.
  - At threshold 0.6, sensitivity 87.5% and specificity 80.7%.
  - Method: retrained on **Lead I extracted from 12-lead ECGs** ("most biologically similar to the watch tracing"), band-passed 0.2–25 Hz, median beat. Automatic selection of clean sinus strips improved AUC (0.88 vs 0.85).
  - This is a direct precedent for our "train on 12-lead-derived Lead I" design.
- **Attia et al., Eur Heart J Digit Health 2022 (Eko ECG stethoscope):** single-lead low-EF detection, 100 patients, AUC 0.89 using the best position. The best single position was V2 (supine), not Lead I.
- **Spaccarotella et al., JAMA Cardiol 2020:** an Apple Watch moved to 9 body positions to build a multi-lead ECG for ST changes. Reported STEMI sensitivity 93% and specificity 95% **(from news coverage; unverified against the paper)**. The point is that MI detection required moving the watch to chest positions; Lead I alone was not enough.

**What a single lead can and can't do today**

| Status | Findings |
|---|---|
| Can (cleared or well-established) | AF vs sinus rhythm, heart rate (brady/tachy), basic rhythm and some ectopy |
| Can (research) | Low EF screening (AUC about 0.88), complete BBB / wide QRS, lateral changes, some LVH signal |
| Can't (reliably) | Inferior and anterior MI, axis-based diagnoses (fascicular blocks), V1-defined patterns (incomplete RBBB, Brugada), territorial ischemia |

### 3.5 Deep learning for ECG

- **Hannun et al., Nat Med 2019 (Stanford / iRhythm):** a 34-layer 1D CNN on 91,232 single-lead ambulatory (Zio patch) ECGs from 53,549 patients, 12 rhythm classes. Average AUC 0.97. F1 was comparable to the average cardiologist.
  - This shows that single lead is enough for *rhythm*. It says nothing about morphology diagnoses like MI.
- **Ribeiro et al., Nat Commun 2020 (Brazil, CODE dataset):** a ResNet-style 1D CNN trained on more than 2 million 12-lead exams, 6 abnormalities. F1 > 80% and specificity > 99%. It outperformed cardiology residents.
- **Why 1D CNNs and ResNets dominate:**
  - ECGs are 1D signals with local, translation-invariant patterns (QRS, P, T) repeating each beat, which is a natural fit for convolutions.
  - Residual connections allow deep networks.
  - They work well with about 10^4 training records (PTB-XL size).
  - Strodthoff: "convolutional neural networks, in particular resnet- and inception-based architectures, show the strongest performance across all tasks".
  - CinC 2021: 35 of 39 official entries (about 90%) were deep learning, often ResNets, and about 30% of those also used hand-crafted features.
- **Transformers, state-space models and foundation models:**
  - Mehari & Strodthoff (arXiv 2023) report that structured state-space models (S4) beat CNN and LSTM baselines on PTB-XL with statistical significance. The exact numbers were not extracted **(unverified)**.
  - Large self-supervised models: HuBERT-ECG (9.1M ECGs), ECGFounder (about 10M ECGs, 150 conditions), ECG-JEPA, ST-MEM, MERL.
  - On PTB-XL Super at 12 leads they score about 0.90–0.935 macro AUROC (LAEF benchmark). That is roughly equal to the 2020 CNN ensemble (0.934).
  - **Conclusion:** they are competitive but not dramatically better on PTB-XL. Their main promise is label efficiency and transfer.

### 3.6 Noise in wearable ECG

**MIT-BIH Noise Stress Test Database (Moody, Muldrow & Mark 1984)**
- Three noise records: **bw** (baseline wander), **ma** (muscle/EMG) and **em** (electrode motion). They were recorded from active volunteers with electrodes on the limbs, placed where the volunteers' own ECG wasn't visible.
- The standard test records are MIT-BIH 118 and 119 mixed with em noise at SNRs of 24, 18, 12, 6, 0 and −6 dB. Noise starts after 5 minutes and alternates 2 minutes on, 2 minutes off.
- "Electrode motion artifact is generally considered the most troublesome", because it mimics ectopic beats and overlaps the ECG spectrum, so simple filters can't remove it.
- Source: https://physionet.org/content/nstdb/1.0.0/

**Typical SNR choices in the literature**
- Classification studies often sweep from about 15 dB down to 0 dB and avoid negative SNR because waveforms become unrecognizable.
- Denoising papers test −6, 0 and 6 dB.
- Our options:
  - Use the NSTDB's own grid (24/18/12/6/0 dB, optionally −6).
  - Report each noise type separately.
  - Define SNR as signal power over noise power on the Lead I segment.

**Effect on models**
- Venton et al. (arXiv 2021): CNNs trained on clean ECGs lost up to 0.18 F1 on noisy data. Training on noisy data kept the drop below 0.05 and cost little on clean data.
- This suggests a follow-up: noise augmentation during training.

**Dry-electrode wearables**
- Joutsen et al., Sci Rep 2024: movement "reduced the SNR markedly" for dry electrodes. Lower skin-electrode impedance gives higher SNR.
- Attia 2022 also notes that watch ECGs have more artifact and baseline wander and lose high-frequency content (for example pacing spikes).

---

## 4. Clinical cheat sheet

D1 (Lead I) AUCs are from Saglietto 2024, PTB-XL, 12-lead → Lead I.

| Condition (PTB-XL superclass) | Leads that show it | Visible from Lead I? | Lead I AUC (12 → I) |
|---|---|---|---|
| Atrial fibrillation / rhythm | Any lead (irregular RR, no P) | **Yes** | n/a (Apple/Kardia cleared) |
| Heart rate, brady/tachy | Any lead | **Yes** | n/a |
| 1st-degree AV block (CD) | Any lead with clear P | **Yes** | 93.7 → 92.5 |
| Complete LBBB (CD) | V1, V6, **I**, aVL | **Yes** | 99.5 → 99.6 |
| Complete RBBB (CD) | V1–V3 (rSR′), **I**, aVL, V5–V6 (slurred S) | **Yes** (mostly) | 99.6 → 98.9 |
| Incomplete RBBB (CD) | V1–V3 | **No** | 96.2 → 73.9 |
| LAFB/LPFB (CD) | Axis: I + II/III/aVF | **Partly** (needs ≥2 limb leads) | 98.2 → 79.0 |
| IVCD, non-specific (CD) | Any (wide QRS) | Partly | 76.4 → 68.4 |
| WPW (CD) | Short PR + delta wave, many leads | Partly | 93.4 → 86.6 |
| Inferior MI (MI) | II, III, aVF | **No** | 95.2 → 71.6 |
| Anterior/septal MI (MI) | V1–V4 | **No** | 96.7 → 83.2 (AMI) |
| Lateral MI (MI) | **I**, aVL, V5–V6 | **Yes** | 99.1 → 98.8 |
| LVH (HYP) | V1, V5–V6 (Sokolow), aVL + V3 (Cornell); strain in lateral leads | Partly | 94.2 → 88.0 |
| Right atrial enlargement (HYP) | II (tall P), V1 | Partly / no | 92.6 → 84.9 |
| Left atrial enlargement (HYP) | II (wide P), V1 | Partly | 82.4 → 80.6 |
| Inferior ischemia (STTC) | II, III, aVF | **No** | 93.9 → 72.0 |
| Anterior ischemia (STTC) | V1–V4 | Partly | 92.0 → 87.9 |
| Non-specific ST/T (STTC) | Varies | Partly | 88.2 → 84.9 |
| Low ejection fraction (not a PTB-XL label) | Global | **Partly** (AI, AUC about 0.88) | Attia 2022 |

---

## 5. Glossary

- **ECG (electrocardiogram):** a recording of the heart's electrical activity over time.
- **Electrode vs lead:** an electrode is a sticker on the skin. A *lead* is a voltage difference between electrodes, so it is a "viewpoint" on the heart. 10 electrodes give 12 leads.
- **Limb leads (I, II, III, aVR, aVL, aVF):** viewpoints in the frontal (face-on) plane, from arm and leg electrodes. Only 2 are independent.
- **Lead I:** right arm to left arm, looking from the left side. A smartwatch touched by the opposite finger records roughly this.
- **Augmented leads (aVR, aVL, aVF):** limb-lead views computed against the average of the other limbs.
- **Precordial / chest leads (V1–V6):** viewpoints in the horizontal plane, from electrodes across the chest. They are close to the heart and best for the ventricles.
- **P wave:** atrial (upper chamber) activation.
- **PR interval:** time from atrial to ventricular activation. Long PR means AV block.
- **QRS complex:** ventricular activation. A wide QRS (≥ 120 ms) means slow or abnormal conduction, as in bundle branch block.
- **ST segment:** the flat part after the QRS. Elevation or depression suggests ischemia or injury.
- **T wave:** ventricular recovery (repolarization). Inversion can mean ischemia or strain.
- **Axis:** the average direction of electrical activation in the frontal plane. It needs at least 2 limb leads to compute.
- **MI (myocardial infarction):** heart attack, meaning death of heart muscle from blocked blood flow.
- **STTC:** ST/T change, PTB-XL's superclass for ischemia-type and non-specific repolarization changes.
- **CD (conduction disturbance):** abnormal electrical wiring or timing (blocks, WPW).
- **RBBB / LBBB:** right or left bundle branch block, where one of the two main "wires" into the ventricles is blocked, so that side activates late. "Incomplete" means a partial, milder version.
- **LAFB:** left anterior fascicular block, a block in one branch of the left bundle. It shows as a left-shifted axis.
- **HYP / LVH:** hypertrophy, a thickened heart wall. LVH is the left-ventricle version. ECG voltage criteria are specific but insensitive.
- **AF (atrial fibrillation):** chaotic atrial rhythm. It is irregularly irregular with no P waves, and it is the main smartwatch use case.
- **Ejection fraction (EF):** the percentage of blood the left ventricle pumps out per beat. EF ≤ 40% is weak pumping.
- **AUROC:** the probability that the model ranks a random abnormal ECG above a random normal one. 0.5 is chance and 1.0 is perfect. It is threshold-free.
- **AUPRC:** area under the precision-recall curve. It is more informative when positives are rare. Its baseline equals the prevalence.
- **Sensitivity (recall):** the share of truly abnormal records that the model flags.
- **Specificity:** the share of truly normal records that the model correctly clears.
- **Sensitivity at 95% specificity:** fix the threshold so only 5% of normal ECGs are falsely flagged, then see how many abnormal ones are caught.
- **SNR (signal-to-noise ratio, dB):** 10·log10(signal power / noise power). 0 dB means equal power. Negative values mean noise dominates.
- **Bootstrap CI:** resample the test set with replacement many times and recompute the metric. The spread gives a confidence interval.
- **Hidden stratification:** a label that contains easy and hard subgroups, so the average score hides the poor subgroup.

---

## 6. Likely questions and good answers

**Q1. Why not a transformer or foundation model?**
- On PTB-XL-sized data (about 17k training records), 1D ResNets are the established strong baseline (0.93 superdiagnostic AUROC). About 90% of CinC 2021 winners and entries were CNN-based.
- Foundation models pretrained on millions of ECGs reach about 0.92–0.935 on PTB-XL Super, so they are not clearly better.
- The project's question is *how much signal is in each lead set*, not which architecture is best. A well-understood CNN keeps the comparison clean.
- A foundation-model comparison would be a natural extension. Note LAEF's finding that 12-lead-trained models collapse on single-lead input unless trained for it.

**Q2. Is Lead I in PTB-XL the same as an Apple Watch?**
- Not exactly. The FDA describes the watch as "similar to a Lead I ECG".
- Differences:
  - Dry electrodes on wrist and fingertip vs clinical gel electrodes.
  - The patient is sitting or standing vs supine.
  - Watch recordings are 30 s vs 10 s.
  - Watch signals are more noisy and narrower-bandwidth. Mayo band-passed Lead I to 0.2–25 Hz to mimic the watch.
- So our Lead I result is an *upper bound* on watch performance. That is why we add NSTDB noise, and a 0.5–25 Hz band-pass would be a cheap extra realism step.

**Q3. Why 95% specificity?**
- A consumer screening device runs on mostly healthy people, so even a small false-positive rate creates many false alarms and unnecessary follow-ups.
- The Apple ECG app's false-positive rate was 0.4%.
- Fixing specificity gives a single, clinically meaningful operating point that is comparable across configurations. AUROC alone can hide differences at the high-specificity end.
- We can also report 90% and 99% as a sensitivity analysis.

**Q4. Why did CD drop more than you predicted?**
- PTB-XL's CD superclass mixes timing findings (AV block, complete BBB, which survive Lead I) with V1-morphology findings (incomplete RBBB) and axis findings (LAFB/LPFB). The latter two collapse with Lead I alone: −22 and −19 AUC points in Saglietto.
- Our refined hypothesis is that *complete* blocks survive and incomplete/fascicular ones don't. We'll check by sub-label, and by whether the limb-lead set recovers LAFB.

**Q5. Your 12-lead GBT gets 0.927. Isn't that already as good as deep learning?**
- Different task (binary vs 5-class macro AUROC) and validation vs test fold, so the numbers are not comparable.
- The relevant comparison is our CNN vs our GBT on identical splits and labels.
- Strodthoff found feature-based baselines about 0.05 below CNNs on the multi-label task.

**Q6. Didn't "Will Two Do?" already show reduced leads are fine?**
- It showed 2 leads (I + II) lose little *on its metric and class mix*, which is rhythm-heavy, with partial credit.
- It never tested one lead.
- Its main lesson was that cross-hospital generalization matters more than the lead set. That is a limitation worth acknowledging for our single-dataset study.

**Q7. How do you handle multi-label records and NORM co-labels?**
- Define "abnormal" as any non-NORM superclass, and decide explicitly what to do with records that have both NORM and an abnormal code.
- Per-category recall then counts an MI record as caught if it is flagged abnormal for any reason. Say that clearly, because it inflates per-class recall for classes with many co-labels (like HYP).

**Q8. Is the per-category difference statistically meaningful?**
- HYP has about 260 records per fold, and at fixed specificity the recall estimates are noisy.
- We'll report paired bootstrap CIs on the *difference* (12-lead minus Lead I) per category, and average over 3 seeds for the CNN.

**Q9. Why 100 Hz?**
- Strodthoff found no significant gain from 500 Hz for time-domain CNNs on diagnostic tasks.
- 100 Hz is also closer to what wearable pipelines effectively keep after filtering.

**Q10. What would clinicians actually gain?**
- A quantified map of which diagnoses a watch can and can't triage.
- Evidence for "which second lead to add". Saglietto shows the best second lead depends on the diagnosis, which is relevant to multi-position watch recordings (Spaccarotella 2020) and 6-lead devices like KardiaMobile 6L (our configuration B).

---

## 7. Research log (2026-10-22)

**Searched / fetched**
1. PhysioNet PTB-XL v1.0.3 page. Confirmed counts, folds and citation.
2. Strodthoff et al. arXiv HTML. Extracted the superdiagnostic Table II. Downloaded the PDF and extracted text for the 500 Hz and IVCD hidden-stratification passages.
3. CinC 2021 challenge page, results page and the Reyna et al. 2022 Physiol. Meas. PDF (text extracted). Got the abstract conclusion, the 2.6% figure, Table 6 and the voting-model scores (0.60 vs 0.58).
4. Saglietto et al. 2024, Frontiers. Full per-class Table 1 for 12-lead vs D1 vs two-lead. **This is the single most useful source for the project.**
5. LITFL pages: LVH, RBBB, LBBB, LAFB, inferior STEMI.
6. FDA De Novo DEN180044 PDF (text extracted). Indications, limitations, false-positive/negative rates, inconclusive rate.
7. Attia et al. Nat Med 2022 (PMC9805528) and Attia et al. EHJ-DH 2022 (PMC9708035).
8. Apple Heart Study, Hannun 2019 and Ribeiro 2020 (via search summaries and abstract mirrors).
9. NSTDB PhysioNet page. Noise robustness papers (Venton 2021). Dry-electrode SNR (Joutsen 2024).
10. Foundation models: HuBERT-ECG (medRxiv), LAEF (arXiv 2608.03690) with its PTB-XL Super lead-subset table.

**Dead ends**
- nature.com article pages redirect to a login and could not be fetched directly. PMC mirrors worked for Attia.
- medRxiv full text (Lead-I AMI paper; the 2025 Strodthoff replication) returned 403.
- I could not find the CinC 2021 numeric top scores per lead set on the results page (they are in linked CSVs). Only ranks were captured.
- The S4 paper's exact PTB-XL numbers were not extracted.

**Open questions worth following up**
- Get the medRxiv Lead-I AMI paper's numbers (MI by territory from Lead I).
- Does limb-lead input (config B) recover LAFB and IMI in our data, as Saglietto's D1 + D2 results predict?
- Break our CD and MI recall down by sub-label (IRBBB, LAFB, CLBBB, CRBBB, 1AVB; IMI, AMI/ASMI, LMI).
- Check how our hand-crafted QRS width from Lead I compares with PTB-XL+ global QRS duration (PTB-XL+ provides 12SL/Uni-G features) **(PTB-XL+ availability not verified this session)**.
- Try simulating watch bandwidth (0.5–25 Hz band-pass) on Lead I.
- Try noise augmentation during training (Venton 2021).
- Verify the Spaccarotella 2020 numbers and the Gubner/aVL LVH limb-lead criteria from primary sources.

---

## 8. References

1. Wagner P, Strodthoff N, Bousseljot R-D, Kreiseler D, Lunze FI, Samek W, Schaeffter T. PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data* 7, 154 (2020). doi:10.1038/s41597-020-0495-6. Dataset: https://physionet.org/content/ptb-xl/1.0.3/
2. Strodthoff N, Wagner P, Schaeffter T, Samek W. Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL. *IEEE Journal of Biomedical and Health Informatics* 25(5):1519–1528 (2021). doi:10.1109/JBHI.2020.3022989. Volume and pages **(unverified)**; DOI verified. arXiv: https://arxiv.org/abs/2004.13701
3. Reyna MA, Sadr N, Perez Alday EA, Gu A, Shah AJ, Robichaux C, Rad AB, Elola A, Seyedi S, Ansari S, Ghanbari H, Li Q, Sharma A, Clifford GD. Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021. *Computing in Cardiology* 48 (2021). https://moody-challenge.physionet.org/2021/
4. Reyna MA et al. Issues in the automated classification of multilead ECGs using heterogeneous labels and populations. *Physiological Measurement* 43(8) (2022). PDF: https://physionet.org/files/challenge-2021/1.0.3/papers/2021ChallengePaperPMEA.pdf ; PubMed: https://pubmed.ncbi.nlm.nih.gov/35815673/
5. PhysioNet/CinC Challenge 2021 Results. https://moody-challenge.physionet.org/2021/results/
6. Saglietto A, Baccega D, Esposito R, Anselmino M, Dusi V, Fiandrotti A, De Ferrari GM. Convolutional neural network (CNN)-enabled electrocardiogram (ECG) analysis: a comparison between standard twelve-lead and single-lead setups. *Frontiers in Cardiovascular Medicine* 11:1327179 (2024). https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2024.1327179/full
7. Coppola E, Fiorini S, Liò P, Savardi M, Signoroni A. LAEF: A Lead-Agnostic ECG Foundation Model Towards Point-of-Care Diagnostics. arXiv:2608.03690 (2026, preprint). https://arxiv.org/abs/2608.03690
8. U.S. FDA. De Novo Classification Request for ECG App (DEN180044), Apple Inc. (2018). https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180044.pdf
9. Perez MV, Mahaffey KW, Hedlin H, et al. Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation. *N Engl J Med* 381:1909–1917 (2019). Open-access copy: https://escholarship.org/uc/item/5n08v244
10. Attia ZI, Harmon DM, Dugan J, et al. Prospective evaluation of smartwatch-enabled detection of left ventricular dysfunction. *Nature Medicine* 28:2497–2503 (2022). doi:10.1038/s41591-022-02053-1. https://pmc.ncbi.nlm.nih.gov/articles/PMC9805528/
11. Attia ZI, Dugan J, Rideout A, et al. Automated detection of low ejection fraction from a one-lead electrocardiogram: application of an AI algorithm to an electrocardiogram-enabled Digital Stethoscope. *European Heart Journal – Digital Health* 3(3):373–379 (2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9708035/
12. Spaccarotella CAM, Polimeni A, Migliarino S, et al. Multichannel Electrocardiograms Obtained by a Smartwatch for the Diagnosis of ST-Segment Changes. *JAMA Cardiology* 5(10):1176–1180 (2020). **(Citation details unverified; seen via news coverage.)** https://www.mobihealthnews.com/news/expert-guidance-apple-watch-could-record-multi-lead-ecgs-and-spot-st-segment-changes
13. AliveCor. FDA clearances for bradycardia/tachycardia detection (2019) and KardiaMobile 6L (2019). https://www.fiercebiotech.com/medtech/fda-clears-alivecor-s-mobile-6-lead-ecg ; https://store.kardia.com/products/kardiamobile6l
14. Hannun AY, Rajpurkar P, Haghpanahi M, Tison GH, Bourn C, Turakhia MP, Ng AY. Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. *Nature Medicine* 25:65–69 (2019). doi:10.1038/s41591-018-0268-3. https://pmc.ncbi.nlm.nih.gov/articles/PMC6784839/
15. Ribeiro AH, Ribeiro MH, Paixão GMM, et al. Automatic diagnosis of the 12-lead ECG using a deep neural network. *Nature Communications* 11:1760 (2020). doi:10.1038/s41467-020-15432-4. https://pubmed.ncbi.nlm.nih.gov/32273514/
16. Mehari T, Strodthoff N. Towards quantitative precision for ECG analysis: Leveraging state space models, self-supervision and patient metadata. arXiv:2308.15291 (2023). https://arxiv.org/abs/2308.15291
17. Coppola E et al. HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications. medRxiv 2024.11.14.24317328. https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v1 (author list **unverified**)
18. Moody GB, Muldrow WE, Mark RG. A noise stress test for arrhythmia detectors. *Computers in Cardiology* 11:381–384 (1984). Database: https://physionet.org/content/nstdb/1.0.0/
19. Venton J, Harris PM, Sundar A, Smith NAS, Aston PJ. Robustness of convolutional neural networks to physiological ECG noise. arXiv:2108.01995 (2021); related DOI 10.1098/rsta.2020.0262. https://arxiv.org/abs/2108.01995
20. Joutsen A, et al. ECG signal quality in intermittent long-term dry electrode recordings with controlled motion artifacts. *Scientific Reports* (2024). doi:10.1038/s41598-024-56595-0. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11024137/
21. LITFL ECG Library: LVH https://litfl.com/left-ventricular-hypertrophy-lvh-ecg-library/ ; RBBB https://litfl.com/right-bundle-branch-block-rbbb-ecg-library/ ; LBBB https://litfl.com/left-bundle-branch-block-lbbb-ecg-library/ ; LAFB https://litfl.com/left-anterior-fascicular-block-lafb-ecg-library/ ; Inferior STEMI https://litfl.com/inferior-stemi-ecg-library/
22. Detection of Acute Myocardial Infarction Using Deep Learning on Lead-I ECG Data. medRxiv 10.1101/2024.10.15.24315544 (2024). **(Full text not accessed; authors and results unverified.)** https://www.medrxiv.org/content/10.1101/2024.10.15.24315544.full.pdf

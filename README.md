# Solar Filament Instance Segmentation

**Physics-guided, topology-aware global-local segmentation of 2048×2048 H-alpha solar imagery**

[![PyTorch](https://img.shields.io/badge/PyTorch-Modeling-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/filament-segmentation-2026)
[![Computer Vision](https://img.shields.io/badge/Task-Instance%20Segmentation-blue)](#)

> **TL;DR** — I developed a two-stage global-local MultiRes Attention U-Net for the **Solar Filament Segmentation Challenge 2026**. The final system combines solar-domain image features, topology-aware supervision, hard-negative mining, three-fold ensembling, and native-resolution local refinement. I experimented and improved my Kaggle Panoptic Quality (PQ) from a **0.28 YOLO baseline to 0.35** with the UNet. In three-fold out-of-fold evaluation, native refinement recovered **230 additional true-positive filament matches** and reduced false negatives by **230** compared with the global-only model.

- **Competition:** [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
- **Framework:** PyTorch
- **Task:** Solar filament instance segmentation
- **Input:** 2048×2048 grayscale H-alpha solar images
- **Model:** Global-Local MultiRes Attention U-Net
- **Leaderboard metric:** Panoptic Quality (PQ)
- **Best Kaggle leaderboard PQ:** **0.35**

---
Solar filaments are dark, elongated structures visible in H-alpha observations of the Sun. Their morphology is important in space-weather research, but automatic segmentation is difficult because filaments can be:

- only a few pixels wide,
- low contrast,
- fragmented or curved,
- visually similar to other dark solar structures,
- affected by image noise and spatial brightness variation.

The competition objective is to generate an **individual segmentation mask for each filament**, preserving fine-scale morphology while minimizing non-filament detections.

This makes the problem more challenging than ordinary foreground segmentation: the model must both **find the correct filament instances** and **preserve their structure**.

---

## Key result

The final model was selected through iterative error analysis rather than architecture scaling alone.

| Model stage | Main change | Kaggle PQ |
|---|---|---:|
| YOLO baseline | Generic instance segmentation + filtering | 0.28 |
| Experiment version 4C | Physics-guided MultiRes Attention U-Net | 0.33 |
| Experiment version 4D | Topology-aware loss + hard-negative mining | 0.34 |
| Experiment version 5A | Global context + native-resolution local refinement | **0.35** |

### Three-fold OOF comparison

| Metric | V4-D global | V5-A global-local | Change |
|---|---:|---:|---:|
| PQ | 0.3833 | **0.4008** | +0.0174 |
| Detection Quality | 0.5789 | **0.5985** | +0.0196 |
| Segmentation Quality | 0.6621 | **0.6696** | +0.0075 |
| True-positive matches | 4,860 | **5,090** | **+230** |
| False positives | 3,731 | **3,721** | -10 |
| False negatives | 3,339 | **3,109** | **-230** |

The most important change was not aggressive false-positive removal. The native-resolution stage mainly **recovered filament instances that the 1024-resolution global model missed**.

> **Finding from the metric:** the dominant remaining bottleneck was loss of thin structures during downsampling, not simply model capacity or threshold calibration.

---

## Final architecture

The final system uses a **global-local design**.

flowchart TD
    A["2048 × 2048 H-alpha image"]

    A --> B["Global physics preprocessing at 1024"]
    B --> B1["Radial-normalized intensity"]
    B --> B2["Local contrast"]
    B --> B3["Multi-scale dark-ridge response"]

    B1 --> C["V4-D Global MultiRes Attention U-Net"]
    B2 --> C
    B3 --> C

    C --> D1["Global filament probability"]
    C --> D2["Global spine probability"]

    A --> E["Native physics preprocessing at 2048"]
    E --> E1["Native radial-normalized intensity"]
    E --> E2["Native local contrast"]
    E --> E3["Native dark-ridge response"]

    E1 --> F["512 × 512 overlapping native tiles"]
    E2 --> F
    E3 --> F

    D1 --> G["Upsample global priors to 2048"]
    D2 --> G

    F --> H["V5-A Local MultiRes Attention U-Net"]
    G --> H

    H --> I["Feathered reconstruction of local probability"]
    I --> J["60% local + 40% global probability"]

    D1 --> J

    J --> K["Average predictions from 3 folds"]
    K --> L["Hysteresis instance extraction"]
    L --> M["Connected-component instances"]
    M --> N["Compressed COCO RLE submission"]

### 1. Physics-guided image representation

“Physics-guided” here means using **domain-informed image features** that make known solar-image structure explicit.

The global model receives three channels:

| Channel | Purpose |
|---|---|
| **Radial-normalized intensity** | Corrects brightness variation across the solar disk |
| **Local contrast** | Highlights structures darker than their local surroundings |
| **Multi-scale dark-ridge response** | Emphasizes elongated dark structures at multiple widths |

This provides the model with useful inductive biases instead of requiring it to learn every solar-specific nuisance factor from a relatively small dataset.

### 2. MultiRes Attention U-Net

The segmentation backbone is a lightweight MultiRes Attention U-Net.

I chose it because solar filaments vary strongly in width and shape:

- **MultiRes blocks** capture structures at multiple receptive-field scales.
- **U-Net skip connections** preserve spatial detail.
- **Attention gates** suppress irrelevant high-resolution skip features.
- **Group normalization** is stable with the small batches required for high-resolution segmentation.

The global model predicts two outputs:

1. a **filament mask probability**, and
2. a **filament spine probability**.

The spine is used as an auxiliary structural signal because filament connectivity matters in addition to pixel overlap.

### 3. Topology-aware learning

Pixel-wise overlap losses can score a fragmented prediction reasonably even when a physically continuous filament has been broken into pieces.

To address this, training combines:

- binary cross-entropy,
- Dice loss,
- **clDice topology loss**, and
- auxiliary spine supervision.

clDice encourages preservation of filament centerline connectivity.

### 4. Hard-negative mining

Random background patches are often too easy.

The training pipeline deliberately samples difficult non-filament regions that resemble filaments through:

- dark normalized intensity,
- strong local contrast, or
- ridge-like structure.

This forces the network to learn the distinction between actual filaments and visually similar background structures instead of relying on simple darkness thresholds.

### 5. Native-resolution local refinement

The original images are 2048×2048. The global model operates at 1024 to keep memory and computation manageable.

That creates a problem: a filament only a few pixels wide at 2048 can become one pixel wide, fragmented, or disappear after downsampling.

Instead of training one very memory-intensive full-resolution U-Net, experiment V5-A uses a second local model on native-resolution tiles.

Each local tile receives five channels:

1. native radial-normalized intensity,
2. native local contrast,
3. native dark-ridge response,
4. upsampled global filament probability,
5. upsampled global spine probability.

This gives the local model both **fine detail** and **full-disk context**.

### 6. Overlapping tiled inference

Native inference uses:

- 512×512 tiles,
- stride 384,
- overlapping predictions,
- feathered/Hann-style blending.

Overlap reduces border artifacts because pixels near one tile edge are also seen closer to the center of neighboring tiles.

### 7. Global-local fusion and ensembling

For each fold:

```text
P_fold = 0.60 × P_local + 0.40 × P_global
```

The three fold probabilities are averaged before instance extraction.

The final calibrated post-processing uses:

```text
Low threshold:   0.40
High threshold:  0.75
Closing radius:  1
Minimum area:    480 native pixels
```

---

## Why hysteresis instead of a single threshold?

Filaments often contain:

- a high-confidence dark core,
- weak thin extensions.

A single high threshold can cut off the weak extensions.  
A single low threshold can introduce background noise.

Hysteresis keeps low-confidence pixels when they are connected to a high-confidence seed, which is better suited to long, thin structures.

Connected-component labeling then converts the semantic foreground map into non-overlapping filament instances.

---

## Experimental reasoning

This project was developed as a sequence of hypotheses.

| Question | Experiment | Outcome |
|---|---|---|
| Is a generic instance-segmentation model sufficient? | YOLO baseline | Useful starting point, but limited on thin structures |
| Would solar-specific representation help? | Physics-guided V4-C | Large improvement |
| Are fragmentation and difficult negatives important? | clDice + hard-negative V4-D | Improved robustness |
| Are remaining errors mainly bad components? | XGBoost component scorer | Did not improve final result |
| Is orientation variance the main issue? | Rotation test-time augmentation | No meaningful gain |
| Can two similar global models average away the errors? | V4-C/V4-D probability blend | OOF gain, leaderboard plateau |
| Is spatial-resolution loss the main bottleneck? | V5-A native local refinement | **Best tested result** |

One of the main lessons from the project was that **not every additional technique is useful**. The strongest improvement came from diagnosing the actual error mode and changing the architecture to address it.

---

## Evaluation

The competition uses **Panoptic Quality (PQ)**, which rewards both:

1. finding the correct filament instances, and
2. achieving good mask overlap.

Conceptually:

```text
PQ = sum(IoU of matched instances)
     ----------------------------------
     TP + 0.5 × FP + 0.5 × FN
```

This metric is useful for this task because a model can be penalized for:

- missing filaments,
- predicting extra filament instances,
- poor segmentation overlap.

### Validation strategy

The later experiments use **three-fold cross-validation grouped by physical image**.

This is important because the dataset contains multiple annotations of some underlying solar observations. Splitting annotation records independently could allow the same physical image to appear in both training and validation.

Multi-annotator masks are averaged into **soft targets**, allowing the model to represent annotation uncertainty rather than treating one annotator's boundary as absolute ground truth.

---
## Reproduce the inference pipeline

### 1. Clone the repository

```bash
git clone https://github.com/NanThawe/Solar-Filament-Segmentation.git
cd Solar-Filament-Segmentation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The main runtime dependencies are PyTorch, OpenCV, NumPy, SciPy, pandas, tqdm, and pycocotools.

### 3. Download the competition data

Join the Kaggle competition and download the test data:

[**Solar Filament Segmentation Challenge 2026**](https://www.kaggle.com/competitions/filament-segmentation-2026)

Expected test-image structure:

```text
MAGFiLO_1.0_Kaggle_2026/
└── test/
    └── test_images/
        ├── *.jpeg
        └── ...
```

### 4. Run the inference notebook

Open:

```text
solar_filament_instance_segmentation_inference.ipynb
```

Set the dataset and checkpoint paths at the top of the notebook, then run all cells.

The notebook:

1. loads all three V4-D global checkpoints,
2. loads all three V5-A local checkpoints,
3. generates global mask and spine priors,
4. performs native-resolution tiled refinement,
5. averages the three folds,
6. extracts non-overlapping connected components,
7. encodes each instance as compressed COCO RLE,
8. writes:

```text
submission.csv
```

A successful run should also verify:

- all expected test images were processed,
- prediction maps are non-zero,
- the submission contains filament instances,
- instance masks do not overlap.

---

## Model checkpoints

The final ensemble contains six PyTorch checkpoints:

```text
Global context:
v4d_fold0.pt
v4d_fold1.pt
v4d_fold2.pt

Native refinement:
v5a_local_fold0.pt
v5a_local_fold1.pt
v5a_local_fold2.pt
```

Each local fold is paired with the corresponding global fold during test inference.

---

## What I learned from this project

### 1. Error analysis matters more than adding techniques

Several plausible improvements—component scoring, rotation TTA, and blending—did not materially improve the leaderboard result.

The useful question I asked myself throughout this project was:

> “What kind of error is the current model making, and what information is missing?”

### 2. Resolution can be a representation bottleneck

The V4-D → V5-A comparison showed that the major improvement was in recovering true filament instances. This was consistent with the hypothesis that downsampling was removing thin structures.

### 3. Domain knowledge and learned features can complement each other

The physics-guided channels make known properties of H-alpha imagery easier for the model to exploit.

### 4. Reproducibility is part of ML engineering

A trained model is only useful if the entire inference path—from preprocessing and checkpoint loading to instance extraction and serialization—can be reproduced and validated.

---

## Limitations and next steps

This repository represents the best configuration tested during the competition, not a claim of theoretical optimality.

Important limitations include:

- Post-processing parameters were calibrated using the same OOF predictions used for model selection, so the OOF PQ should be interpreted as a calibrated development metric rather than an untouched test estimate.
- A stricter follow-up would generate **strictly out-of-fold global priors for every local-model training image** to remove train/validation prior-distribution mismatch.
- Performance should be analyzed explicitly by filament size and annotator agreement to verify where native-resolution refinement contributes most.
- Future work could explore lightweight feature-level global/local fusion or cross-attention after establishing a stricter validation protocol.

---

## Dataset and acknowledgements

The competition uses **MAGFiLO — Manually Annotated GONG Filaments from H-Alpha Observations**.

- Competition: [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
- MAGFiLO paper: [Scientific Data (2024)](https://doi.org/10.1038/s41597-024-03876-y)

The underlying H-alpha observations originate from the Global Oscillation Network Group (GONG) and the National Solar Observatory ecosystem. Please refer to the competition page and dataset publication for the authoritative data license, attribution, and usage conditions.

---

## Project status

This repository contains the experimentation history, final V5-A inference pipeline, and the six trained checkpoints used for the final ensemble.

The project was developed as a computer-vision research and engineering exercise in:

- domain-informed preprocessing,
- high-resolution segmentation,
- topology-aware learning,
- cross-validation,
- ensemble inference,
- error analysis,
- and reproducible model evaluation.

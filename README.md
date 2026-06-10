# From Tokens to Faces: Investigating Discrete Speech Representations for 3D Facial Animation

### Official Github repository for the work accepted at the 2026 Interspeech (Sydney, Australia, from Sep 27th to Oct 1st)

#### Authors: _Pedro Corrêa, Olivier Perrotin, Samir Sadok, Paula Costa, Thomas Hueber_

<img width="1332" height="652" alt="pipeline_interspeech_2026" src="https://github.com/user-attachments/assets/a7bac5fe-5538-4441-bb73-b6f944e8c675" />

#### Abstract

_The choice of speech representation is critical in speech-driven 3D facial animation. Representations differ in what they encode: SSL features emphasize segmental and semantic cues, neural codecs yield latents optimized for acoustic reconstruction, and ASR-style objectives produce label-based spaces. We evaluate four speech representation families for 3D facial synthesis, comparing their facial reconstruction quality across two facial decoders using objective metrics and a perceptual evaluation. We additionally conduct probing analyses that relate tokenized representations to phonetic units and to articulatory deformations. We found that encoding phonetic classes is beneficial for accurate facial animation prediction on both semantic and label-based representations with comparable facial animation quality. From the latter, we introduce an Audio Visual Text-to-Speech (AVTTS) pipeline that leverages, as a shared space, discrete representations to decode speech and 3D facial motion._

#### General Instructions

The part of this repository dedicated to reproducing the results from the research paper is divided into "comparison" and "probing". The folders "data" and "pretrained_models" are suggestions for places to insert data from the BEAT 2 dataset and model weights for each evaluated speech encoder, respectively.

#### Environment Installations

In the "comparison", each model has its own conda environment. To create the environment, simply run

```bash
conda env create -f environment.yml
```

When GRU is the archicture from the facial decoder, all the environments are the same, extracted from the FaceDiffuser repository. 

```bash
conda activate face_diffuser
```

When the archicture is the Transformer, each speech encoder has its own environment, extracted from each model repository.

```bash
conda activate {MODEL_NAME}
```

Where {MODEL_NAME} can be cosyvoice, wavtokenizer, or speechtokenizer.

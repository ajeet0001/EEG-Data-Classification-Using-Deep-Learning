# EEG-Based Cognitive State Classification

Binary classification of **Rest vs. Mental Arithmetic Task** states from EEG signals using three deep learning models - EEGNet, TSCeption, and ATCNet - built with MNE and PyTorch.

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| EEGNet | 90.6% | 97.3% | 90.0% | 93.5% |
| TSCeption | 86.9% | 89.7% | 93.1% | 91.4% |
| **ATCNet** | **95.6%** | **96.1%** | **98.1%** | **97.1%** |

**ATCNet** achieves the best performance across all metrics. EEGNet outperforms TSCeption despite having ~10× fewer parameters.

---

## Dataset

**PhysioNet Mental Arithmetic Tasks Dataset (eegmat v1.0.0)**
- 36 subjects, two EDF recordings per subject:
  - **Task** - EEG recorded during mental arithmetic (serial subtraction)
  - **Rest** - EEG recorded at rest (eyes open, no task)
- Original sampling rate: 500 Hz | 19 EEG channels
- Access: https://physionet.org/content/eegmat/1.0.0/
- Paper: https://www.mdpi.com/2306-5729/4/1/14

> ⚠️ The dataset requires a free PhysioNet account and signing the data use agreement before download.

---

## Project Structure

```
├── Deep_learning_eeg_task.ipynb   # Main notebook (all steps end-to-end)
├── README.md
├── eegnet_best.pt                 # Saved EEGNet weights (after training)
├── tsception_best.pt              # Saved TSCeption weights
├── atcnet_best.pt                 # Saved ATCNet weights
├── psd_comparison.png             # Band-wise PSD bar chart
├── psd_spectrum.png               # Full Welch spectrum plot
├── model_comparison.png           # Three-way metric comparison chart
├── eegnet_curves.png              # EEGNet training curves
├── tsception_curves.png           # TSCeption training curves
└── atcnet_curves.png              # ATCNet training curves
```

---

## Pipeline Overview

```
EDF Files
    │
    ▼
MNE Preprocessing
(bandpass 1–100 Hz → resample 250 Hz → 4s overlapping epochs)
    │
    ▼
PSD Analysis (Welch)
(Delta / Theta / Alpha / Beta / Gamma band powers)
    │
    ▼
Channel-wise Z-score Normalisation
    │
    ▼
Train / Val / Test Split  (70% / 15% / 15%, stratified)
    │
    ├──► EEGNet      → depthwise-separable CNN
    ├──► TSCeption   → multi-scale temporal CNN
    └──► ATCNet      → attention + dilated causal TCN
    │
    ▼
Evaluation
(Accuracy · Precision · Recall · F1 · Confusion Matrix)
```

---

## Models

### EEGNet
Compact depthwise-separable CNN designed specifically for EEG-based BCIs (Lawhern et al., 2018).
- Temporal conv → depthwise spatial conv → separable conv
- ~2K trainable parameters
- Strong generalisation on small EEG datasets

### TSCeption
Multi-scale temporal CNN that captures different EEG frequency components simultaneously (Liu et al., 2022).
- Three parallel temporal conv branches (0.5 s / 0.25 s / 0.125 s kernels)
- Followed by spatial convolution across channels
- ~20K parameters

### ATCNet
Attention-based Temporal Convolutional Network for EEG (Altaheri et al., 2022).
- **Block 1:** EEGNet-style spatial-spectral feature extraction
- **Block 2:** Multi-head self-attention (captures long-range temporal context)
- **Block 3:** Dilated causal TCN (exponentially growing receptive field)
- Sliding-window ensemble over 3 overlapping segments → soft prediction averaging
- ~50K parameters

---

## PSD Analysis Findings

Welch's method applied across all epochs and channels:

| Band | Range | Observation |
|---|---|---|
| **Delta** | 1–4 Hz | Increased during task (8.2e-11 vs 5.9e-11 µV²/Hz) — strongest difference |
| **Theta** | 4–8 Hz | Minimal difference between states |
| **Alpha** | 8–12 Hz | Minimal difference between states |
| **Beta** | 12–30 Hz | Minimal difference between states |
| **Gamma** | 30–100 Hz | Slightly decreased during task |

> Delta increase during mental arithmetic and Gamma suppression are the primary spectral biomarkers in this dataset.

---

## Setup & Usage

### Requirements

```bash
pip install mne torch torchvision torchaudio scikit-learn \
            scipy numpy pandas matplotlib seaborn tqdm requests wfdb
```

Tested with: Python 3.10+, MNE 1.3+, PyTorch 2.0+

### Running on Google Colab

1. Open the notebook in Colab
2. Upload your EDF files to `/content/your_folder/`
3. Set the three config variables in **Cell 3**:

```python
DATA_DIR    = Path('/content/your_folder')   # folder with EDF files
TASK_SUFFIX = '_1'                            # substring in task filenames
REST_SUFFIX = '_2'                            # substring in rest filenames
```

4. Run all cells (`Runtime → Run all`)

### Running Locally

```bash
git clone https://github.com/ajeet0001/EEG-Data-Classification-Using-Deep-Learning.git
cd EEG-Data-Classification-Using-Deep-Learning
pip install -r requirements.txt
jupyter notebook Deep_learning_eeg_task.ipynb
```

---

## Preprocessing Details

| Parameter | Value |
|---|---|
| Bandpass filter | 1–100 Hz (FIR, Hamming window) |
| Resample | 500 Hz → 250 Hz |
| Epoch duration | 4 seconds |
| Epoch overlap | 50% |
| Normalisation | Channel-wise z-score per epoch |
| Train / Val / Test | 70% / 15% / 15% (stratified) |
| Batch size | 64 |

---

## Training Details

| Parameter | EEGNet | TSCeption | ATCNet |
|---|---|---|---|
| Optimiser | Adam | Adam | Adam |
| Learning rate | 1e-3 | 5e-4 | 5e-4 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| LR schedule | Cosine annealing | Cosine annealing | Cosine annealing |
| Max epochs | 80 | 80 | 80 |
| Early stopping patience | 20 | 20 | 20 |
| Dropout | 0.5 | 0.5 | 0.3 |
| Gradient clipping | 1.0 | 1.0 | 1.0 |

---

## References

- Lawhern et al. (2018). *EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces.* Journal of Neural Engineering.
- Liu et al. (2022). *TSception: Capturing Temporal Dynamics and Spatial Asymmetry from EEG for Emotion Recognition.* IEEE Transactions on Affective Computing.
- Altaheri et al. (2022). *Physics-Informed Attention Temporal Convolutional Network for EEG-Based Motor Imagery Classification.* IEEE Transactions on Industrial Informatics.
- Zyma et al. (2019). *Electroencephalograms during Mental Arithmetic Task Performance.* Data, MDPI.

---

## Author

**Ajeet Choudhary**
M.Tech Software Engineering, Delhi Technological University
[github.com/ajeet0001](https://github.com/ajeet0001) · [linkedin.com/in/ajeet-choudhary](https://linkedin.com/in/ajeet-choudhary)

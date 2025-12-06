# Depression Detection Using Multi-Modal Data

This repository implements an automated depression detection system that integrates **Convolutional Neural Networks (CNNs)** and **Transformer-based Pre-Trained Language Models** to analyze multi-modal data—text, audio, and video. The model emulates clinical depression assessment by fusing these three modalities with adaptive weighting to predict mental health status. The system is evaluated on the **DAIC-WOZ dataset** and incorporates advanced deep learning techniques such as **LSTM with gating mechanisms**, **BiLSTM**, and **hybrid fusion strategies**.

---

## Dataset

The **DAIC-WOZ dataset** (Distress Analysis Interview Corpus) is used, containing:
- 189 interview sessions (59 depressed, 130 non-depressed)
- Audio recordings
- Video recordings with facial landmarks, gaze, and pose features
- Transcribed text with timestamps and speaker labels
- Annotations for depression levels

Data is aligned at sentence and word levels for multi-modal fusion.

---

## Model Architecture

The system explores several model architectures:

1. **SVM & Random Forest** – Baseline models applied to each modality and late fusion.
2. **CNN** – Separate CNNs for text (Conv2D) and audio/video (Conv1D) with max pooling and fully connected layers.
3. **LSTM with/without Gating** – Sentence-level and word-level LSTMs with gating mechanisms to modulate modality influence.
4. **BiLSTM** – Bidirectional LSTM for enhanced contextual understanding.
5. **Hybrid Fusion** – Combines early and late fusion with highway layers for improved feature integration.

---

## Key Features

- **Multi-modal fusion** of text, audio, and video data
- **Gating mechanisms** to weight modalities dynamically
- **Sentence-level alignment** for contextual learning
- **Support for both traditional ML (SVM, RF) and deep learning models (CNN, LSTM, BiLSTM)**
- **Late fusion strategy** for decision-level integration

---

## File Descriptions

| File | Description |
|------|-------------|
| `Dataset.ipynb` | Downloads, extracts, and organizes the DAIC-WOZ dataset |
| `SVM&RF_Text.ipynb` | SVM and Random Forest on text modality |
| `SVM&RF_Video.ipynb` | SVM and Random Forest on video modality |
| `SVM&RF_Audio.ipynb` | SVM and Random Forest on audio modality |
| `SVM&RF_CombinedModalities.ipynb` | Late fusion SVM/RF on all three modalities |
| `Rf_prune.ipynb` | Random Forest pruning implementation |
| `CNN_Audio.ipynb` | CNN model for audio features |
| `CNN_Text.ipynb` | CNN model for text features |
| `CNN_Video.ipynb` | CNN model for video features |
| `LSTM_Without_Gating_Sentence_Level.ipynb` | Sentence-level LSTM without gating |
| `LSTM_With_Gating_Sentence_Level.ipynb` | Sentence-level LSTM with gating |
| `LSTM_WITH_GATING_WordLevel.ipynb` | Word-level LSTM with gating |
| `BiLSTM_WordLevel.ipynb` | Word-level BiLSTM on all modalities |

---

## Results Summary

| Model | Modality | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| LSTM with Gating (Sentence-Level) | Text+Audio+Video | 0.647 | 0.639 | 0.653 |
| CNN | Text | 0.597 | 0.567 | 0.627 |
| SVM Late Fusion | Text+Audio+Video | 0.423 | 0.443 | 0.386 |

*Sentence-level LSTM with gating performed best, highlighting the importance of contextual fusion and modality weighting.*

---

## Conclusion

The proposed model demonstrates that **sentence-level multi-modal fusion with gating mechanisms** significantly improves depression detection accuracy. The integration of CNN and transformer-based features enhances robustness, while late fusion strategies help in interpreting complementary modality information.

---

## Future Work

- Incorporate additional audio features (e.g., pauses, response time)
- Explore motion capture data for body language analysis
- Fine-tune hyperparameters and validate on diverse datasets
- Extend to real-time deployment for clinical support systems

---

## References

See the full reference list in the accompanying PDF for cited works and related research.

---

## How to Use

1. Clone the repository.
2. Run `Dataset.ipynb` to prepare the DAIC-WOZ dataset.
3. Execute the desired model notebooks (e.g., `LSTM_With_Gating_Sentence_Level.ipynb`).
4. Modify hyperparameters and fusion strategies as needed for experimentation.

---

**Keywords:** Depression Detection, CNN, LSTM, BiLSTM, Multi-Modal Fusion, DAIC-WOZ, Gating Mechanism, Late Fusion, Mental Health AI

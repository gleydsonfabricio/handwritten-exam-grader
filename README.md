# 📝 Handwritten Exam Grader

**Final Project — Artificial Intelligence**

An end-to-end pipeline that takes a **photo of a handwritten exam**, corrects its perspective and image quality, extracts the text with **handwriting OCR (TrOCR)**, cleans and organizes the answers with an **LLM (Qwen)**, and finally **grades each open-ended question** by comparing the student's answer to an answer key using semantic similarity.

## How it works

```
 photo ──► 1. Preprocessing ──► 2. Line segmentation ──► 3. Handwriting OCR
                                                                │
 grade ◄── 5. Semantic grading ◄── 4. LLM cleanup & structuring ◄┘
```

| Step | What it does | Tools |
|---|---|---|
| 1️⃣ **Preprocessing** | Detects the sheet of paper (Canny + largest 4-sided contour), applies a perspective transform ("scanner effect"), sharpens the image and binarizes it with adaptive thresholding | OpenCV |
| 2️⃣ **Line segmentation** | Detects text lines on the page and crops each one, sorted top-to-bottom | docTR (`db_resnet50`) |
| 3️⃣ **Handwriting OCR** | Transcribes each cropped line individually | TrOCR (`microsoft/trocr-base-handwritten`) |
| 4️⃣ **LLM cleanup** | Fixes obvious OCR errors, separates the raw text by question number, keeps sub-items, and reduces multiple-choice answers to just the chosen letter | Qwen2.5-1.5B-Instruct |
| 5️⃣ **Automatic grading** | Embeds the student's answer and the answer key, computes cosine similarity, and converts it into a score | `sentence-transformers` (`all-MiniLM-L6-v2`) |

### Grading scale

Each open-ended question is scored from **0.0 to 2.0** in steps of 0.5, based on the cosine similarity between the student's answer and the answer key:

| Similarity | Score |
|---|---|
| ≥ 0.80 | 2.0 |
| ≥ 0.65 | 1.5 |
| ≥ 0.50 | 1.0 |
| ≥ 0.35 | 0.5 |
| < 0.35 | 0.0 |

The thresholds live in `LIMIARES_NOTA` and can be tuned to make grading stricter or more lenient.

## Dataset

This project uses the public dataset [**A Dataset of Digitized Student Examination Papers, Answer Keys, and Manual Evaluations for Automated Grading Research**](https://data.mendeley.com/datasets/sf3kvjwknt/1) (Mendeley Data, DOI: [10.17632/sf3kvjwknt.1](https://doi.org/10.17632/sf3kvjwknt.1)), licensed under **CC BY 4.0**.

It contains the records of a university-level Data Science exam taken by 50 students (Anna University, Chennai), anonymized for public release. The exam has two parts, for a maximum of 50 marks:

- **Part I:** 20 multiple-choice questions (Q1–Q20), 1 mark each
- **Part II:** 15 short-answer questions (Q21–Q35), 2 marks each

| Content | Description |
|---|---|
| `Question.txt` / `answerkey.txt` | The exam questionnaire and the grading rubric / answer key |
| `Student_Pdf/` | 50 scanned, uncorrected handwritten answer sheets (the pipeline's input) |
| `Corrected_Pdf/` | The same 50 sheets manually graded, with teacher annotations |
| `Teacher_manual_marks_Anonymized.csv` | Item-level manual scores (Q1–Q35) for every student, useful as ground truth to evaluate the automatic grader |

The short-answer questions (Q21–Q35) and their 0–2 marks are what the `gabarito_discursivas` answer key and the scoring scale in this project are based on.

> The dataset is **not** included in this repository. Download it from the link above and upload the answer sheet image(s) you want to grade when the notebook asks for a file.

**Citation**

> Dinesh K P (2026). *A Dataset of Digitized Student Examination Papers, Answer Keys, and Manual Evaluations for Automated Grading Research*. Mendeley Data, V1. https://doi.org/10.17632/sf3kvjwknt.1

## Project structure

```
.
├── handwritten-exam-grader.ipynb   # Full pipeline (Google Colab notebook)
└── README.md
```

## Getting started

### Requirements

- Google Colab (the notebook uses `google.colab` for file upload and secrets)
- A GPU runtime is recommended (TrOCR, docTR and Qwen all run much faster on GPU)
- A [Hugging Face access token](https://huggingface.co/settings/tokens)

### Running

1. Open `projeto_ia_apresentacao.ipynb` in Google Colab.
2. Add your Hugging Face token as a Colab secret named `HF_TOKEN`.
3. Run the dependency installation cell (`transformers`, `opencv-python-headless`, `python-doctr[torch]`, `sentence-transformers`, `bitsandbytes`, among others).
4. Run the cells in order. When prompted, upload a photo of the handwritten exam.
5. The notebook prints each intermediate result (preprocessed image, recognized lines, cleaned text) and ends with the per-question scores and the total.

### Example output

```
Q22 | Similarity: 0.71 | Score: 1.5 | Semantic similarity with the answer key: 0.71
...
Grading finished! Total score: X / Y
```

## Customizing the answer key

The answer key is a Python dictionary (`gabarito_discursivas`) mapping each question number to its expected answer. Replace it with your own exam's key. Questions not present in the dictionary are skipped during grading.

```python
gabarito_discursivas = {
    21: "A type of ML where the model is trained on labeled data...",
    22: "Overfitting; the model has memorized the training data...",
    # ...
}
```

## Limitations

- **Photo quality matters.** Preprocessing expects the whole sheet to be visible against a contrasting background; if no 4-sided contour is found, the pipeline stops.
- **OCR errors propagate.** Handwriting recognition is imperfect, and the 1.5B-parameter Qwen model can only fix obvious mistakes.
- **Semantic similarity is an approximation.** Grading is based on embedding similarity, so it may miss factual errors in answers that are worded similarly to the key.
- **Only open-ended questions are graded.** Multiple-choice answers are cleaned by the LLM, but the notebook's answer key and scoring cover the open-ended questions only.

## Acknowledgments

Thanks to Dinesh K P and Anna University, Chennai, for releasing the exam dataset under CC BY 4.0.

## Built with

[OpenCV](https://opencv.org/) · [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten) · [docTR](https://github.com/mindee/doctr) · [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) · [Sentence-Transformers](https://www.sbert.net/) · [Hugging Face Transformers](https://huggingface.co/docs/transformers)

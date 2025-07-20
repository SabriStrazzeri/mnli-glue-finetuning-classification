# 🤖 MNLI Fine-Tuning with Transformers


## 📌 Objective
Fine-tune a transformer-based model (RoBERTa) on the MNLI subset of the GLUE benchmark to classify sentence pairs into Entailment, Neutral, or Contradiction.


## 📂 Dataset
The dataset is loaded from the Hugging Face datasets library using:

`load_dataset("glue", "mnli")`,

which includes a train subset (used for model training), a validation matched subset (as the in-domain validation set), and a validation mismatched subset (as the out-of-domain validation set).


## 🛠️ Tools Used
`Python`, `PyTorch`, `Transformers`, `Datasets`, `Scikit-learn`, `Matplotlib`


## 🧪 Approach
1. **Data Preprocessing**:
   - Filtered short-premise examples.
   - Tokenized sentence pairs using RoBERTa tokenizer.
   - Converted inputs to PyTorch tensors.

2. **Model Setup**:
   - Loaded roberta-base for sequence classification.
   - Configured training parameters using TrainingArguments.

3. **Training**:
   - Trained the model using Hugging Face Trainer.
   - Evaluated performance on validation sets during training.

4. **Evaluation**:
   - Generated predictions on all splits.
   - Computed confusion matrices and classification metrics (accuracy, precision, recall).
   - Compared performance across train, matched, and mismatched splits.
   - Identified class-wise strengths and weaknesses.


## 📈 Results
- Train Accuracy: 89% — strong performance with high recall for Contradiction.
- Validation Matched Accuracy: 84% — good generalization to in-domain data.
- Validation Mismatched Accuracy: 80% — slight drop due to domain shift.
- Neutral class consistently showed lower precision and recall, indicating it is the most challenging to classify.


## 🤓 What I Learned
- Fine-tuning transformer models on NLI tasks requires careful preprocessing and evaluation.
- Hugging Face’s ecosystem simplifies model training and evaluation.
- Confusion matrices and class-wise metrics are essential for diagnosing model behavior.
- Domain mismatch can significantly affect performance, especially for ambiguous classes.


## 📁 Repo Contents
- `notebooks/` – Full training and evaluation pipeline (`mlni_glue_finetuning_classification`) notebook.


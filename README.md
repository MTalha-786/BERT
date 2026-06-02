# BERT
Fine-tune a transformer model (e.g., BERT) to classify news headlines into topic categories. Dataset: AG News Dataset (Available on Hugging Face Datasets)
#  News Topic Classifier: Fine-Tuning BERT with Hugging Face & Gradio

This notebook demonstrates an end-to-end Machine Learning pipeline to fine-tune a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model for text classification.

Using the **AG News dataset**, the model is trained to classify news headlines into four distinct categories:
1. **World**
2. **Sports**
3. **Business**
4. **Sci/Tech**

---

##  Project Workflow Overview
* **Data Preparation:** Loading the AG News dataset and selecting optimized subsets.
* **Text Preprocessing:** Tokenizing text using the `bert-base-uncased` tokenizer.
* **Model Configuration:** Setting up `AutoModelForSequenceClassification` with 4 target labels.
* **Evaluation Metrics:** Tracking accuracy and weighted F1-score during training.
* **Model Training:** Executing the training loop using Hugging Face's `Trainer` API.
* **Interactive Deployment:** Launching a live web interface using `Gradio` for real-time model inference.

---

##  Step 1: Libraries & Dataset Loading
In this stage, we import standard data science libraries (`numpy`, `pandas`), deep learning backends (`torch`), and Hugging Face's core ecosystem (`transformers`, `datasets`, `evaluate`).

To optimize compute resources and training time within Google Colab's environment, we select a shuffled subset:
* **Training Data:** 5,000 samples
* **Testing Data:** 1,000 samples

---

##  Step 2: Tokenization & Data Preprocessing
Raw text cannot be fed directly into a neural network. We utilize the `AutoTokenizer` mapped to `bert-base-uncased` to convert our text headlines into numerical token IDs.

### Preprocessing Configuration:
* **Padding (`max_length`):** Pads shorter sequences to a uniform length.
* **Truncation (`True`):** Cuts off text sequences exceeding our budget limit.
* **Max Length (`64`):** Fixed token length optimized for rapid training on short news headlines.
* **Batched Mapping:** Processes data efficiently in batches across the dataset.

---

##  Step 3: Model Initialization & Training Arguments
We load a pre-trained sequence classification variant of BERT configured explicitly for **4 output classes**.

### Hyperparameter Selection:
* `learning_rate=2e-5`: A small learning rate suited to prevent catastrophic forgetting during transfer learning.
* `per_device_train_batch_size=16`: Optimized batch size to balance GPU VRAM utilization.
* `num_train_epochs=2`: The model will iterate through the training dataset twice.
* `weight_decay=0.01`: Regularization technique to prevent model overfitting.

---

##  Step 4: Evaluation Metrics
To evaluate how well our fine-tuned model generalizes on unseen data, we extract two core classification metrics via the `evaluate` library:
* **Accuracy:** The ratio of correctly predicted headlines over total samples.
* **F1-Score (Weighted):** Accounts for any slight class imbalances by computing the harmonic mean of precision and recall.

The custom `compute_metrics` function applies a `softmax`-equivalent `np.argmax` over model logits to isolate the highest-probability category for comparison against the ground-truth labels.

---

##  Step 5: Model Training & Artifact Preservation
We encapsulate the model, arguments, tokenized datasets, and evaluation tracking within the Hugging Face `Trainer` block.

Once `trainer.train()` concludes, both the optimized model weights and corresponding tokenizing rules are permanently compiled into the `./my_bert_model` directory for local or production use.

---

##  Step 6: App Inference Logic & Gradio Web Deployment
The final phase moves the fine-tuned model onto the available hardware acceleration backend (**CUDA GPU** if active, otherwise **CPU**) and constructs a user-facing interactive application layer.

### How Inference Works:
1. The user types a raw headline string into the UI text field.
2. The tokenizer transforms the input text into dynamic PyTorch tensors (`pt`).
3. The model processes the input with gradient tracking disabled (`torch.no_grad()`).
4. Raw output logits are passed through a `Softmax` layer to generate clean probability distributions across **World, Sports, Business, and Sci/Tech**.
5. **Gradio** deploys an interactive, shareable web link (`share=True`) to test the classifier in real-time.


---
---

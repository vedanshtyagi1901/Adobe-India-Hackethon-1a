from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Updated model path to the fine-tuned model
model_path = "./hf-model/fine_tuned_quantized"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4, local_files_only=True)

def classify_heading(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
    return predicted_class
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from pdfplumber import open as pdf_open

# Load the pre-trained MiniLM model with ignore_mismatched_sizes
model_path = "./hf-model/nreimers/MiniLM-L6-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=4,
    local_files_only=True,
    ignore_mismatched_sizes=True
)

import difflib

class HeadingDataset(Dataset):
    def __init__(self, pdf_dir, json_dir, tokenizer, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        pdf_dir = os.path.abspath(pdf_dir)
        json_dir = os.path.abspath(json_dir)
        
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            pdf_base = os.path.splitext(pdf_file)[0]
            json_file = os.path.join(json_dir, pdf_base + '.json')
            
            if os.path.exists(json_file):
                with pdf_open(os.path.join(pdf_dir, pdf_file)) as pdf:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        outline = json_data.get('outline', [])
                        for item in outline:
                            level = item['level']
                            json_text = item['text']
                            page_num = item['page'] - 1
                            if 0 <= page_num < len(pdf.pages):
                                page_text = pdf.pages[page_num].extract_text()
                                lines = [ln.strip() for ln in page_text.split('\n') if ln.strip()]
                                for line in lines:
                                    label = {'H1': 1, 'H2': 2, 'H3': 3}.get(level, 0)
                                    self.data.append((line, label))
                                    print(f"Matched: {line} -> Label: {label}")
        print(f"Number of training samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Function to train the model
def train_model(model, train_dataloader, num_epochs=1, learning_rate=2e-5):
    device = torch.device('cpu')  # No GPU as per constraints
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")
    model.eval()

# Function to quantize the model
def quantize_model(model, tokenizer, output_dir="./hf-model/fine_tuned_quantized"):
    model.eval()
    # Set up weight-only quantization
    model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.PlaceholderObserver.with_args(dtype=torch.quint8),
        weight=torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    )
    # Prepare model for quantization (weights only)
    torch.quantization.prepare(model, inplace=True)
    
    # Convert to quantized model without relying on activation observers
    torch.quantization.convert(model, inplace=True, mapping=None)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Quantized model saved to {output_dir}")

# Main execution
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(base_dir, "input")
    json_dir = os.path.join(base_dir, "input")  # Adjust if JSONs are elsewhere
    
    os.makedirs(pdf_dir, exist_ok=True)
    
    dataset = HeadingDataset(pdf_dir, json_dir, tokenizer)
    print(f"Number of training samples: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    train_model(model, dataloader)
    
    # Save the fine-tuned model
    os.makedirs("./hf-model/fine_tuned", exist_ok=True)
    model.save_pretrained('./hf-model/fine_tuned')
    tokenizer.save_pretrained('./hf-model/fine_tuned')
    
    # Quantize the fine-tuned model
    try:
        quantize_model(model, tokenizer)
    except Exception as e:
        print(f"Quantization failed with error: {e}")
        print("Reverting to unquantized model as fallback.")
        # Fallback to save the original fine-tuned model
        model.save_pretrained('./hf-model/fine_tuned_quantized')
        tokenizer.save_pretrained('./hf-model/fine_tuned_quantized')
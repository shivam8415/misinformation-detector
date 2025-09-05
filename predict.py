# predict.py
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    pipeline
)
import torch

# 1. Load the saved model and tokenizer
model_path = "./my_fake_news_model"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 2. Create a classification pipeline for easy use
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
)

# 3. Get user input and predict
print("🤖 AI Misinformation Detector Prototype")
print("Type a news headline or sentence to check (type 'quit' to exit).")

while True:
    user_input = input("\nYour text: ")
    if user_input.lower() in ['quit', 'exit']:
        break

    # Perform prediction
    result = classifier(user_input)[0]
    label = result['label']
    score = result['score']

    print(f"Prediction: {label} (confidence: {score:.4f})")
    if label == "FAKE":
        print("⚠️  Warning: This appears to be potentially misleading information.")
    else:
        print("✅ This appears to be reliable information.")
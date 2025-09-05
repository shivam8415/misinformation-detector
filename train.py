# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Load the tiny dataset
df = pd.read_csv('better_news_data.csv')
print("Dataset loaded:")
print(df.head())

# 2. Split the data
X_train, X_temp, y_train, y_temp = train_test_split(
    df['text'], df['label'], test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Create Hugging Face Dataset objects for easier handling
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# 3. Load Tokenizer and Model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2, # Binary classification
    id2label={0: "REAL", 1: "FAKE"},
    label2id={"REAL": 0, "FAKE": 1}
)

# 4. Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128, # Use smaller max_length for faster prototyping
        return_tensors="pt"
    )

# Apply tokenization
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 5. Define Metrics Function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 6. Set Up Training Arguments
training_args = TrainingArguments(
    output_dir='./prototype_model',  # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    eval_strategy="epoch",           # evaluate after each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 7. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 8. Train!
print("Starting training...")
trainer.train()
print("Training finished!")

# 9. Evaluate on the test set
print("\nEvaluating on test set:")
results = trainer.evaluate(tokenized_test)
print(f"Test Results: {results}")

# 10. Save the model for later use
trainer.save_model("./my_fake_news_model")
tokenizer.save_pretrained("./my_fake_news_model")
print("Model saved to './my_fake_news_model'")
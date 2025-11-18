# Install dependencies (run once)
# pip install transformers torch

from transformers import pipeline
import torch

# Load pre-trained NER model with PyTorch backend
print(f"Using PyTorch version: {torch.__version__}")
ner_model = pipeline("ner", 
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    grouped_entities=True,
                    framework="pt")  # Explicitly use PyTorch

# Example text
text = "Apple Inc. was founded by Steve Jobs in California."

# Perform NER
entities = ner_model(text)

# Display results
print("Named Entity Recognition Results:")
print("=" * 50)
print(f"Input Text: {text}")
print("\nNamed Entities Found:")
for i, entity in enumerate(entities, 1):
    print(f"{i}. Entity: '{entity['word']}' | Label: {entity['entity_group']} | Confidence: {entity['score']:.2f}")

print(f"\nTotal entities found: {len(entities)}")
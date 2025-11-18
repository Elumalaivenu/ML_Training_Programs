# Install dependencies (run once)
# pip install transformers torch

from transformers import pipeline
import torch

# Load pre-trained sentiment-analysis model with PyTorch backend
print(f"Using PyTorch version: {torch.__version__}")
sentiment_analyzer = pipeline("sentiment-analysis", 
                            model="distilbert-base-uncased-finetuned-sst-2-english",
                            framework="pt")  # Explicitly use PyTorch

# Example product reviews
reviews = [
    "This phone has amazing battery life!",
    "The laptop stopped working after a week. Terrible quality.",
    "The headphones are okay, nothing special."
]

print("Sentiment Analysis Results:")
print("=" * 60)

# Analyze each review
for i, review in enumerate(reviews, 1):
    result = sentiment_analyzer(review)[0]
    print(f"Review {i}: {review}")
    print(f"Sentiment: {result['label']} | Confidence: {result['score']:.2f}")
    print("-" * 60)
from transformers import pipeline
from huggingface_hub import login

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

pipe = pipeline(
    model=model_path, tokenizer=model_path, return_all_scores=True, device="mps:0"
)

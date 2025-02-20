from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

tokens = tokenizer("Hello world", return_tensors="pt")
output = model(**tokens)[0]

output.shape

for token in tokens["input_ids"][0]:
    print(tokenizer.decode(token))

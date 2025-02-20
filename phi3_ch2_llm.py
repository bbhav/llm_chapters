import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="mps",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")


prompt = "Write an email apologizing for the delay in the project delivery. Explain how it happened.<|assistant|>"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("mps")

generation_output = model.generate(input_ids=input_ids, max_new_tokens=20)

print(tokenizer.decode(generation_output[0]))

for id in input_ids[0]:
    print(tokenizer.decode(id))

text = """ ENGLISH and CAPITALIZATON 

show_tokens False None elif ==> = else: two tabs: "" Three tabs: " "
 12.0 * 50 = 600

"""

colors_list = [
    "102;194;165",
    "252;141;98",
    "141;160;203",
    "231;138;195",
    "166;216;84",
    "255;217;47",
]


def show_tokens(sentence, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids
    for idx, t in enumerate(token_ids):
        print(
            f"\x1b[0;30;48;2;{colors_list[idx % len(colors_list)]}m"
            + tokenizer.decode(t)
            + "\x1b[0m",
            end="",
        )

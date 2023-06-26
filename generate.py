# https://huggingface.co/docs/transformers/main_classes/pipelines
# text-generation pipeline

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import tqdm
import torch
from accelerate import Accelerator
import os

LOAD = "/home/willm/checkpoints/BookCorpus2"
SAVE = "/home/willm/splits/BookCorpus2"

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoints = [int(c) for c in os.listdir(LOAD) if c.isnumeric()]
save_path = os.path.join(LOAD, str(max(checkpoints)))

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"

model = GPT2LMHeadModel(GPT2Config())
model.cuda()
optimizer = torch.optim.AdamW(model.parameters())

accelerator = Accelerator()
model = accelerator.prepare(model)
accelerator.load_state(save_path)

# Compare: https://huggingface.co/blog/how-to-generate

def iterate_generate(tokenizer, model, full_length: int, context_length: int = 512, stride: int = 512, n_beams: int = 32):
    input_ids = tokenizer.encode("The", return_tensors="pt")
    pbar = tqdm.tqdm(total=full_length)
    pbar.update(input_ids.size(1))
    while input_ids.size(1) < full_length:
        context = input_ids[:, -context_length:].to(device)
        output_ids = model.generate(context,
                                    max_new_tokens=stride,
                                    pad_token_id=50256,
                                    num_beams=n_beams,
                                    no_repeat_ngram_size=2,)
        input_ids = torch.cat([input_ids[:, :-context_length], output_ids.cpu()], dim=1)
        pbar.update(input_ids.size(1) - pbar.n)
    pbar.close()
    return input_ids

print("Generating...")
input_ids = iterate_generate(tokenizer, model, 10000).squeeze()
print("Saving...")

with open(os.path.join(SAVE, "gpt2.txt"), "w") as tokens_fh:
    contents = " ".join(str(x.item()) for x in input_ids)
    tokens_fh.write(contents)

with open(os.path.join(SAVE, "gpt2-raw.txt"), "w") as raw_fh:
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    raw_fh.write(text)

print("Done!")

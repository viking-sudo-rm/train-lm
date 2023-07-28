# https://huggingface.co/docs/transformers/main_classes/pipelines
# text-generation pipeline

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import set_seed
import tqdm
import torch
from accelerate import Accelerator
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("corpus", type=str)
# parser.add_argument("--ckpts", "-c", nargs="+", type=int)
# parser.add_argument("--all", "-a", action="store_true")
args = parser.parse_args()

LOAD = f"/home/willm/checkpoints/{args.corpus}"
SAVE = f"/home/willm/splits/{args.corpus}"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"

# Compare: https://huggingface.co/blog/how-to-generate

def get_all_params():
    all_params = {}
    for b in [1, 2, 4, 8, 16, 32]:
        all_params[f"beam{b}-norepeat"] = dict(num_beams=b, no_repeat_ngram_size=2)
        all_params[f"beam{b}"] = dict(num_beams=b)
    all_params["sample"] = dict(do_sample=True)
    for k in [10, 100, 400, 800]:
        all_params[f"topk{k}"] = dict(do_sample=True, top_k=k)
    for p in [0.8, 0.85, 0.9, 0.95]:
        all_params[f"topp{p}"] = dict(do_sample=True, top_p=p)
    for temp in [0.8, 0.9, 1.1, 1.2]:
        all_params[f"temp{temp}"] = dict(do_sample=True, temperature=temp)
    return all_params

@torch.no_grad()
def iterate_generate(tokenizer, model, full_length: int, context_length: int = 512, stride: int = 512, params: dict = {}, seed=42):
    set_seed(42)
    input_ids = tokenizer.encode("The", return_tensors="pt")
    pbar = tqdm.tqdm(total=full_length)
    pbar.update(input_ids.size(1))
    while input_ids.size(1) < full_length:
        context = input_ids[:, -context_length:].to(device)
        output_ids = model.generate(context,
                                    max_new_tokens=stride,
                                    pad_token_id=50256,
                                    **params)
        input_ids = torch.cat([input_ids[:, :-context_length], output_ids.cpu()], dim=1)
        pbar.update(input_ids.size(1) - pbar.n)
    pbar.close()
    return input_ids

checkpoints = [int(c) for c in os.listdir(LOAD) if c.isnumeric()]
ckpt = max(checkpoints)
# if args.ckpts:
#     checkpoints = args.ckpts
# elif not args.all:
#     checkpoints = [max_ckpt]

all_params = get_all_params()

# for ckpt in checkpoints:
# print("Processing checkpoint", ckpt)
model = GPT2LMHeadModel(GPT2Config())
model.cuda()
accelerator = Accelerator()
model = accelerator.prepare(model)
load_path = os.path.join(LOAD, str(ckpt))
accelerator.load_state(load_path)

for name, params in all_params.items():
    print(f"Generating {name}...")
    input_ids = iterate_generate(tokenizer, model, 10000, params=params).squeeze()

    dir_path = os.path.join(SAVE, "gpt2")
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    with open(os.path.join(SAVE, "gpt2", f"{name}.txt"), "w") as tokens_fh:
        contents = " ".join(str(x.item()) for x in input_ids)
        tokens_fh.write(contents)

    with open(os.path.join(SAVE, "gpt2", f"{name}-raw.txt"), "w") as raw_fh:
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        raw_fh.write(text)

print("Done!")

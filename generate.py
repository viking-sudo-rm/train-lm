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
parser.add_argument("--ckpts", "-c", nargs="+", type=int)
parser.add_argument("--all", "-a", action="store_true")
parser.add_argument("--mode", choices=["grid", "standard"], default="grid")
parser.add_argument("--n_tokens", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_seeds", type="int", default=10)
args = parser.parse_args()

LOAD = f"/home/willm/checkpoints/{args.corpus}"
SAVE = f"/home/willm/splits/{args.corpus}"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"

# Compare: https://huggingface.co/blog/how-to-generate

def get_params_grid(args):
    all_params = {}
    for b in [1, 2, 4, 8, 16, 32]:
        all_params[f"beam{b}-norepeat"] = dict(num_beams=b, no_repeat_ngram_size=2)
        all_params[f"beam{b}"] = dict(num_beams=b)
    all_params["sample"] = dict(do_sample=True)
    for k in [2, 8, 32, 128, 512]:
        all_params[f"topk{k}"] = dict(do_sample=True, top_k=k)
    for p in [0.75, 0.8, 0.85, 0.9, 0.95]:
        all_params[f"topp{p}"] = dict(do_sample=True, top_p=p)
    for temp in [0.75, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1]:
        all_params[f"temp{temp}"] = dict(do_sample=True, temperature=temp)
    return all_params

@torch.no_grad()
def iterate_generate(tokenizer, model, full_length: int, context_length: int = 512, stride: int = 512, params: dict = {}, seed=42):
    set_seed(seed)
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

def process_checkpoint(ckpt, all_params):
    model = GPT2LMHeadModel(GPT2Config())
    model.cuda()
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    load_path = os.path.join(LOAD, str(ckpt))
    accelerator.load_state(load_path)

    for name, params in all_params.items():
        # Only do multiple seeds for nondeterministic decoding strategies.
        seeds = range(args.seed, args.seed + args.n_seeds) if "do_sample" in params else [args.seed]
        for seed in seeds:
            print(f"Generating {name} (seed={seed})...")
            input_ids = iterate_generate(tokenizer, model, args.n_tokens, params=params).squeeze()

            dir_path = os.path.join(SAVE, "gpt2", str(ckpt))
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            with open(os.path.join(SAVE, "gpt2", str(ckpt), f"{name}-{seed}.txt"), "w") as tokens_fh:
                contents = " ".join(str(x.item()) for x in input_ids)
                tokens_fh.write(contents)
            with open(os.path.join(SAVE, "gpt2", str(ckpt), f"{name}-{seed}-raw.txt"), "w") as raw_fh:
                text = tokenizer.decode(input_ids, skip_special_tokens=True)
                raw_fh.write(text)

standard = dict(
    do_sample=True,
    # It seems important to have no_repeat_ngram_size on wikitext to avoid generating !!...
    # https://discuss.huggingface.co/t/gpt-2-trained-models-output-repeated/12962/2
    no_repeat_ngram_size=3,
    top_p=.95,
    # top_k=100,
    # temperature=.8,
)

all_params = get_params_grid(args) if args.mode == "grid" else {"standard": standard}

checkpoints = list(sorted([int(c) for c in os.listdir(LOAD) if c.isnumeric()]))
if args.ckpts:
    checkpoints = args.ckpts
elif not args.all:
    checkpoints = [max(checkpoints)]

print(f"All checkpoints: {checkpoints}")
print(f"Total # params: {len(all_params)}\n")

for ckpt in checkpoints:
    print("=" * 5, "Checkpoint:", ckpt, "=" * 5)
    process_checkpoint(ckpt, all_params)
print("Done!")

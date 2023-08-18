from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from argparse import ArgumentParser
import tqdm
import torch
import json
# from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, ConstantLR
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

from src.utils import get_decay_partition

CONTEXT_LENGTH = 1024
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16

# Hyperparameters suggested by Ananya.
# TODO: Cosine schedule after warm up, not constant.
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["BookCorpus2", "Wikipedia (en)", "YoutubeSubtitles", "HackerNews", "wikitext-2-raw", "wikitext-103-raw"])
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--wd", type=float, default=.1)
    parser.add_argument("--beta1", type=float, default=.9)
    parser.add_argument("--beta2", type=float, default=.95)
    # Batch size should be: 500000 tokens / (1024 tokens/example) = 488 examples, which is ~ 30 * 16.
    parser.add_argument("--step_threshold", type=int, default=30)
    parser.add_argument("--save_threshold", type=int, default=10000)
    parser.add_argument("--eval_threshold", type=int, default=1000)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=1)
    # This argument can be used to replicate original training procedure on long corpora like BookCorpus2 (changed Aug 7).
    parser.add_argument("--throw_out_short", action="store_true",
                        help="When creating examples, throw out sequences below max length.")
    return parser.parse_args()

def get_examples(path) -> TensorDataset:
    examples = []
    lengths = []
    with open(path, "r") as fh:
        for line in tqdm.tqdm(fh.readlines()):
            tokens = [int(x) for x in line.split()]
            for idx in range(0, len(tokens), CONTEXT_LENGTH):
                subtokens = tokens[idx:idx + CONTEXT_LENGTH]
                if not args.throw_out_short or len(subtokens) == CONTEXT_LENGTH:
                    examples.append(torch.tensor(subtokens))
                    lengths.append(len(subtokens))
    examples = pad_sequence(examples, batch_first=True)
    mask = torch.arange(examples.size(1)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    return TensorDataset(examples, mask)


args = parse_args()
if args.save_threshold == -1:
    args.save_threshold = float("inf")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading train examples...")
train = get_examples(f"/home/willm/splits/{args.dataset}/train.txt")
# train = get_examples(f"/home/willm/splits/{args.dataset}/val.txt")
train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

print("Loading val examples...")
val = get_examples(f"/home/willm/splits/{args.dataset}/val.txt")
val_dataloader = DataLoader(val, batch_size=EVAL_BATCH_SIZE)

model = GPT2LMHeadModel(GPT2Config())
model.to(device)
all_params, decay, no_decay = get_decay_partition(model.transformer)  # Sidestep weight tying in LM head.
optim_groups = [
    {"params": [all_params[pn] for pn in sorted(list(decay))], "weight_decay": args.wd},
    {"params": [all_params[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(args.beta1, args.beta2))
# scheduler = LinearLR(optimizer, start_factor=1e-3, total_iters=len(train) // BATCH_SIZE)
scheduler = ChainedScheduler([
    LinearLR(optimizer, start_factor=1e-3, total_iters=400000000 / (CONTEXT_LENGTH * BATCH_SIZE)),
    ConstantLR(optimizer, factor=1., total_iters=1e20),
])

# Save optimizer state, but should delete afterwards if you don't need it!
accelerator = Accelerator()
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

last_step = 0
steps = []
losses = []

for epoch in range(args.n_epochs):
    print(f"Starting epoch {epoch}...")
    pbar = tqdm.trange(len(train_dataloader))
    for step, (batch, batch_mask) in enumerate(iter(train_dataloader)):
        model.train()
        batch = batch.to(device)
        batch_mask = batch_mask.to(device)
        output = model(batch, attention_mask=batch_mask, labels=batch)
        accelerator.backward(output.loss)

        if (step + 1) % args.step_threshold == 0 or step + 1 == len(train_dataloader):
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % args.save_threshold == 0 or step + 1 == len(train_dataloader):
            if not args.no_save:
                pbar.set_description("SAVE")
                accelerator.save_state(output_dir=f"/home/willm/checkpoints/{args.dataset}/{last_step + step}")

        if (step + 1) % args.eval_threshold == 0 or step + 1 == len(train_dataloader):
            pbar.set_description("EVAL")
            with torch.no_grad():
                model.eval()
                cum_loss = 0.
                cum_sum = 0
                for val_batch, val_batch_mask in iter(val_dataloader):
                    val_batch = val_batch.to(device)
                    val_batch_mask = val_batch_mask.to(device)
                    output = model(val_batch, attention_mask=val_batch_mask, labels=val_batch)
                    cum_loss += output.loss.item() * len(val_batch)
                    cum_sum += len(val_batch)
            steps.append(last_step + step)
            losses.append(cum_loss / cum_sum)

        pbar.set_description(f"train: {output.loss.item():.2f}, val: {losses[-1] if losses else 0.:.2f}")
        pbar.update()
    pbar.close()
    last_step += len(train_dataloader)
    steps_per_epoch = len(train_dataloader)

print("=== OUTPUT ===")
print()
print("steps:", steps)
print()
print("losses:", losses)

blob = {
    "steps": steps,
    "losses": losses,
    "steps_per_epoch": steps_per_epoch,
}
with open(f"/home/willm/checkpoints/{args.dataset}/metrics.json", "w") as fh:
    json.dump(blob, fh)

import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from accelerate import Accelerator
from tqdm import tqdm, trange
import pickle
import os
from argparse import ArgumentParser
from collections import Counter

from py_rusty_dawg import Dawg

def get_load_path(corpus):
    ckpts = os.listdir(f"/home/willm/checkpoints/{corpus}")
    ckpt = max(int(ckpt) for ckpt in ckpts if ckpt.isnumeric())
    return f"/home/willm/checkpoints/{corpus}/{ckpt}"

parser = ArgumentParser()
parser.add_argument("corpus", type=str)
parser.add_argument("--n_samples", "-n", type=int, default=10000000)
parser.add_argument("--max_length", "-d", type=int, default=10)
parser.add_argument("--batch_size", "-b", type=int, default=16)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dawg_path = f"/home/willm/results/{args.corpus}.dawg" if not args.debug else "/home/willm/results/wikitext-2-raw.dawg"
save_path = f"/home/willm/results/{args.corpus}/count-vs-loss.dat"
load_path = get_load_path(args.corpus)

print(f"Loading DAWG from {dawg_path}...")
dawg = Dawg.load(dawg_path)
print(f"Loading model from {load_path}...")
model = GPT2LMHeadModel(GPT2Config())
model.eval()
model.to(device)
accelerator = Accelerator()
model = accelerator.prepare(model)
accelerator.load_state(load_path)
loss_fn = CrossEntropyLoss(reduction="none")

vocab_size = model.lm_head.weight.size(0)

def get_target(next_tokens):
    global vocab_size
    # TODO: Sparse tensor.
    counter = Counter(next_tokens)
    total = sum(counter.values())
    tensor = torch.zeros(vocab_size)
    for index, value in counter.items():
        tensor[index] = value / total
    tensor = tensor.to_sparse()
    return tensor

@torch.no_grad()
def get_losses(factors, target_dists, batch_size):
    lengths = torch.LongTensor([len(f) for f in factors])
    input_ids = pad_sequence([torch.LongTensor(f) for f in factors], batch_first=True)
    labels = torch.LongTensor([f[-1] for f in factors])
    all_losses = []
    all_cross_entropies = []

    for batch_idx in trange(0, len(factors), batch_size):
        batch_lengths = lengths[batch_idx:batch_idx + batch_size].to(device)
        batch_input_ids = input_ids[batch_idx:batch_idx + batch_size].to(device)
        batch_labels = labels[batch_idx:batch_idx + batch_size].to(device)
        # batch_target_dists = target_dists[batch_idx:batch_idx + batch_size].to(device)
        last_idx = min(batch_idx + batch_size, len(target_dists))
        batch_target_dists = torch.stack([target_dists[i].to_dense()
                                          for i in range(batch_idx, last_idx)], dim=0)
        output = model(batch_input_ids, output_hidden_states=True)
        hidden_states = output.hidden_states[-1] # Final layer
        arange = torch.arange(len(batch_lengths)).to(device)

        # Compute loss of final token.
        final_states = hidden_states[arange, batch_lengths - 2]
        logits = model.lm_head(final_states)
        losses = loss_fn(logits, batch_labels)
        all_losses.extend(losses.tolist())

        # Compute cross entropy of possible continuations.
        final_states = hidden_states[arange, batch_lengths - 2]
        logits = model.lm_head(final_states)
        cross_entropies = loss_fn(logits, batch_target_dists.to(device))
        all_cross_entropies.extend(cross_entropies.tolist())

    return all_losses, all_cross_entropies

def explore_counts_and_logprobs():
    print("Searching states in DAWG...")
    stack = [(0, -1, None, 0)]
    explored = {}
    next_tokens = {}
    factors = {0: ()}
    pbar = tqdm(total=args.n_samples)
    while stack and len(explored) < args.n_samples:
        prefix, prev_state, token, state = stack.pop(-1)
        edges = dawg.get_edges(state)
        pbar.update()
        if prefix > 0:
            train_count = dawg.get_count(state)
            train_logprob = np.log(dawg.get_count(state)) - np.log(dawg.get_count(prev_state))
            explored[state] = (prefix, token, train_count, train_logprob)
            factors[state] = factors[prev_state] + (token,)
            next_tokens[state] = [pair[1] for pair in edges]
        if prefix < args.max_length:
            for next_state, token in edges:
                if next_state in explored:
                    continue
                new_entry = (prefix + 1, state, token, next_state)
                stack.append(new_entry)
    pbar.close()

    print("Batching loss computation...")
    states = list(explored.keys())
    factors_list = [factors[state] for state in states]
    target_dists = [get_target(next_tokens[state]) for state in states]
    losses, cross_entropies = get_losses(factors_list, target_dists, batch_size=args.batch_size)
    for state, loss, cross_entropy in zip(states, losses, cross_entropies):
        explored[state] = explored[state] + (loss, state, cross_entropy)

    dir = os.path.dirname(save_path)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    with open(save_path, "wb") as fh:
        pickle.dump(explored, fh)


if __name__ == "__main__":
    explore_counts_and_logprobs()

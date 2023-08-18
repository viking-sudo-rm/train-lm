from argparse import ArgumentParser
import numpy as np
import torch
import os
import pickle

from py_rusty_dawg import Dawg, KNLM, good_turing_estimate

# TODO: Compute suffix length here, in Python.

def get_suffix_context(dawg, dataset):
    state = dawg.get_initial()
    length = 0
    lengths = []
    for token in dataset:
        state, length = dawg.transition_and_count(state, token, length)
        lengths.append(length)
    return lengths


parser = ArgumentParser()
parser.add_argument("corpus", type=str)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--save_path", type=str, default="/home/willm/results/{corpus}/suffix-contexts.dat")
args = parser.parse_args()

if args.corpus == "BookCorpus2":
    paths = ["/home/willm/splits/BookCorpus2/val.txt"]
    basedir = "/home/willm/splits/BookCorpus2/gpt2/138688"
    paths.extend(os.path.join(basedir, fname) for fname in os.listdir(basedir)
                 if fname.endswith(".txt") and not fname.endswith("-raw.txt"))
elif args.corpus in ["wikitext-2-raw", "wikitext-103-raw"]:
    paths = [f"/home/willm/splits/{args.corpus}/val.txt"]
    basedir = f"/home/willm/splits/{args.corpus}/gpt2"
    for ckpt in os.listdir(basedir):
        ckpt_path = os.path.join(basedir, ckpt)
        if not os.path.isdir(ckpt_path):
            continue
        paths.extend(os.path.join(ckpt_path, fname) for fname in os.listdir(ckpt_path)
                     if fname.endswith(".txt") and not fname.endswith("-raw.txt"))

dawg_path = f"/home/willm/results/{args.corpus}.dawg" if not args.debug else "/home/willm/results/wikitext-2-raw.dawg"
print(f"Loading DAWG from {dawg_path}...")
dawg = Dawg.load(dawg_path)
print("DAWG loaded!")

results = {}
for path in paths:
    print(f"Evaluating {path}...")
    with open(path, "r") as fh:
        tokens = [int(x) for x in fh.read().strip().split()]
    results[path] = get_suffix_context(dawg, tokens)

save_path = args.save_path.format(corpus=args.corpus)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "wb") as fh:
    pickle.dump(results, fh)
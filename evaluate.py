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

def get_avg_cross_entropy(lm, dawg, dataset):
    # FIXME: Use train length here.
    gte = good_turing_estimate(dawg, len(dataset))
    cross_entropy = 0.
    lm.reset(dawg)
    for token in dataset:
        prob = lm.get_probability(dawg, token, gte)
        cross_entropy -= np.log2(prob)
        lm.update(dawg, token)
    return cross_entropy / len(dataset)

parser = ArgumentParser()
parser.add_argument("corpus", type=str)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--save_path", type=str, default="/home/willm/results/BookCorpus2/suffix-contexts.dat")
args = parser.parse_args()

paths = ["/home/willm/splits/BookCorpus2/val.txt"]
basedir = "/home/willm/splits/BookCorpus2/gpt2"
paths.extend(os.path.join(basedir, fname) for fname in os.listdir(basedir)
             if fname.endswith(".txt") and not fname.endswith("-raw.txt"))

dawg_path = f"/home/willm/results/{args.corpus}.dawg" if not args.debug else "/home/willm/results/wikitext-2-raw.dawg"
print(f"Loading DAWG from {dawg_path}...")
dawg = Dawg.load(dawg_path)
# lm = KNLM("4gram_kn-0.2", 0.2, 4)
print("DAWG loaded!")

results = {}
for path in paths:
    print(f"Evaluating {path}...")
    with open(path, "r") as fh:
        tokens = [int(x) for x in fh.read().strip().split()]
        test_tokens = tokens[:10000]
    lengths = get_suffix_context(dawg, tokens)
    results[path] = lengths

with open(args.save_path, "wb") as fh:
    pickle.dump(results, fh)

# print("cross ent:", get_avg_cross_entropy(lm, dawg, gen_tokens))

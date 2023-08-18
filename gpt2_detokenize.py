"""Convert a file of space-separated tokens to plaintext"""

from argparse import ArgumentParser
from tqdm import tqdm

from transformers import GPT2TokenizerFast

parser = ArgumentParser()
parser.add_argument("read_path", type=str)
parser.add_argument("save_path", type=str)
args = parser.parse_args()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

decoded_lines = []

print("Reading from", args.read_path)
with open(args.read_path) as fh:
    for line in tqdm(fh.readlines()):
        token_ids = [int(x) for x in line.strip().split()]
        decoded_line = tokenizer.decode(token_ids)
        decoded_lines.append(decoded_line)

print("Saving to", args.save_path)
with open(args.save_path, "w") as fh:
    for line in tqdm(decoded_lines):
        fh.write(line)
        fh.write("\n")
    
print("Done!")
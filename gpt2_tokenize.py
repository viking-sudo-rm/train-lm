from argparse import ArgumentParser
from tqdm import tqdm

from transformers import GPT2TokenizerFast

parser = ArgumentParser()
parser.add_argument("read_path", type=str)
parser.add_argument("save_path", type=str)
args = parser.parse_args()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

read_fh = open(args.read_path)
save_fh = open(args.save_path, "w")

for line in tqdm(read_fh.readlines()):
    line = line.strip()
    if not line:
        continue
    token_ids = tokenizer(line).input_ids
    save_fh.write(" ".join(str(x) for x in token_ids) + "\n")

read_fh.close()
save_fh.close()

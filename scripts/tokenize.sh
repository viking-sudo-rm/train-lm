CORPUS="/home/willm/splits/wikitext-2-raw"

python gpt2_tokenize.py $CORPUS/wiki.train.raw $CORPUS/train.txt
python gpt2_tokenize.py $CORPUS/wiki.valid.raw $CORPUS/val.txt
python gpt2_tokenize.py $CORPUS/wiki.test.raw $CORPUS/test.txt
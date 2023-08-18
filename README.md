# train-lm

Minimal code for training GPT-2 on data from The Pile with good hyperparameter setup.

## Tokenizing an untokenized dataset

```
source scripts/tokenize.sh /home/willm/splits/wikitext-2-raw
```

## Training Wikitext2 models for multiple epochs

To train the models, we can do the following:

```
python train.py wikitext-2-raw --n_epochs=10 --save_threshold=-1
```

To generate text from the models, we can do:

```
python generate.py wikitext-2-raw --all
```
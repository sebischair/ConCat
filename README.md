# ConCat
Repository containing the code described in the ICAART 2025 paper: *Lexical Substitution is not Synonym Substitution: On the Importance of Producing Contextually Relevant Word Substitutes*

![ConCat](https://github.com/sebischair/ConCat/blob/main/concat.png?raw=true "ConCat")

## Getting Started
Using `Concat` is simple! First, simply install the package using the following command:

`pip install concat-ls`

Then, load ConCat and getting LSing!

`from concat import ConCat`

`X = ConCat()`

`X.lexsub("hello world", "hello", K=5)`

The `lexsub` function takes the following parameters: `lexsub([CONTEXT (sentence)], [TARGET WORD (taregt)], [TOPK (K)])`. By default, `K=5`.

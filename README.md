doc2vec_pytorch


# Build dataset
```bash
python src/data.py build --datapath ./data/example.txt --savedir cache
```


# Train
```bash
python src/train.py run --datadir ./cache/ --savedir results
```


# Evaluate
```bash
python src/eval.py most_similar --datadir ./cache/ --textpath ./data/example.txt --modelpath results/D.pth --doc-id 0
```

doc2vec_pytorch


# Build dataset
```bash
python src/data.py build --datapath ./data/example.txt --savedir cache
```


# Train
```bash
python src/train.py run --datadir ./cache/ --savedir results
```

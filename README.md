
## Project

```bash
tree -I .venv -I  ml.egg-info -I  __pychache__  -L 3 
```


```bash
ml
├── README.md
├── dataset
│   └── aclImdb
│       ├── README
│       ├── imdb.vocab
│       ├── imdbEr.txt
│       ├── test
│       └── train
├── logic_regression
│   ├── __init__.py
│   ├── assets
│   │   ├── evaluation_embedding_dim=8.csv
│   │   ├── model.pth
│   │   └── tokenizer.json
│   ├── evaluate.py
│   ├── model.py
│   ├── test.py
│   └── train.py
├── main.py
├── pyproject.toml
└── uv.lock
```
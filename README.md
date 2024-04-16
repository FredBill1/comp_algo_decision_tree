# comp_algo_decision_tree

Decision Trees for Comparison-based Algorithms

## Setup Environment

Created with python 3.11

```bash
pip install -r requirements.txt
```

## Run

- Using `waitress`

```bash
pip install waitress
```

```bash
waitress-serve --listen=127.0.0.1:8000 --threads 12 'comp_algo_decision_tree.app:server'
```

- Using `gunicorn` (UNIX only)

```bash
pip install gunicorn
```

```bash
gunicorn -b 127.0.0.1:8000 --threads 12 'comp_algo_decision_tree.app:server'
```

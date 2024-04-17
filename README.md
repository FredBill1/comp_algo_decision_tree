# comp_algo_decision_tree

Decision tree visualization tool for comparison-based algorithms.

## Setup Environment

Requires `python>=3.11`.

```bash
pip install -r requirements.txt
```

> Note that `atomics==1.0.2` does not ship with the correct binary for Apple arm64 chips (M1, M2, etc., as per [this issue](https://github.com/doodspav/atomics/issues/5#issue-1592191416)). You may need to build the library from source manually, as documented [here](https://github.com/doodspav/atomics/blob/v1.0.2/README.md#building).

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

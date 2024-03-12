# GRiD: GPT Reddit Dataset

This repository contains the offical code and data for the paper "GPT-generated Text Detection: Benchmark Dataset and
Tensor-based Detection Method" published in the WWW 2024 short paper track.

## Dataset

The Reddit dataset is available in [`datasets/reddit_datasets`](datasets/reddit_datasets/), where the filtered and unfiltered variants of the dataset can be found.

Each of the files is a Comma Separated Values (CSV) file with two columns:

- `Data`: The text of the snippet.
- `Label`: The label of the snippet, where `0` indicates that the snippet is human-written and `1` indicates that the snippet is GPT-generated.

## Running GpTen

The code for our method, GpTen, is available in the [`pipeline.py`](/pipeline.py) script. The required Python packages and versions can be found in the [`requirements.txt`](/requirements.txt) file.

## Credits

Some data in the `datasets` directory is sourced from: https://github.com/kaize0409/GCN_AnomalyDetection.

## Citation

If you use this code or data, please cite the following paper:

```latex
TODO
```

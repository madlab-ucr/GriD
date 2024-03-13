# GRiD: GPT Reddit Dataset

This repository contains the official code and data for the paper "GPT-generated Text Detection: Benchmark Dataset and Tensor-based Detection Method" published in the WWW 2024 short paper track.

## Dataset

The Reddit dataset is available in [`reddit_datasets`](/reddit_datasets/), where the filtered and unfiltered variants of the dataset can be found.

Each of the files is a Comma Separated Values (CSV) file with two columns:

- `Data`: The text of the snippet.
- `Label`: The label of the snippet, where `0` indicates that the snippet is human-written and `1` indicates that the snippet is GPT-generated.

The reddit data scrapers are located in the [`scrapers`](/scrapers) folder. The dataset construction and cleaning process is facilitated through the [`Load_Full_Dataset.ipynb`](/Load_Full_Dataset.ipynb) notebook.

Please refer to the paper for more details on how the dataset was collected and processed.

## Running GpTen

The code for our method, GpTen, is primarily available in the [`pipeline.py`](/pipeline.py) script. If the input tensor is too large to be processed by the [`pipeline.py`](/pipeline.py) script, you can use the [`decomp.m`](/decomp.m) MATLAB script to decompose the tensor into its factors. After decomposition, you can use the [`norms.m`](/norms.m) MATLAB script to calculate the reconstruction error. This allows you to handle larger datasets/tensors that might otherwise be un-manageable by the python pipeline script.

The required Python packages and versions can be found in the [`requirements.txt`](/requirements.txt) file. For the MATLAB scripts, `tensor_toolbox` is required. Please note that MATLAB 2021b or higher is required to run the `decomp.m` and `norms.m` scripts.

## Credits

The data itself is sourced from Reddit and from OpenAI's GPT3.5.

## Citation

If you use this code or data, please cite the following paper:

```latex
TODO
```

# difFOCI

Code to replicate the experimental results from [A differentiable rank-based objective for better feature learning](https://openreview.net/pdf?id=KiN7g8mf9N).

## Replicating the main results

### Installing dependencies

To set up a working environment for this repository, you can create a conda environment using the following commands:

```bash
conda env create -f environment.yaml
conda activate diffoci
```	

If conda is not available, please install the dependencies listed in the requirements.txt file.

### Running all experiments

The code for reproducing the experiments and obtaining the numbers in Tables 1-4 can be found in the corresponding ipynb notebooks:
* Table_1.ipynb -> Synthetic and Toy experiments
* Table_2.ipynb -> Feature selection and Dimensionality Reduction
* Table_3.ipynb -> Spurious correlations and Domain Adaptation
* Table_4.ipynb -> Fairness


### Download, extract and generate metadata for Waterbirds dataset

This script downloads, extracts and formats the datasets metadata so that it works with the rest of the code out of the box.

```bash
python setup_datasets.py --download --data_path data
```

### Launch jobs for Waterbirds

To reproduce the Waterbirds experiments on a SLURM cluster :

```bash
python train.py --data_path data --output_dir main_sweep --partition <slurm_partition>
```


## License

This source code is released under the CC-BY license, included [here](LICENSE).


## Citations:

1. [Chatterjee (2020). "A new coefficient of correlation"](https://arxiv.org/abs/1909.10140)
2. [Azadkia and Chatterjee (2021). "A simple measure of conditional dependence"](https://arxiv.org/abs/1910.12327)


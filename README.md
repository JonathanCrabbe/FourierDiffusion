# Time Series Diffusion in the Frequency Domain

This repository implements time series diffusion in the frequency domain.
For more details, please read our paper: [Time Series Diffusion in the Frequency Domain](https://arxiv.org/abs/2402.05933).
 
# 1. Install


From repository:
1. Clone the repository.
2. Create and activate a new environment with conda (with `Python 3.10` or newer).

```shell
conda env create -n fdiff python=3.10
conda activate fdiff
```
3. Install the requirement.
```shell
pip install freqdiff
```

4. If you intend to train models, make sure that wandb is correctly configured on your machine by following [this guide](https://docs.wandb.ai/quickstart). 
5. Some of the datasets are automatically downloaded by our scripts via kaggle API. Make sure to create a kaggle token as explained [here](https://towardsdatascience.com/downloading-datasets-from-kaggle-for-your-ml-project-b9120d405ea4).

When the packages are installed, you are ready to train diffusion models!

# 2. Use

## 2.1 Train
In order to train models, you can simply run the following command:

```shell
python cmd/train.py 
```

By default, this command will train a score model in the time domain with the `ecg` dataset. In order to modify this behaviour, you can use [hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/). The following hyperparameters can be modified to retrain all the models appearing in the paper:

| Hyperparameter | Description | Values |
|----------------|-------------|---------------|
|fourier_transform | Whether or not to train a diffusion model in the frequency domain. | true, false |
| datamodule | Name of the dataset to use. | ecg, mimiciii, nasa, nasdaq, usdroughts|
| datamodule.subdataset | For the NASA dataset only. Selects between the charge and discharge subsets. | charge, discharge |
| datamodule.smoother_width | For the ECG dataset only. Width of the Gaussian kernel smoother applied in the frequency domain. | $\mathbb{R}^+$
| score_model | The backbone to use for the score model. | default, lstm |

At the end of training, your model is stored in the `lightning_logs` directory, in a folder named after the current `run_id`. You can find the `run_id` in the logs of the training and in the [wandb dashboard](https://wandb.ai/) if you have correctly configured wandb.

## 2.2 Sample

In order to sample from a trained model, you can simply run the following command:

```shell
python cmd/sample.py model_id=XYZ
```
    
where `XYZ` is the `run_id` of the model you want to sample from. At the end of sampling, the samples are stored in the `lightning_logs` directory, in a folder named after the current `run_id`. 

One can then reproduce the plots in the paper by including the  `run_id` to the `run_list` list appearing in [this notebook](notebooks/results.ipynb) and running all cells.

# 3. Contribute

If you wish to contribute, please make sure that your code is compliant with our tests and coding conventions. To do so, you should install the required testing packages with:

```shell
pip install freqdiff[test]
```

Then, you can run the tests with:

```shell
pytest
```

Before any commit, please make sure that your staged code is compliant with our coding conventions by running:

```shell
pre-commit
```

# 4. Cite us
If you use this code, please acknowledge our work by citing

```
@misc{crabbé2024time,
      title={Time Series Diffusion in the Frequency Domain}, 
      author={Jonathan Crabbé and Nicolas Huynh and Jan Stanczuk and Mihaela van der Schaar},
      year={2024},
      eprint={2402.05933},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

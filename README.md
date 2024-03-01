# Accelerating Greedy Coordinate Gradient via Probe Sampling 

This repository contains code for the paper "Accelerating Greedy Coordinate Gradient via Probe Sampling". Below is the workflow of probe sampling.

![./](./probe-sampling.png)



## Installation

Our codebase comes from the paper [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043) [(github)](https://github.com/llm-attacks/llm-attacks). The package can be installed by running the following command at the root of this repository: 

```
pip install -e .
```

## Parameters

Beyond the parameters of the original GCG, probe sampling necessitates the specification of two additional key parameters: **probe set size** and **filtered set size**, referred to as `probe_set` and `filtered_set`, respectively. These can be configured within `./experiments/launch_scripts/` as follows.

```sh
--config.probe_set=xx \
--config.filtered_set=xx
```

## Models

 

### Target Models

Address of the target model is set in `experiments/configs/`, where `/DIR` is the address that you store the model.

```sh
  config.model_paths = [
      "/DIR/Llama2-7b-chat",
  ]
  config.tokenizer_paths = [
      "/DIR/Llama2-7b-chat",
  ]
```

### Draft Model

The address of the draft model is specified in `experiments/main.py`; replace `/DIR` with the directory path where the model is stored. Additionally, the GPU on which the model is placed is determined by the setting `params_small.devices`.

```python
params_small.model_paths = ["/DIR/GPT2"]
params_small.tokenizer_paths = ["/DIR/GPT2"]
params_small.devices = ['cuda:0']
```

## Experiments

* To execute specific experiments involving harmful behaviors and strings, execute the code below within the `experiments` directory. Note that replacing `vicuna` with `llama2` and substituting `behaviors` with `strings` will transition to alternative experimental configurations:

  ```sh
  cd launch_scripts
  bash run_gcg_individual.sh vicuna behaviors
  ```

- To perform multiple behaviors experiments, run the following code inside `experiments`:

  ```sh
  cd launch_scripts
  bash run_gcg_multiple.sh vicuna
  ```


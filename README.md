Research code accompanying the article **“Attention-Based Surface Representation Learning for Robot State Prediction and
Open-Ended Surface Classification.”** The project explores attention-based models for two tasks: predicting
future robot states from recent observations and classifying the surface the robot is currently traversing.

The latest model is the **Prototypical Transformer**, implemented
in [`src/models/prototypical_transformer.py`](src/models/prototypical_transformer.py).

## Method overview

The Prototypical Transformer encodes a history of robot observations and prepends a learnable surface token to the
sequence. The encoded surface representation is compared with learnable class prototypes using cosine similarity. In
parallel, learned future-query tokens attend to the encoded history and are used to predict future robot states.

## Requirements

- Python 3.13 or newer
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management

## Installation

Clone the repository and install its dependencies:

```bash
git clone https://github.com/okushnarev/transformer-cls.git
cd transformer-cls
uv sync
```

The default trainer configuration uses a TensorBoard logger. Install the optional logging dependency as well if you plan
to use that logger:

```bash
uv sync --extra logging
```

Run project commands through `uv run` to use the environment managed by `uv`.

## Datasets

The datasets are distributed separately from this repository:

- **BorealTC:** [norlab-ulaval/BorealTC](https://github.com/norlab-ulaval/BorealTC)
- **Belyaev–Kushnarev:** [https://doi.org/10.5281/zenodo.23291838](https://doi.org/10.5281/zenodo.23291838)

The default `BorealTCFutureDataModule` reads a prepared CSV file from `datasets/boreal_imu.csv`. Download and prepare
the data before training, and place the CSV at that path. The data module derives future regression targets from the
configured feature columns and uses the `surf` column for surface classification.

## Configuration

The project uses PyTorch Lightning's CLI and YAML configuration files.

Example of configs:

| Purpose                                                             | Configuration                                                                                |
|---------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Trainer defaults, logging, callbacks, precision, and epoch settings | [`configs/trainer/default.yaml`](configs/trainer/default.yaml)                               |
| Prototypical Transformer model and loss settings                    | [`configs/model/prototypical_transformer.yaml`](configs/model/prototypical_transformer.yaml) |
| BorealTC future-prediction datamodule                               | [`configs/data/borealtc_future.yaml`](configs/data/borealtc_future.yaml)                     |

The CLI entry point is [`scripts/cli.py`](scripts/cli.py). It combines the trainer, model, and datamodule
configurations. These YAML files provide shared baseline settings, while experiment-specific parameters are set through
command-line arguments.

A direct training invocation with the three baseline configs looks like this:

```bash
uv run scripts/cli.py fit \
  --seed_everything 69 \
  --trainer configs/trainer/default.yaml \
  --data configs/data/borealtc_future.yaml \
  --model configs/model/prototypical_transformer.yaml
```

Inspect the available arguments with:

```bash
uv run scripts/cli.py --help
uv run scripts/cli.py fit --help
```

## Experiment launch script

Experiment-specific shell scripts are not included in the repository. The following example is based on the launch
script used for an experiment. Copy it into a local file named `exp01_proto_transf_opt.sh`, for example, into repository
root or any suitable folder (like `scripts/experiments`).

```bash
#!/usr/bin/env bash
set -euo pipefail

# Help
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: ${0##*/} <command> [extra_args...]" >&2
  echo "Example: ./${0##*/} predict" >&2
  exit 1
fi

# Variables
subdir="borealtc"
trainer="default.yaml"
data="borealtc_future.yaml"
model="prototypical_transformer.yaml"

extra_args=(
  "--data.init_args.segment_size" "200"
  "--data.init_args.sequence_length" "103"

  "--model.init_args.in_dim" "11"
  "--model.init_args.output_dim" "11"
  "--model.init_args.n_classes" "5"

  "--model.init_args.in_mlp_hidden_dims" "[128]"
  "--model.init_args.out_mlp_hidden_dims" "[128]"
  "--model.init_args.d_model" "128"
  "--model.init_args.dim_feedforward" "32"

  "--model.init_args.lr_patience" "20"
  "--trainer.check_val_every_n_epoch" "5"
)
seed=69

filename=${0##*/}
exp_name=${filename%.*}
cmd=$1

if [[ "$cmd" = "predict" ]]; then
  extra_args+=(
    "--trainer.logger.init_args.overwrite" "false"
    "--ckpt_path" "experiments/$subdir/$exp_name/checkpoints/best.ckpt"
  )
fi

# Script
uv run scripts/cli.py "$cmd" \
  --seed_everything "$seed" \
  --trainer configs/trainer/"$trainer" \
  --data configs/data/"$data" \
  --model configs/model/"$model" \
  --trainer.logger.init_args.name "$subdir" \
  --trainer.logger.init_args.version "$exp_name" \
  --data.init_args.batch_size 100 \
  --data.init_args.num_workers 2 \
  ${extra_args[@]+"${extra_args[@]}"} \
  "${@:2}"
```

### Running an experiment

Run the script from the repository root, since its CLI and configuration paths are relative to the current working
directory.

To train a model:

```bash
bash ./exp01_proto_transf_opt.sh fit
```

To run prediction using the corresponding trained checkpoint:

```bash
bash ./exp01_proto_transf_opt.sh predict
```

The script also accepts additional Lightning CLI arguments after the command.

### Configuring the experiment

The script combines the baseline YAML configurations with experiment-specific parameters.

**Configuration files.** The `trainer`, `data`, and `model` variables select the YAML files
under `configs/trainer/`, `configs/data/`, and `configs/model/`. Changing these variables allows the same launch
structure to be reused with different baseline configurations.

**Experiment parameters.** The `extra_args` array contains the parameters specific to this experiment:

- `segment_size` and `sequence_length` configure the data module.
- `in_dim`, `output_dim`, `n_classes`, `d_model`, and `dim_feedforward` configure the model dimensions and number of
  classes.
- `in_mlp_hidden_dims` and `out_mlp_hidden_dims` configure the hidden dimensions of the input and output MLPs.
- `lr_patience` and `check_val_every_n_epoch` configure the learning-rate scheduler patience and validation frequency.

The script also sets the random seed to `69`, the batch size to `100`, and the number of dataloader workers to `2`.

**Experiment naming.** The script derives the experiment name from its own filename. For
example, `exp01_proto_transf_opt.sh` produces the experiment name `exp01_proto_transf_opt`. The variables `subdir`
and `exp_name` are passed to the logger as its name and version, respectively.

This means that the script filename is part of the experiment configuration. Keep it unchanged when you want to refer to
the same experiment in subsequent commands.

**Command-line overrides.** To change a parameter for one run without editing the script, append the corresponding CLI
arguments after the command. For example:

```bash
bash ./exp01_proto_transf_opt.sh fit --trainer.max_epochs 100
```

These additional arguments are forwarded to Lightning CLI after the arguments defined in the script, allowing
command-line overrides without changing the shared YAML configurations.

**Prediction and checkpoints.** When the command is `predict`, the script adds two arguments: it disables logger
overwriting and sets the checkpoint path to:

```text
experiments/$subdir/$exp_name/checkpoints/best.ckpt
```

Because `exp_name` is derived from the script filename, the prediction command expects the corresponding checkpoint to
exist under that experiment name. Use the same script filename for training and prediction, or edit the checkpoint path
to point to the intended experiment.

This approach keeps the common configuration in YAML files, experiment-specific settings in the launch script, and
occasional changes on the command line.
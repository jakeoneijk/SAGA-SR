# SAGA-SR

Text-conditioned latent flow-matching model for audio super-resolution.

This is the implementation of [SAGA-SR](https://arxiv.org/abs/2509.24924).

> ⚠️ **This code is under construction.** The repository currently contains the
> preprocessing and training code only, and may still contain errors.

## TODO

- [ ] Update `requirements.txt`
- [ ] Upload pretrained model weights
- [ ] Add inference / evaluation code
## Installation

This project is built on top of [TorchJaekwon](https://github.com/jakeoneijk/TorchJaekwon),
which must be installed first.

```bash
# Clone TorchJaekwon
git clone https://github.com/jakeoneijk/TorchJaekwon
cd TorchJaekwon

# Install TorchJaekwon
source install_torchjk.sh
```

## Usage

The entry point follows the TorchJaekwon convention. Pass the config file with
`--config_path` and the stage with `--stage`:

```bash
python main.py --config_path CONFIG_PATH --stage STAGE_NAME
# STAGE choices: ['preprocess', 'train', 'inference', 'evaluate']
```

Config files live under `config/`. The reference config is `config/saga_sr.yaml`.

### Preprocess

```bash
python main.py --config_path config/saga_sr.yaml --stage preprocess
```

### Train

```bash
python main.py --config_path config/saga_sr.yaml --stage train
```

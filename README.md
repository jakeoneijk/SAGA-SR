# SAGA-SR

Text-conditioned latent flow-matching model for audio super-resolution.

> ⚠️ **This code is under construction.** The repository currently contains the
> preprocessing and training code only. APIs, configs, and file layout are
> subject to change.

## TODO

- [ ] Finalize and publish `requirements.txt` (pinned dependency versions)
- [ ] Upload pretrained model weights
- [ ] Add inference / evaluation code
- [ ] Add usage examples and documentation

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

See the [TorchJaekwon README](https://github.com/jakeoneijk/TorchJaekwon) for
more details.

### Other dependencies

In addition to TorchJaekwon, this code depends on (a pinned `requirements.txt`
is still TODO):

- `stable_audio_tools` (Stable Audio Open)
- `transformers`
- `ema_pytorch`
- `einops`
- `librosa`
- `scipy`

## Usage

The entry point follows the TorchJaekwon convention:

```bash
python main.py --stage STAGE_NAME
# STAGE choices: ['preprocess', 'train', 'inference', 'evaluate']
```

Config files live under `config/`. The reference config is:

```
config/250613_text_sr/250822_prompt_sr.yaml
```

### Preprocess

```bash
python main.py --stage preprocess
```

### Train

```bash
python main.py --stage train
```

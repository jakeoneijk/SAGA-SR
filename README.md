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

### Vendored code

A minimal subset of [Stable Audio Open's](https://github.com/Stability-AI/stable-audio-tools)
`stable_audio_tools` package is vendored under `stable_audio_tools/` so it does
not need to be installed separately. Only one function is actually used by this
code:

```python
from stable_audio_tools.models.conditioners import create_multi_conditioner_from_conditioning_config
```

It is called in `LatentSRTextFlow.__init__` to build the conditioner (T5 prompt
encoder + numeric cutoff-ratio conditioners) from the `conditioning_config`
block in the YAML config.

### Other dependencies

In addition to TorchJaekwon, this code depends on (a pinned `requirements.txt`
is still TODO):

- `transformers`
- `ema_pytorch`
- `einops`
- `librosa`
- `scipy`

The vendored `stable_audio_tools` subset additionally requires:

- `descript-audio-codec` (provides `dac`)
- `einops_exts`
- `packaging`
- `safetensors`
- `torchaudio`

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

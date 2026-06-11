import os
import torch
from tqdm import tqdm
from torch_jaekwon.util import util_data, util_torch, util_audio

from util.util_audio_lowpass_filter import UtilAudioLowPassFilterNVSR
from train.trainer.flow_sr_trainer import FlowSRTrainer

class FlowSRTextTrainer(FlowSRTrainer):
    def log_media(self) -> None:
        cutoff_freq = self.log_input_cutoff_freq
        filter_name = "butter"
        filter_order =  4
        lowpass_filter = UtilAudioLowPassFilterNVSR()
        self.model.use_lfr_postprocess = False
        index = 0
        for data_path in tqdm(self.log_test_data_list,desc='plot audio'):
            data_path = data_path['file_abspath']
            audio, sr = util_audio.read(data_path, mono=self.mono, sample_rate=self.sample_rate)
            audio = audio.squeeze()
            assert sr == self.sample_rate, f"sr is {sr}"
            lr_audio = lowpass_filter.lowpass(audio, self.sample_rate, filter_name=filter_name, filter_order=filter_order, cutoff_freq=cutoff_freq)

            propmt_list = [prompt.replace('\n', '')for prompt in util_data.txt_load(f"{os.path.splitext(data_path)[0]}.txt")]

            for prompt in propmt_list:
                pred_hr_audio = self.model.infer(
                    cond = {
                        'audio': torch.from_numpy(lr_audio).unsqueeze(0).float().to(self.device),
                        'prompt': prompt,
                        'output_cutoff_ratio': self.log_media_output_cutoff_ratio
                    }
                )

                audio_dict = {
                    'lr': lr_audio, 
                    'hr': audio,
                    prompt: util_torch.to_np(pred_hr_audio[0])
                }
                
                for key in audio_dict:
                    if len(audio_dict[key].shape) == 2:
                        audio_dict[key] = audio_dict[key][0]
                    if isinstance(audio_dict[key], torch.Tensor):
                        audio_dict[key] = util_torch.to_np(audio_dict[key])
                    audio_dict[key] = audio_dict[key][:self.sample_rate * 5]

                self.logger.plot_audio(name = f'testcase{index:02}', audio_dict = audio_dict, global_step=self.global_step, sample_rate=self.sample_rate, log_local=True)
                index += 1

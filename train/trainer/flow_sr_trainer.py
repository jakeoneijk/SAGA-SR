import torch
from tqdm import tqdm

from torch_jaekwon.train.trainer.trainer import Trainer
from torch_jaekwon.util import util_audio, util_data, util_torch

from util.util_audio_lowpass_filter import UtilAudioLowPassFilterNVSR

class FlowSRTrainer(Trainer):
    def __init__(
        self,
        segment_sec:float,
        sample_rate:int,
        mono:bool,
        log_input_cutoff_freq: float,
        log_media_output_cutoff_ratio:float = 1,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.segment_sec:int = segment_sec
        self.sample_rate:int = sample_rate
        self.mono:bool = mono
        root_data_dir:str = 'artifacts/input/train_logging'
        self.log_test_data_list = util_data.walk(root_data_dir)
        self.log_input_cutoff_freq:int = log_input_cutoff_freq
        self.log_media_output_cutoff_ratio = log_media_output_cutoff_ratio
    
    def run_step(self,data,metric,train_state):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].float().to(self.device)
        loss = self.model(data['target_audio'], cond = data)
        metric = self.update_metric(metric, {'loss': loss}, data['target_audio'].shape[0])
        return loss, metric
    
    def log_media(self) -> None:
        cutoff_freq = self.log_input_cutoff_freq
        filter_name = "butter"
        filter_order =  4
        lowpass_filter = UtilAudioLowPassFilterNVSR()
        self.model.use_lfr_postprocess = False
        for i,data_path in tqdm(enumerate(self.log_test_data_list),desc='plot audio'):
            data_path = data_path['file_path']
            audio, sr = util_audio.read(data_path, mono=self.mono, sample_rate=self.sample_rate)
            audio = audio.squeeze()
            assert sr == self.sample_rate, f"sr is {sr}"
            lr_audio = lowpass_filter.lowpass(audio, self.sample_rate, filter_name=filter_name, filter_order=filter_order, cutoff_freq=cutoff_freq)

            pred_hr_audio = self.model.infer(
                cond = {
                    'audio': torch.from_numpy(lr_audio).unsqueeze(0).float().to(self.device),
                    'output_cutoff_ratio': self.log_media_output_cutoff_ratio
                }
            )

            audio_dict = {
                'lr': lr_audio, 
                'hr': audio,
                'pred': util_torch.to_np(pred_hr_audio)
            }
            
            for key in audio_dict:
                if len(audio_dict[key].shape) == 2:
                    audio_dict[key] = audio_dict[key][0]
                if isinstance(audio_dict[key], torch.Tensor):
                    audio_dict[key] = util_torch.to_np(audio_dict[key])
                audio_dict[key] = audio_dict[key][:self.sample_rate * 5]

            self.logger.plot_audio(name = f'testcase{i}', audio_dict = audio_dict, global_step=self.global_step, sample_rate=self.sample_rate, log_local=True)
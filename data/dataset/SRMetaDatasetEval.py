import os
import torch

from torch_jaekwon.util import util_data, util_audio

from util.util_audio_lowpass_filter import UtilAudioLowPassFilterNVSR

class SRMetaDatasetEval(torch.utils.data.Dataset):
    def __init__(self,
                 root_data_dir:str = '/home/jakeoneijk/220101_data/audiosr/test/ResampleTo48k',
                 sample_rate:int = 48000,
                 mono:bool = False,
                 segment_sec:float = 10) -> None:
        data_dir_list = [f'{root_data_dir}/{data_name}' for data_name in os.listdir(root_data_dir) if os.path.isdir(f'{root_data_dir}/{data_name}')]
        self.data_list = list()
        for data_dir in data_dir_list:
            self.data_list += [f'{data_dir}/{file_name}' for file_name in os.listdir(data_dir) if os.path.splitext(file_name)[-1] == '.wav'][:10]
        
        self.cutoff_freq = 4000
        self.filter_name = "cheby"
        self.filter_order =  8
        self.lowpass_filter = UtilAudioLowPassFilterNVSR()
        self.segment_sec:float = segment_sec
        self.segment_length:int = round(segment_sec * sample_rate)
        self.sample_rate = sample_rate
        self.mono = mono
        self.origin_sample_rate = 48000

    def __getitem__(self, index):
        audio_path:str = self.data_list[index]
        audio, sr = util_audio.read(
            audio_path=audio_path,
            sample_rate=self.sample_rate,
            mono=self.mono,
            sample_length=self.segment_length,
            start = 0,
            end = self.segment_sec,
            segment_type = 'time',
            origin_sample_rate = self.origin_sample_rate
        )
        audio = audio.squeeze()
        audio = util_data.fix_length(audio,self.segment_length)
        lr_audio = self.lowpass_filter.lowpass(audio, sr, filter_name=self.filter_name, filter_order=self.filter_order, cutoff_freq=self.cutoff_freq)
           
        return {
            "audio": lr_audio, 
            "target_audio": audio, 
            'cutoff_freq': self.cutoff_freq, 
            "prompt": "",
            "seconds_start": 0,
            "seconds_total": 5,
            'data_type': 'speech',
            "input_cutoff_ratio": self.cutoff_freq / (self.sample_rate / 2),
            "output_cutoff_ratio": 1.0
        }
    
    def __len__(self):
        return len(self.data_list)
        
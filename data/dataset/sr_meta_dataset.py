from typing import Dict
from tqdm import tqdm

import os
import random
import numpy as np

from torch_jaekwon.util import util_data, util_audio, util
from torch_jaekwon.data.dataset.balanced_multi_dataset import BalancedMultiDataset

from util.util_audio_lowpass_filter import UtilAudioLowPassFilterNVSR
from util.util_audiosr import UtilAudioSR

class SRMetaDataset(BalancedMultiDataset):
    def __init__(
        self,
        meta_dir_path:str,
        data_name_list: list,
        data_config_name:str,
        subset_name:str,
        input_cut_off_freq_range:tuple = [2000, 16000],
        output_cut_off_freq_range:tuple = None,
        target_cutoff_ratio_threshold:float = None,
        calc_input_cutoff_ratio:bool = True,
        sample_rate:int = 48000,
        segment_sec:float = 5.12,
        mono:bool = True,
        data_set_type_list:list = ['music','speech','general'],
        use_prompt:bool = False,
        **kwargs,
    ) -> None:
        self.data_config_name:str = data_config_name
        self.meta_dir_path:str = f'{meta_dir_path}/{data_config_name}'
        self.data_name_list:list = data_name_list
        self.subset_name:str = subset_name if subset_name is not None else 'total'
        self.data_set_type_list:list = data_set_type_list
        self.dataset_type_map:dict = {
            'Freesound48k': 'general',
            'MedleyDB-V1': 'music',
            'MedleyDB-V2': 'music',
            'MoisesDB': 'music',
            'MUSDB18-HQ': 'music',
            'OpenSLR': 'speech'
        }
        print(f'dataset type list: {data_set_type_list}')
        
        self.lowpass_filter = UtilAudioLowPassFilterNVSR()
        self.filter_list = ["cheby","butter","bessel","ellip"]
        self.filter_order_list =  [2,3,4,5,6,7,8,9,10] #order of the lowpass filter is randomly selected between 2 and 10
        
        self.input_cut_off_freq_range = input_cut_off_freq_range
        self.input_max_cut_off = 0.94
        self.output_cut_off_freq_range = output_cut_off_freq_range
        self.target_cutoff_ratio_threshold:float = target_cutoff_ratio_threshold
        self.calc_input_cutoff_ratio:bool = calc_input_cutoff_ratio

        self.segment_sec:float = segment_sec
        self.sample_rate = sample_rate
        self.mono = mono
        self.use_prompt:bool = use_prompt

        super().__init__(**kwargs)
        util.log(f"Total train data num:{sum([len(self.data_list_dict[data_name]) for data_name in self.data_list_dict])}")
    
    def init_data_list_dict(self) -> Dict[str,list]: # {data_type1: List, data_type2: List}
        subset_data_list_path:str = f'{self.meta_dir_path}/subset/{self.subset_name}.pkl'
        if os.path.exists(subset_data_list_path):
            data_list_dict =  util_data.pickle_load(subset_data_list_path)
            return data_list_dict
        
        data_list_dict:dict = {data_set_type: list() for data_set_type in self.data_set_type_list}
        for data_set_name in tqdm(self.data_name_list, desc='Get Meta Data Param'):
            meta_list_of_data = util_data.walk(f'{self.meta_dir_path}/{data_set_name}/meta',ext=['.pkl'])
            if self.target_cutoff_ratio_threshold is not None:
                for meta in tqdm(meta_list_of_data, desc=f'Sanity check [{data_set_name}]'):
                    audio_meta_data:dict = util_data.pickle_load(meta['file_path'])       
                    #min_cutoff_ratio = self.output_cut_off_freq_range[1] / (self.sample_rate / 2)
                    if audio_meta_data['cutoff_ratio'] >= self.target_cutoff_ratio_threshold:
                        data_list_dict[self.dataset_type_map[data_set_name]].append({**meta, 'data_type': self.dataset_type_map[data_set_name]})
            else:
                data_list_dict[self.dataset_type_map[data_set_name]] += [{**meta, 'data_type': self.dataset_type_map[data_set_name]} for meta in meta_list_of_data]
                                                                         
        util_data.pickle_save(subset_data_list_path, data_list_dict)
        return data_list_dict

    def read_data(self,meta_data):
        #print(meta_data['file_path'])
        audio_meta_data:dict = util_data.pickle_load(meta_data['file_path'])
        original_sr = audio_meta_data['sample_rate']
        audio, sr = util_audio.read(
            audio_path=audio_meta_data['file_path'],
            sample_rate=self.sample_rate,
            mono=self.mono,
            sample_length= round(self.segment_sec * self.sample_rate),
            start = audio_meta_data['start_sec'],
            end = audio_meta_data['end_sec'],
            segment_type = 'time',
            origin_sample_rate = original_sr
        )
        audio = audio.squeeze()
        assert sr == self.sample_rate, f"sr is {sr}"

        filter_name = random.choice(self.filter_list)
        filter_order = random.choice(self.filter_order_list)

        output_cufoff_ratio = audio_meta_data['cutoff_ratio']
        if self.output_cut_off_freq_range is not None:
            output_cutoff_freq = random.randint(*self.output_cut_off_freq_range)
            audio = self.lowpass_filter.lowpass(audio.numpy(), self.sample_rate, filter_name=filter_name, filter_order=filter_order, cutoff_freq=output_cutoff_freq)
            output_cufoff_ratio = output_cutoff_freq / (self.sample_rate / 2)

        cutoff_freq = random.randint(*self.input_cut_off_freq_range)
        cutoff_freq = max(self.input_cut_off_freq_range[0], min(cutoff_freq, int(audio_meta_data['cutoff_ratio'] * (self.sample_rate / 2)) * self.input_max_cut_off))
        lr_audio = self.lowpass_filter.lowpass(audio.numpy(), self.sample_rate, filter_name=filter_name, filter_order=filter_order, cutoff_freq=cutoff_freq)
        input_cutoff_ratio = cutoff_freq / (self.sample_rate / 2)
        if self.calc_input_cutoff_ratio:
            input_cutoff_ratio = UtilAudioSR.get_cutoff_ratio(lr_audio)
            #if input_cutoff_ratio == 0.0: print('')
        
        result_dict = {
            "audio": lr_audio, 
            "target_audio": audio, 
            "input_cutoff_ratio": input_cutoff_ratio,
            "output_cutoff_ratio": output_cufoff_ratio,
            'data_type': meta_data['data_type'],
        }
        
        if self.use_prompt:
            result_dict['prompt'] = util_data.pickle_load(meta_data['file_path'].replace(self.data_config_name, self.data_config_name + '_text'))

        '''
        from TorchJaekwon.Util import UtilTorch
        debug1 = UtilAudioSR.get_cutoff_index_np(lr_audio[0], nfft=2048) / (2048 // 2 + 1)
        debug2 = UtilAudioSR.get_cutoff_index_np(audio[0], nfft=2048) / (2048 // 2 + 1)'
        util_audio.write(f"./{result_dict['prompt']}.wav", result_dict['target_audio'], 44100)
        '''
        return result_dict
    
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

import os
import torch
import numpy as np
from torch_jaekwon.data.preprocess.preprocessor import Preprocessor
from torch_jaekwon.util import util_data, util_audio, util_torch

class SRTextPreprocessor(Preprocessor):
    def __init__(
        self,
        preprocessed_dir_path:str,
        data_config_name:str,
        text_tag:str,
        subset_name:str = None,
        batch_size:int = 32,
        audio_data_dir_path:str = None,
        **kwargs
    ) -> None:
        self.subset_name = subset_name
        self.data_config_name = data_config_name
        self.data_dir_path = f'{preprocessed_dir_path}/{data_config_name}' if audio_data_dir_path is None else audio_data_dir_path
        self.output_dir_path = f'{preprocessed_dir_path}/{data_config_name}{text_tag}'
        super().__init__(**kwargs)
        self.num_workers = 1
        self.batch_size = batch_size
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True, device_map= f'{self.device.type}')
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)
        self.prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
    
    def get_output_dir(self) -> str:
        return self.output_dir_path

    def get_meta_data_param(self) -> list:
        if self.subset_name is None:
            all_data_path = f'{self.data_dir_path}/subset/all_data_meta.pkl'
            if not os.path.exists(all_data_path):
                meta_data_list =  util_data.walk(self.data_dir_path, ext='.pkl')
                meta_data_list = [meta_data for meta_data in meta_data_list if 'subset' not in meta_data['file_path']]
                util_data.pickle_save(all_data_path, meta_data_list)
            else:
                meta_data_list = util_data.pickle_load(all_data_path)
        else:
            meta_data_dict = util_data.pickle_load(f'{self.data_dir_path}/subset/{self.subset_name}.pkl')
            meta_data_list = list()
            for _, value in meta_data_dict.items():
                meta_data_list += value

        #from random import shuffle
        #shuffle(meta_data_list)
        meta_data_list = [meta_data for meta_data in meta_data_list if not os.path.exists(self.get_preprocessed_data_path(meta_data))]
        meta_data_list = [meta_data_list[i:i + self.batch_size] for i in range(0, len(meta_data_list), self.batch_size)]
        return meta_data_list
    
    def preprocess_one_data(self, param_list:dict) -> None:
        '''
        ex) (subset, file_name) = param
        '''
        audios = list()
        for param in param_list:
            audios.append(self.get_audio(param))
        
        with torch.no_grad():
            inputs = self.processor(text=[self.prompt for i in range(len(audios))], audios=audios, return_tensors="pt", padding=True, sampling_rate=16000)
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            inputs['input_ids'] = inputs['input_ids'].to(self.device)
            inputs.input_ids = inputs.input_ids.to(self.device)

            generated_ids = self.model.generate(**inputs, max_length=256)
            generated_ids = generated_ids[:, inputs.input_ids.size(1):]
            response_list = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i, (response, meta_data) in enumerate(zip(response_list, param_list)):
            response = response.lstrip()
            output_path = self.get_preprocessed_data_path(meta_data)
            util_data.pickle_save(output_path, response)
            #util_audio.write(f'./artifacts/temp/{i:02}_{response}.wav', audios[i], sample_rate=16000)
        #print('')
    
    def get_preprocessed_data_path(self, meta_data) -> str:
        return self.output_dir_path + meta_data['file_path'].split(self.data_config_name)[-1]
    
    def get_audio(self, meta_data:dict) -> np.ndarray:
        audio_meta_data = util_data.pickle_load(meta_data['file_path'])
        audio, sr = util_audio.read(
            audio_path=audio_meta_data['file_path'],
            sample_rate=16000,
            mono=True,
            start = audio_meta_data['start_sec'],
            end = audio_meta_data['end_sec'],
            segment_type = 'time',
            origin_sample_rate = audio_meta_data['sample_rate']
        )
        return util_torch.to_np(audio)
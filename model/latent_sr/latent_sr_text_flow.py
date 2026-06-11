from torch_jaekwon.util import util
util.set_sys_path_to_parent_dir(__file__, 2)
from typing import List, Optional, Tuple, Union
from torch import Tensor

import torch
from torch_jaekwon.util import util_data
from torch_jaekwon.model.flow_matching.flow_matching import FlowMatching

from model.conditioner.conditioners import create_multi_conditioner_from_conditioning_config

from util.util_audiosr import UtilAudioSR
from model.latent_sr.autoencoder.autoencoder_pretransform import AutoencoderPretransform

class LatentSRTextFlow(FlowMatching):
    def __init__(
        self, 
        autoencoder_ckpt_path:str = 'artifacts/ckpt/autoencoder.pth', 
        sample_rate:int = 44100,
        audio_seg_sec:float = 1.0,
        cross_attention_cond_name_list = list(),
        global_cond_name_list = list(),
        conditioning_config:List[dict] = list(),
        use_lfr_postprocess:bool = True,
        use_only_text_cond:bool = False,
        **kwargs 
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate:int = sample_rate
        self.audio_length:int=round(audio_seg_sec * sample_rate)
        self.autoencoder_dim:int = 64
        self.autoencoder_downsampling_ratio:int = 2048
        # Autoencoder
        self.autoencoder:AutoencoderPretransform
        self.__dict__["autoencoder"] = AutoencoderPretransform()
        autoencoder_ckpt = torch.load(autoencoder_ckpt_path, map_location='cpu')
        self.autoencoder.load_state_dict(autoencoder_ckpt)
        # Conditioner
        self.cross_attention_cond_name_list:List[str] = cross_attention_cond_name_list
        self.global_cond_name_list:List[str] = global_cond_name_list
        self.conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config)
        # Post process
        self.use_lfr_postprocess:bool = use_lfr_postprocess
        self.use_only_text_cond:bool = use_only_text_cond

        self.device:torch.device = torch.device('cpu')
    
    def to(self, device:torch.device) -> None:
        super().to(device)
        self.autoencoder.to(device)
        self.device = device
        
    def preprocess(
        self,
        x_start:Tensor, 
        cond:dict = None, # {'audio': [batch, 2, self.audio_length], 'text': List[str]}
    ) -> Tuple[Tensor,Tensor]: 
        lr_audio = util_data.fix_length(cond['audio'], self.audio_length)
        lr_audio = util_data.fit_shape_length(lr_audio,3)

        if x_start is not None:
            x_start = util_data.fix_length(x_start, self.audio_length)
            with torch.no_grad():
                z:Tensor = self.autoencoder.encode(x_start)
        else:
            z = None
            if 'input_cutoff_ratio' not in cond:
                cond['input_cutoff_ratio'] = [UtilAudioSR.get_cutoff_ratio(lr_audio[i]) for i in range(lr_audio.shape[0])]

        with torch.no_grad():
            lr_z:Tensor = self.autoencoder.encode(lr_audio)

        model_cond_dict:dict = {"input_concat_cond": lr_z}

        cond_dict = dict()
        for key, conditioner in self.conditioner.conditioners.items():
            if f'{key}_embed' in cond:
                cond_dict[key] = cond[f'{key}_embed']
                continue
            raw_condition = cond[key]
            if isinstance(raw_condition, float) or isinstance(raw_condition, int) or isinstance(raw_condition, str):
                raw_condition = [raw_condition]
            cond_dict[key], condition_mask = conditioner(raw_condition, self.device)
        
        if len(self.global_cond_name_list) > 0:
            global_cond_list = list()
            for key in self.global_cond_name_list:
                global_cond_list.append(cond_dict[key])
            global_cond = torch.cat(global_cond_list, dim=-1)
            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)
            model_cond_dict['global_embed'] = global_cond
        
        if len(self.cross_attention_cond_name_list) > 0:
            # Concatenate all cross-attention inputs over the sequence dimension
            # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
            cross_attention_cond_list = list()
            for key in self.cross_attention_cond_name_list:
                # Add sequence dimension if it's not there
                if len(cond_dict[key].shape) == 2:
                    cond_dict[key] = cond_dict[key].unsqueeze(1)

                cross_attention_cond_list.append(cond_dict[key])

            cross_attention_cond = torch.cat(cross_attention_cond_list, dim=1)
            model_cond_dict['cross_attn_cond'] = cross_attention_cond
        
        additional_data_dict = {'audio': cond['audio']}
        return z, model_cond_dict, additional_data_dict
        
    def postprocess(
        self, 
        x: Tensor, #[batch_size, self.autoencoder_dim, self.audio_length // self.autoencoder_downsampling_ratio]
        additional_data_dict
    ) -> Tensor:
        lr_audio:Tensor = additional_data_dict['audio']
        with torch.no_grad():
            pred_audios:Tensor = self.autoencoder.decode(x)
        pred_audio_list = list()
        for i in range(pred_audios.shape[0]):
            pred_audio = util_data.fix_length(pred_audios[i], lr_audio[i].shape[-1])
            if self.use_lfr_postprocess:
                pred_audio = UtilAudioSR.wav_replace_ops(pred_audio, lr_audio[i])
            pred_audio_list.append(pred_audio)
        return pred_audio_list

    def get_x_shape(self, cond) -> tuple:
        return cond['input_concat_cond'].shape
    
    def get_unconditional_condition(
        self,
        cond:Optional[dict] = None,
        condition_device:Optional[torch.device] = None
    ) -> dict:
        uncond_dict = dict()
        if 'input_concat_cond' in cond:
            uncond_dict['input_concat_cond'] = torch.zeros_like(cond['input_concat_cond'], device=cond['input_concat_cond'].device) - 5
        if 'cross_attn_cond' in cond:
            uncond_dict['cross_attn_cond'] = torch.zeros_like(cond['cross_attn_cond'], device=cond['cross_attn_cond'].device)
        return uncond_dict
    
    def apply_model(
        self,
        x:Tensor,
        t:Tensor,
        cond:Optional[Union[dict,Tensor]],
        cfg_scale:Optional[float] = None,
        **kwargs
    ) -> Tensor:
        batch_size = x.shape[0]
        if cfg_scale is None or cfg_scale == 1.0:
            if cond is None:
                return self.model(x, t)
            else:
                return self.model(x, t, **cond)
        else:
            if 'cross_attn_cond' not in cond:
                return super().apply_model(x, t, cond, cfg_scale)
            # InstructPix2Pix: Learning to Follow Image Editing Instructions
            uncond_key_list = ['all', 'cross_attn_cond']
            uncond_dict:dict = dict()
            uncond_dict['all'] = self.get_unconditional_condition(cond=cond)

            if self.use_only_text_cond:
                cond['input_concat_cond'] = uncond_dict['all']['input_concat_cond']
                cfg_scale['input_concat_cond'] = 0.0
                
            for key in uncond_key_list:
                if key == 'all': continue
                uncond_dict[key] = {key: uncond_dict['all'][key]}

            uncond = dict()
            for uncond_key in uncond_key_list:
                uncond[uncond_key] = {key: uncond_dict[uncond_key].get(key, cond[key]) for key in cond}

            if self.cfg_calc_type == 'sequential':
                raise NotImplementedError("Sequential CFG calculation is not implemented for LatentSRDiff.")
            else:
                # InstructPix2Pix: Learning to Follow Image Editing Instructions
                cfg_x = torch.cat([x for _ in range(len(uncond_key_list) + 1)], dim=0)
                cfg_t = torch.cat([t for _ in range(len(uncond_key_list) + 1)], dim=0)
                cfg_cond = {key: torch.cat([cond[key], *[uncond[uncond_key][key] for uncond_key in uncond_key_list]], dim=0) for key in cond}
                output_cond_uncond = self.model(cfg_x, cfg_t, **cfg_cond)
                
                output_dict = dict()
                for i, output_key in enumerate(['cond'] + uncond_key_list):
                    output_dict[output_key] = output_cond_uncond[i*batch_size : (i+1)*batch_size]

            output_cfg = output_dict['all']
            output_cfg += cfg_scale['input_concat_cond'] * (output_dict['cross_attn_cond'] - output_dict['all'])
            output_cfg += cfg_scale['cross_attn_cond'] * (output_dict['cond'] - output_dict['cross_attn_cond'])

            return output_cfg

if __name__ == "__main__":
    from torch_jaekwon.h_params import HParams
    from torch_jaekwon.get_module import GetModule as get_module
    from torch_jaekwon.util import util_data
    #meta_list = util_data.walk('artifacts/data/preprocessed/audiosr/test_text_4000/ResampleTo48k/ESC50_fold5', '.pkl')
    #for meta in meta_list:
    #    print(util_data.pickle_load(meta['file_path']))
    config_path = 'config/saga_sr.yaml'
    
    HParams().set_config(config_path)
    model = get_module.get_module(class_type = 'model', module_name = HParams().model.class_meta['name'], arg_dict=HParams().model.class_meta['args'])
    
    sample_length= round(HParams().data.config['segment_sec'] * HParams().data.config['sample_rate'] )
    model(
        x_start = torch.randn((1, 2, sample_length)), #torch.randn((1, 2, sample_length)), None
        cond = {
            'audio': torch.randn((1, 2, sample_length)),
            'input_cutoff_ratio': torch.tensor([0.5]),
            'output_cutoff_ratio': torch.tensor([0.9]),
            'prompt': 'wow thats amazing dssdsds dsdsds sdsdsdsd dsdsd'
        }
    )
    print('')
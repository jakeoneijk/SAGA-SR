from typing import List

import librosa
import numpy as np
np.float = float
import torch

from torch_jaekwon.util import util_data, util_torch

class UtilAudioSR:
    @staticmethod
    def get_norm_scale_for_audio_list(audio_list:List[torch.Tensor]) -> dict:
        '''
        state_dict:dict = {
                'mean_scale_factor': cond_dict['mean_scale_factor'],
                'var_scale_factor': cond_dict['var_scale_factor']
            }
        '''
        mean_scale_factor = None
        var_scale_factor_list = []
        for audio in audio_list:
            if audio is not None:
                _, scale_dict = UtilAudioSR.normalize_wav(audio, None)
                if mean_scale_factor is None:
                    mean_scale_factor = scale_dict['mean_scale_factor']
                var_scale_factor_list.append(scale_dict['var_scale_factor'])

        var_scale_factor = torch.stack(var_scale_factor_list, dim=0)
        var_scale_factor = torch.max(var_scale_factor, dim=0).values
        return {'mean_scale_factor': mean_scale_factor, 'var_scale_factor': var_scale_factor}

    @staticmethod
    def normalize_wav(waveform:torch.Tensor, scale_dict:dict):
        mean_scale_factor = torch.mean(waveform, dim=1, keepdim=True) if scale_dict is None else scale_dict['mean_scale_factor']
        waveform = waveform - mean_scale_factor

        var_scale_factor = torch.max(torch.abs(waveform), dim=1, keepdim=True)[0] if scale_dict is None else scale_dict['var_scale_factor']

        waveform = waveform / (var_scale_factor + 1e-8)
        return waveform * 0.5, {'mean_scale_factor':mean_scale_factor, 'var_scale_factor':var_scale_factor}

    @staticmethod
    def denormalize_wav(waveform:torch.Tensor, scale_dict:dict):
        waveform = waveform * 2.0
        waveform = waveform * (scale_dict['var_scale_factor'] + 1e-8)
        waveform = waveform + scale_dict['mean_scale_factor']
        return waveform

    @staticmethod
    def mel_replace_ops(
        predict_mel:torch.Tensor,       #[batch, 1, time, melbin], log mel spectrogram
        gt_low_pass_mel:torch.Tensor,   #[batch, 1, time, melbin], log mel spectrogram
        debug_message:bool=False
    ) -> torch.Tensor:              #[batch, 1, time, melbin]
        batch_size = predict_mel.size(0)
        for i in range(batch_size):
            cutoff_melbin = UtilAudioSR.locate_cutoff_freq(torch.exp(gt_low_pass_mel[i].squeeze()))

            if debug_message:
                ratio = predict_mel[i][...,:cutoff_melbin]/gt_low_pass_mel[i][...,:cutoff_melbin]
                print(torch.mean(ratio), torch.max(ratio), torch.min(ratio))

            predict_mel[i][..., :cutoff_melbin] = gt_low_pass_mel[i][..., :cutoff_melbin]
        return predict_mel
    
    @staticmethod
    def wav_replace_ops(
        pred_wav:torch.Tensor,          #[batch, 1, time]
        gt_low_pass_wav:torch.Tensor    #[batch, 1, time]
    ) -> torch.Tensor:              #[batch, 1, time]
        device = pred_wav.device
        pred_wav = util_data.fit_shape_length(pred_wav, 2).cpu().detach().numpy()
        gt_low_pass_wav = util_data.fit_shape_length(gt_low_pass_wav, 2).cpu().detach().numpy()
        for i in range(pred_wav.shape[0]):

            out = pred_wav[i]
            x = gt_low_pass_wav[i]
            cutoffratio = UtilAudioSR.get_cutoff_index_np(x)

            length = out.shape[0]
            stft_gt = librosa.stft(x)

            stft_out = librosa.stft(out)
            energy_ratio = np.mean(
                np.sum(np.abs(stft_gt[cutoffratio]))
                / np.sum(np.abs(stft_out[cutoffratio, ...]))
            )
            energy_ratio = min(max(energy_ratio, 0.8), 1.2)
            stft_out[:cutoffratio, ...] = stft_gt[:cutoffratio, ...] / energy_ratio

            out_renewed = librosa.istft(stft_out, length=length)
            pred_wav[i] = out_renewed
        
        return torch.FloatTensor(pred_wav).to(device)
    
    @staticmethod
    def get_cutoff_ratio(audio, nfft=2048) -> float:
        """
        Get the cutoff ratio for the audio signal based on its STFT energy.
        :param audio: Audio signal as a numpy array.
        :param nfft: Number of FFT points for STFT.
        :return: Cutoff ratio as a float.
        """
        if isinstance(audio, torch.Tensor):
            audio = util_torch.to_np(audio)
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=0)
        assert len(audio.shape) == 1, f'Audio shape must be 1D or 2D, but got {audio.shape}'
        try:
            return UtilAudioSR.get_cutoff_index_np(audio, nfft=nfft) / (nfft // 2 + 1)
        except Exception as e:
            print(f"Error in get_cutoff_ratio: {e}")
            return 0.0

    @staticmethod
    def get_cutoff_index_np(x, nfft=2048):
        stft_x = np.abs(librosa.stft(x, n_fft=nfft))
        energy = np.cumsum(np.sum(stft_x, axis=-1))
        return UtilAudioSR.find_cutoff(energy, 0.985)
    
    @staticmethod
    def locate_cutoff_freq(stft, percentile=0.985):
        magnitude = torch.abs(stft)
        energy = torch.cumsum(torch.sum(magnitude, dim=0), dim=0)
        return UtilAudioSR.find_cutoff(energy, percentile)
    
    @staticmethod
    def find_cutoff(x, percentile=0.95):
        percentile = x[-1] * percentile
        for i in range(1, x.shape[0]):
            if x[-i] < percentile:
                return x.shape[0] - i
        return 0
    
    
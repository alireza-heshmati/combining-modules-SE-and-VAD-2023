import torch
import numpy as np
import torch.nn as nn

from proposed_framework.utils import (preprocessing_for_GAGNet,
                                      postprocessing_for_GAGNet,
                                      decision_rule,
                                      post_processing_VAD,
                                      cal_frame_sample_pyannote,
                                      STFT
                                      )

from speechbrain.dataio.preprocess import AudioNormalizer



class ENCODER_FRAMEWORK(nn.Module):
    """End-to-End our proposed framework according to Phase-3 report.

    Arguments
    ---------
        vad : class
            The pretrained VAD model.
        enhancement : class
            The pretrained enhancement model.
        asr_encoder : class
            The pretrained and freezed Hamrah encoder.
        vad_configs : dict
            Configs of the VAD model.
        decision_th_ms : int
            Threshold to detect valid file.
        pre_proc_sensitivity_ms : int
            Threshold for preprocessing of zero labels.

    Returns
    -------
        enh_embed : float (Tensor)
            Out put of framework according to the ecoder output.
        active_ind : bool
            The indices of valid files.
        disactive_ind : bool
            The indices of invalid files.

    """
    def __init__(self,
                 vad,
                 enhancement,
                 asr_encoder,
                 vad_configs,
                 decision_th_ms = 200,
                 pre_proc_sensitivity_ms = 100
                 ):
        super(ENCODER_FRAMEWORK, self).__init__()

        self.normalizer = AudioNormalizer(16000)
        self.vad = vad
        self.stft_layer = STFT().cuda()
        self.enhancement = enhancement
        self.asr_encoder = asr_encoder
        self.decision_th_ms = decision_th_ms
        self.sensitivity_ms = pre_proc_sensitivity_ms
        self.vad_configs = vad_configs


    def VAD_module(self, speechfiles):
        """Voice Activity Detection (VAD) module according to Phase-3 report.

        Arguments
        ---------
        speechfiles : float (Tensor)
            The normalized audio files.

        Returns
        -------
        active_ind : bool
            The indices of valid files.
        disactive_ind : bool
            The indices of invalid files.

        """
        if self.vad_configs["is_onnx"]:
            speechfiles = speechfiles.numpy()
            model_input = self.vad.get_inputs()[0]


            vad_predict = self.vad.run(None, {model_input.name: speechfiles})[0]
            vad_predict = (vad_predict.squeeze() > 0.5).astype(np.int32)
            speechfiles = torch.tensor(speechfiles)
            vad_predict = torch.tensor(vad_predict)
        else :
            vad_predict = self.vad(speechfiles)
            vad_predict = (vad_predict.squeeze() > 0.5).int()

        _ , len_frame = cal_frame_sample_pyannote(speechfiles.shape[-1],
                                                         sinc_step= self.vad_configs["sincnet_stride"]
                                                         )
        l_fr_ms = len_frame/16
        vad_predict = post_processing_VAD(vad_predict, goal = 1, len_frame_ms = l_fr_ms,
                                       sensitivity_ms = self.sensitivity_ms)
        
        active_ind, disactive_ind = decision_rule(vad_predict, l_fr_ms = l_fr_ms,
                                       decision_th_ms = self.decision_th_ms)
        
        return active_ind, disactive_ind
    
    def SE_module(self, noisy_stft):
        """Speach Enhancement (SE) module according to Phase-3 report.

        Arguments
        ---------
        noisy_stft : float (Tensor)
            The standard STFT of valid audio files.

        Returns
        -------
        enh_stft : float (Tensor)
            The standard enhanced STFT of valid audio files.

        """
        noisy_stft = preprocessing_for_GAGNet(noisy_stft)
        enh_stft = self.enhancement(noisy_stft)[-1]
        enh_stft = postprocessing_for_GAGNet(enh_stft)
        
        return enh_stft
    
    def forward(self, speechfiles, length_ratio):
        """This method should implement forwarding operation in the ENCODER_FRAMEWORK.

        Arguments
        ---------
        speechfiles : float (Tensor)
            The audio files.
        length_ratio : float (Tensor)
            The original length ratio of each audio in input related to maximum length.

        Returns
        -------
        enh_embed : float (Tensor)
            Out put of framework according to the ecoder output.
        active_ind : bool
            The indices of valid files.
        disactive_ind : bool
            The indices of invalid files.

        """
        
        speechfiles = speechfiles
        speechfiles= self.normalizer(speechfiles.unsqueeze(dim=1),16000)
        
        active_ind, disactive_ind = self.VAD_module(speechfiles)

        valid_speechfiles = speechfiles[active_ind].cuda()
        length_ratio = length_ratio[active_ind].cuda()

        valid_stft = self.stft_layer(valid_speechfiles)

        enh_stft = self.SE_module(valid_stft)

        enh_embed = self.asr_encoder(enh_stft, length_ratio)
        
        return enh_embed, active_ind, disactive_ind


import torch
import torch.nn as nn


class STFT(nn.Module):
    """Short-time Fourier transform (STFT).

    Arguments
    ---------
        input : float (Tensor)
            The input tensor of shape `(B, L)` where `B` is an optional.
        n_fft : int
            Size of Fourier transform.
        hop_length : int
            The distance between neighboring sliding window.
        window : str
           The optional window function.
        normalized : bool
            Controls whether to return the normalized STFT results.
        pad_mode : str
            controls the padding method used.
        return_complex : bool
            Whether to return a complex tensor, or a real tensor with 
            an extra last dimension for the real and imaginary components.

    Returns
    -------
        x_istft : float (Tensor)
            STFT of the input.

    """
    def __init__(self,
                 n_fft=400,
                 hop_length=160,
                 window="hamming_window",
                 normalized=False,
                 pad_mode="constant",
                 return_complex=False):
        super(STFT, self).__init__()
        
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalized = normalized
        self.return_complex = return_complex
        self.pad_mode = pad_mode
        self.window = getattr(torch, window)(n_fft)
        
        
    def forward(self, x):
        """This method should implement forwarding operation in the STFT.

        Arguments
        ---------
        x : float (Tensor)
            The input of STFT.

        Returns
        -------
        x_stft : float (Tensor)
            The output of STFT.
        """
        x_stft = torch.stft(input=x,
                            n_fft=self.n_fft, 
                            hop_length=self.hop_length,
                            window=self.window.to(x.device),
                            normalized=self.normalized,
                            pad_mode=self.pad_mode,
                            return_complex=self.return_complex)
        
        return x_stft 
        
        
def load_asr_encoder(path, device):
    """Load ASR encoder and adapt it with device.

    Arguments
    ---------
        path : str
            The path of Hamrah ASR encoder checkpoint.
        device : str
            "cuda" or "cpu".

    Returns
    -------
        encoder : float (Tensor)
            Pre-trained Hamrah ASR encoder, adapting with device.

    """
    network = torch.load(path)
    
    encoder = network["encoder"]
    encoder.compute_features.compute_STFT = torch.nn.Identity()

    if device != "cpu":
        encoder = encoder.to(device)
        encoder.normalize.glob_mean = encoder.normalize.glob_mean.to(device)
        encoder.normalize.glob_std = encoder.normalize.glob_std.to(device)
        
    return encoder


def preprocessing_for_GAGNet(noisy_stft):
    """Pre-processing of GAGNet models for input of them.

    Arguments
    ---------
        noisy_stft : float (Tensor)
            STFT of the input of GAGNet models.

    Returns
    -------
        noisy_stft : float (Tensor)
            Changed STFT of the input of GAGNet models.

    """
    noisy_stft = noisy_stft.permute(0,3,2,1)
    noisy_mag = torch.norm(noisy_stft, dim=1) ** 0.5
    noisy_phase = torch.atan2(noisy_stft[:, -1, ...], noisy_stft[:, 0, ...])
    
    noisy_stft = torch.stack((noisy_mag * torch.cos(noisy_phase),
                               noisy_mag * torch.sin(noisy_phase)), dim=1)

    return noisy_stft


def postprocessing_for_GAGNet(enhancement_output):
    """Post-processing of GAGNet models for output of them.

    Arguments
    ---------
        enhancement_output : float (Tensor)
            Output of GAGNet models.

    Returns
    -------
        enhancement_output : float (Tensor)
            Changed output of GAGNet models.

    """
    enhancement_output = enhancement_output.permute([0,2,3,1])
    est_mag = torch.norm(enhancement_output, dim=-1)**2.0
    est_phase = torch.atan2(enhancement_output[..., -1], enhancement_output[...,0])
    enhancement_output = torch.stack((est_mag*torch.cos(est_phase),
                                      est_mag*torch.sin(est_phase)), dim=-1)

    return enhancement_output.permute([0,2,1,3])

def cal_frame_sample_pyannote(wav_length,
                              sinc_step=10,
                              sinc_filter=251,
                              n_conv=2,
                              conv_filter= 5,
                              max_pool=3):
    """Define the number and the length of frames according to Pyannote model

    Arguments
    ---------
    wav_length : int
        Length of wave
    sinc_step : int
        Frame shift
    sinc_filter : int
        Length of sincnet filter
    n_conv : int
        Number of convolutional layers
    conv_filter : int
        Length of convolution filter
    max_pool : int
        Lenght of maxpooling
    
    Returns
    -------
    n_frame : float
        The number of frames according to Pyannote model
    sample_per_frame : float
        The length of frames according to Pyannote model

    """

    n_frame = (wav_length - (sinc_filter - sinc_step)) // sinc_step
    n_frame = n_frame // max_pool

    for _ in range(n_conv):
        n_frame = n_frame - (conv_filter - 1)
        n_frame = n_frame // max_pool

    sample_per_frame = wav_length // n_frame

    return n_frame, sample_per_frame

def changed_index(ind, step = 0):
    ind_bool = ind < ind.min() - 1
    if step == -1 :
        ind_bool[1:] = (ind+1)[:-1] == ind[1:] 
    else:
        ind_bool[:-1] = (ind-step)[1:] == ind[:-1]
    
    ind_bool = ~ind_bool
    return ind_bool


def post_processing_VAD(vad_out, goal = 1, len_frame_ms = 20, sensitivity_ms = 200):
    """Post-processing of VAD models to change 0 label0 with 1 labels according to a sensitivity.

    Arguments
    ---------
        vad_out : float (Tensor)
            Output of the VAD model.
        goal : int (Tensor)
            The goal of change.
        len_frame_ms : float 
            Length of decision frame.
        sensitivity_ms : float 
            Threshold to change labels that are less than it.

    Returns
    -------
        vad_out : float (Tensor)
            The pre-processed output.

    """

    Th = max(int(sensitivity_ms // len_frame_ms), 1)
    
    ind0,ind1 = torch.where(vad_out== goal)
    
    if len(ind0) != 0:
        ind1_max = vad_out.shape[-1] - 1
        ind0_last_bool = changed_index(ind0.clone())

        ind0_last = torch.where(ind0_last_bool)[0]
        ind0_first = torch.zeros_like(ind0_last)
        ind0_first[1:] = ind0_last[:-1] + 1
        ind0_first[0] = 0

        ind1_l1_bool = changed_index(ind1.clone(), step = 1)
        ind1_l1_bool[ind0_last] = False

        ind1_f1_bool = changed_index(ind1.clone(), step = -1)
        ind1_f1_bool[ind0_first] = False


        dif_bool = ind1[ind1_f1_bool] - ind1[ind1_l1_bool] > Th + 1
        l1_bool_temp = ind1_l1_bool[ind1_l1_bool].clone()
        l1_bool_temp[dif_bool] = False
        ind1_l1_bool[ind1_l1_bool.clone()] = l1_bool_temp

        f1_bool_temp = ind1_f1_bool[ind1_f1_bool].clone()
        f1_bool_temp[dif_bool] = False
        ind1_f1_bool[ind1_f1_bool.clone()] = f1_bool_temp


        second_ind = ind1[ind1_l1_bool].clone()
        for i in range(1,Th+1):
            second_ind = torch.clip(ind1[ind1_l1_bool]+i,0,ind1_max)
            desired_out = (second_ind < ind1[ind1_f1_bool])
            temp_b = vad_out[ind0[ind1_l1_bool], second_ind].clone()
            temp_b[desired_out] = goal
            vad_out[ind0[ind1_l1_bool], second_ind] = temp_b.clone()
    
    return vad_out


def decision_rule(vad_out, l_fr_ms = 20, decision_th_ms = 200):
    """DEcision rule to detect valid speech files for VAD module

    Arguments
    ---------
        vad_out : float (Tensor)
            The post-processed output of VAD model.
        l_fr_ms : float 
            Length of decision frame.
        decision_th_ms : float 
            Threshold to have valid speech files.

    Returns
    -------
        active_ind : bool
            The indices of valid files.
        disactive_ind : bool
            The indices of invalid files.

    """
    
    decision_th = max(decision_th_ms // l_fr_ms,1)
    
    ind0,ind1 = torch.where(vad_out== 1)
    
    if len(ind0) != 0:
        ind1_l1_bool = changed_index(ind1.clone(), step = 1)
        ind1_f1_bool = changed_index(ind1.clone(), step = -1)

        dif_bool = ind1[ind1_l1_bool] - ind1[ind1_f1_bool]+1 >= decision_th 
        active_ind = ind0[ind1_l1_bool][dif_bool].unique()
        disactive_ind = torch.range(0,vad_out.shape[0]-1,dtype=active_ind.dtype)
        disactive_ind[active_ind] = -1
        disactive_ind = disactive_ind.unique()[1:]
    else:
        active_ind = ind0
        disactive_ind = torch.range(0,vad_out.shape[0]-1,dtype=active_ind.dtype)
    
    return active_ind, disactive_ind

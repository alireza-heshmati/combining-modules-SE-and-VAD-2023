import onnxruntime as ort
import torch
from recipes.utils import load_model_config
from models.gagnet import GaGNet
from models.pyannote import PyanNet

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

def create_pretrained_enhancement_and_configs(model_name = "gagnet-v4"):
    """Create the pretrained enhancement model and its configs

    Arguments
    ---------
    model_name : str
        Name of enhancement model ("gagnet-v4").

    Returns
    -------
    enhacement : class
        The pretrained enhancement model.
    model_configs : dict
        The enhancement model configs.

    """
    if model_name == "gagnet-v4":
        config_path = "./recipes/gagnet-v4.json"
    else:
        raise ValueError("the name of the speech enhancement is not supported!!")

    model_configs = load_model_config(config_path)

    enhacement = GaGNet(
        cin=model_configs["cin"],
        k1=tuple(model_configs["k1"]),
        k2=tuple(model_configs["k2"]),
        c=model_configs["c"],
        kd1=model_configs["kd1"],
        cd1=model_configs["cd1"],
        d_feat=model_configs["d_feat"],
        p=model_configs["p"],
        q=model_configs["q"],
        dilas=model_configs["dilas"],
        fft_num=model_configs["fft_num"],
        is_u2=model_configs["is_u2"],
        is_causal=model_configs["is_causal"],
        is_squeezed=model_configs["is_squeezed"],
        acti_type=model_configs["acti_type"],
        intra_connect=model_configs["intra_connect"],
        norm_type=model_configs["norm_type"],
        )
    enhacement.load_state_dict(torch.load(model_configs["save_path"]))
    return enhacement, model_configs

def create_pretrained_vad_and_configs(model_name = "pyannote-v2.3"):
    """Create the pretrained VAD model and its configs

    Arguments
    ---------
    model_name : str
        Name of VAD model ("gagnet-v4").

    Returns
    -------
    VAD : class 
        The pretrained VAD model.
    model_configs : dict
        The VAD model configs.

    """
    if model_name == "pyannote-v2.3":
        config_path = "./recipes/pyannote-v2.3.json"
    else:
        raise ValueError("the name of the VAD is not supported!!")

    model_configs = load_model_config(config_path)

    if model_configs["is_onnx"]:
        vad = ort.InferenceSession(model_configs["save_path_onnx"],
                                    providers=["CPUExecutionProvider"])
    else:
        vad = PyanNet(model_configs)
        vad.load_state_dict(torch.load(model_configs["save_path"]))
    return vad, model_configs


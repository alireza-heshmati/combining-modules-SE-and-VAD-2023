
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from proposed_framework.framework import ENCODER_FRAMEWORK
from dataio.utils import creat_dataset_path, collate_fn
from dataio.dataset import DS_FA_v04
from models.utils import (create_pretrained_enhancement_and_configs,
                          create_pretrained_vad_and_configs,
                          load_asr_encoder
                          )


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-gagnet_name", "--name_of_enhancement_model",
                required=True,
                type=str,
                help="name of GAGNet models, such as gagnet-v4" )

ap.add_argument("-pyannote_name", "--name_of_vad_model",
                required=True,
                type=str,
                help="name of VAD models, such as pyannote-v2.3" )

ap.add_argument("-data_path", "--path_of_data_files", 
                required=True,
                type=str,
                help="path of main file of data, such as ./dataset")

ap.add_argument("-invalid_filenames", "--path_of_invalid_data_filenames", 
                required=True,
                type=str,
                help="path of invalid data filename, such as SAD_noise_filenames.txt")

ap.add_argument("-valid_filenames", "--path_of_valid_data_filenames", 
                required=True,
                type=str,
                help="path of valid data filename, such as test_filenames.csv")

args = vars(ap.parse_args())


gagnet_name = args["name_of_enhancement_model"]
pyannote_name = args["name_of_vad_model"]
BASEPATH = args["path_of_data_files"]
enhancedfile = args["name_of_enhanced_file"]
invalid_filenames = ['path_of_invalid_data_filenames']
valid_filenames = ["path_of_valid_data_filenames"]

dataset_path = creat_dataset_path(invalid_filenames, valid_filenames, BASEPATH)

print(len(dataset_path))

data_loader = DataLoader(
    DS_FA_v04(dataset_path),
    batch_size=128,
    shuffle=False,
    drop_last=True,
    collate_fn= collate_fn,
    num_workers=8,
    pin_memory=False
)


pretrained_enhacement, enhancement_configs \
    = create_pretrained_enhancement_and_configs(model_name = gagnet_name)

pretrained_vad, vad_configs = create_pretrained_vad_and_configs(model_name = pyannote_name)

asr_encoder = load_asr_encoder(enhancedfile, device="cuda")


if vad_configs["is_onnx"]:
    print("VAD with ONNX.")
else:
    print(f"VAD with cpu.")
    pretrained_vad.eval()

pretrained_enhacement.cuda()

pretrained_enhacement.eval()
asr_encoder.eval()

print(f"Rest of encoder with cuda.")

proposed_encoder = ENCODER_FRAMEWORK(pretrained_vad,
                                pretrained_enhacement,
                                asr_encoder,
                                vad_configs,
                                decision_th_ms = 200,
                                pre_proc_sensitivity_ms = 100
                                )
n_correct = 0
n_total = 0
embeded_out = []
with torch.no_grad():  
    for data, file_target, length_ratio in tqdm(data_loader):
        enh_embed, active_ind, disactive_ind = proposed_encoder(data, length_ratio)
        embeded_out.append(enh_embed.cpu())
        preds = torch.zeros_like(file_target)
        preds[active_ind] = 1
        n_total += len(data)
        n_correct += len(preds[preds == file_target])

print(f"Number of correct: {n_correct} from {n_total}.  Accuracy: {round(n_correct/n_total, 6)}")
print(embeded_out[-1].shape)

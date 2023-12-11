# Dataset 
**Note1: For inference of the models, you should put unzipped data in [**Noisy_Dataset_V1**](./dataset/Noisy_Dataset_V1),  [**Noisy_Dataset_V2**](./dataset/Noisy_Dataset_V2) and [**SAD_noise_dataset**](./dataset/SAD_noise_dataset).**

**Note2: For running the codes, it is necessary to unzip test_filenames.zip and SAD_noise_filenames.zip files in this [**folder**](./dataset)**

This [**folder**](../dataset) contains the DS-Fa-V04 dataset that was introduced in Phase-3 report. This dataset is used for test of VAD decision rule and in this repo, as a test dataset beacuse it has both valid and invalid noisy speech files to evaluate VAD module and enhancement module. In DS-Fa-V04 dataset, valid speech files were randomly selected from test part of DS-Fa-V01 dataset, and invalid speech files (total noise files) were created by QUT dataset with length 2 - 10 seconds. Both parts of DS-Fa-V04 dataset have equal samples, about 50000. 


# Deep Image Fingerprint: Accurate And Low Budget Synthetic Image Detector

This is the official repository of the paper: "Deep Image Fingerprint: Accurate And Low Budget Synthetic Image Detector"

By: Sergey Sinitsa and Ohad Fried

Paper (add link) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Project Page (add link)


### Installation

This project was tested using Python 3.10 with a GPU. However, it is not necessary to have a GPU for the evaluation process.
The required dependencies are specified in the 'requirements.txt' file.


### Usage

After setting up the repository you may train the model or reproduce some of the experiments.
We provide code for three experiments as described below.

#### Gray image experiment

To reproduce the artifacts with a gray image, simply run 'blank_experiment.py' with the default parameters. 
An output directory will be created where you can find the reconstruction in both image space and Fourier space.

Example:
```
python blank_experment.py
```

#### Training the Model

To run 'train_dif.py', you need to specify the data directory and the model directory. 
The data directory should include two subdirectories: '0_real' and '1_fake', for real and fake images, respectively. The model directory will be used to store the extracted fingerprints.

Example for Dall $\cdot$ E-2 model:
```
python train_dif.py data_root/dalle_2 checkpoint_directory/dalle_2
```

#### Testing the Model

We included extracted fingerprints of LTIMs and GAN models described in the paper.
In both cases models were trained with 1024 samples. Due to file size constrains, we provide:
- An external [**link**](https://drive.google.com/drive/folders/1lo2-VRR8q3Elazt9P-AF1GDVypo0cpTl?usp=sharing) to test data archive, <u>which resides on anonymous google account</u>.
- 20 samples of images per each model in /data folder

To reproduce the results per model run eval_dif.py and specify fingerprint directory and data directory.
Example for DallE-2 model:
```
python eval_dif.py checks/dalle_2 data_root/dalle_2 0
```
'data_root' refers to folder which contain sub-folders for each generative model.

The expected accuracy values (%) are below:

| Method | SD 1.4 | SD 2.1 | MJ | DallE-Mini | GLIDE | DallE-2 | CycleGAN | StyleGAN | StyleGAN2 | StarGAN | BigGAN | GauGAN | ProGAN |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|DIF             | 99.3  | 89.5  | 99.0 | 99.0 | 90.3 | 79.5 | 94.4 | 96.6 | 91.5 | 99.9 | 96.9 | 91.8 | 57.7 |




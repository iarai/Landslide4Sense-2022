# [Landslide4Sense](https://gitlab.lan.iarai.ac.at/land4seen/land4seen): Multi-sensor landslide detection competition & benchmark dataset

## Contents
- [LandSlide4Sense 2022](#landslide4sense-2022)
- [Globally Distributed Landslide Detection](#globally-distributed-landslide-detection)
- [Challenge Description](#challenge-description)
- [Data Description](#data-description)
- [Baseline Code](#baseline-code)
- [Evaluation Metric](#evaluation-metric)
- [Submission Guide](#submission-guide)
- [Awards and Prizes](#awards-and-prizes)
- [Timeline](#timeline)
- [Citation](#citation)


## LandSlide4Sense 2022
![Logo](/image/Competition_figure.png?raw=true "landslide_detection")

The [Landslide4Sense](https://www.iarai.ac.at/landslide4sense/) competition, organized by [Institute of Advanced Research in Artificial Intelligence (IARAI)](https://www.iarai.ac.at/), aims to promote research in large-scale landslide detection from multi-source satellite remote sensing imagery. Landslide4Sense dataset has been derived from diverse landslide-affected areas around the world from 2015 through 2021. This benchmark dataset provides an important resource for remote sensing, computer vision, and machine learning communities to support studies on image classification and landslide detection studies.
Interested in automatically extracting landslide features from satellite imagery? Join us to help shape the first landslide detection competition.

## Globally Distributed Landslide Detection

Landslides are a frequent phenomenon in many parts of the world, including thousands of small and medium-sized ground movements following earthquakes or severe weather events. In recent years, landslides have become even more damaging due to climate change, population growth, and unplanned urbanization in mountainous areas that are highly dynamic in terms of sedimentation and erosion. Early landslide detection and inventory mapping are critical for sending humanitarian aid and responding quickly to crises. In addition, accurate landslide detection to obtain spatial information about landslides, including their exact location and extent, is a prerequisite for further analysis, such as susceptibility modeling, risk, and vulnerability assessments. Therefore, in the wake of recent advances in computer vision as well as the increased availability of both imagery and computational resources, machine/deep learning models are taking off in landslide detection like other remote sensing fields. This competition seeks to help this development and challenges participants to detect landslides around the globe based on multisensor Earth observation images. The images are collected from diverse geographical regions offering an important resource for remote sensing, computer vision, and machine learning communities to support studies on landslide detection.

## Challenge Description

The aim of the challenge is to promote innovative algorithms for automatic landslide detection using remotely sensed images around the globe, as well as to provide objective and fair comparisons among different methods. The ranking is based on a quantitative accuracy metric (F1 score) computed with respect to undisclosed test samples. Participants will be given a limited time to submit their landslide detection results after the competition starts. The top three ranked participants will be announced as the winners.

We also seek to reward promising, innovative solution approaches and thus intend to present special prizes to participants who provide such a solution regardless of the score they achieve. The award of these prizes is determined by the competition's scientific committee taking into account originality, innovation, generality and scalability.

The competition will consist of two phases:

**Phase 1 (April 1st - June 14th):** Participants are provided with training data (with labels) and additional validation images (without labels) to train and validate their methods. Participants can submit their landslide detection results for the validation set to the competition website to get feedback on the performance (Precision, Recall, and F1 score). The performance of the submission will be displayed on the online leaderboard. In addition, participants should submit a short description of the methodology (1-2 pages) to https://cloud.iarai.ac.at/index.php/s/sYQgdHryGMPQsHa according to the IJCAI template.

**Phase 2 (June 15th - June 20th):** Participants receive the test data set (without labels) and must submit their landslide detection results within 5 days from the release of the test data set. The submissions during that week will be limited to 10 times and only the F1 score will be displayed on the online leaderboard. 

This part only concerns the selected winners in Phase 2. Following the announcement of the winners, they will have to write a 4-page IJCAI-style formatted manuscript that will be included in the CDCEO workshop. Each manuscript describes the addressed problem, the proposed method, and the experimental results. The winners will be asked to send a short pre-recorded video presentation. However, they should still be present for a live Question-and-Answer period with the audience and session chair.

The winners **must submit** the working code, the learned parameters, and present their work in the CDCEO workshop at IJCAI-ECAI 2022 proceedings to receive the prizes in accordance with the terms and conditions of the competition.


## Data Description


The Landslide4Sense dataset has three splits, training/validation/test, consisting of 3799, 245, and 800 image patches, respectively. Each image patch is a composite of 14 bands. The detailed information about the 14 bands is listed below:

- **Multispectral data** from [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2): B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12.

- **Slope data** from [ALOS PALSAR](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-radar-alos-palsar-radar-processing-system): B13.       

- **Digital elevation model (DEM)** from [ALOS PALSAR](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-radar-alos-palsar-radar-processing-system): B14.

All bands in the Landslide4Sense dataset are resized to ~10m pixel size. The image patches and their corresponding labels are 128 x 128 pixels.

**Download links:** [training](https://cloud.iarai.ac.at/index.php/s/KrwKngeXN7KjkFm) and [validation](https://cloud.iarai.ac.at/index.php/s/N6TacGsfr5nRNWr).   


![Logo](/image/Data_figure.png?raw=true "landslide_detection")

The _Landslide4Sense_ dataset is structured as follows:
```
├── TrainData/
│   ├── img/     
|   |   ├── image_1.h5
|   |   ├── ...
|   |   ├── image_3799.h5
│   ├── mask/
|   |   ├── mask_1.h5
|   |   ├── ...
|   |   ├── mask_3799.h5
├── ValidData/
|   ├── img/     
|   |   ├── image_1.h5
|   |   ├── ...
|   |   ├── image_245.h5
├── TestData/
    ├── img/     
        ├── image_1.h5
        ├── ...
        ├── image_800.h5
```

Note that the label files (mask files) are only accessible in the training set.

Mapping classes used in the competition:

| Class Number |        Class Name     | Class Code in the Label |
 :-: | :-: | :-:
| 1 | Non-landslide | 0 |
| 2 | Landslide | 1 |


## Baseline Code

This repository provides a simple baseline for the [Landslide4Sense](https://www.iarai.ac.at/landslide4sense/) competition based on the state-of-the-art DL model for semantic segmentation, implemented in PyTorch. It contains a customizable training script for [U-Net](https://arxiv.org/abs/1505.04597) along with the dataloader for reading the training and test samples (see `landslide_dataset.py` in the `dataset` folder).

The provided code can be used to predict baseline results for the competition or as a comparison method for your solutions. Feel free to fork this repository for further use in your work!

**Required packages and libraries:**

- Pytorch 1.10
- CUDA 10.2
- h5py

**To train the baseline model:**

```
python Train.py --data_dir <THE-ROOT-PATH-OF-THE-DATA> \
                --gpu_id 0
```  

Please replace `<THE-ROOT-PATH-OF-THE-DATA>` with the local path where you store the Landslide4Sense data.

The trained model will then be saved in `./exp/`

**To generate prediction maps on the validation set with the trained model:**

```
python Predict.py --data_dir <THE-ROOT-PATH-OF-THE-DATA> \
               --gpu_id 0 \
               --test_list ./dataset/valid.txt \
               --snapshot_dir ./validation_map/ \
               --restore_from ./exp/<THE-SAVED-MODEL-NAME>.pth
```  
Please replace `<THE-SAVED-MODEL-NAME>` with the name of your trained model.

Alternatively, our **pretrained model** is available at    [here](https://cloud.iarai.ac.at/index.php/s/CgbjDRK6B5KYaLE).



The generated prediction maps (in `h5` format) will then be saved in `./validation_map/`

**To generate prediction maps on the test set with the trained model:**

```
python Predict.py --data_dir <THE-ROOT-PATH-OF-THE-DATA> \
               --gpu_id 0 \
               --test_list ./dataset/test.txt \
               --snapshot_dir ./test_map/ \
               --restore_from ./exp/<THE-SAVED-MODEL-NAME>.pth
```  

The generated prediction maps (in `h5` format) will then be saved in `./test_map/`

## Evaluation Metric

The F1 score of the landslide category is adopted as the evaluation metric in **Track DLD** for the leaderboard:

![](https://latex.codecogs.com/svg.image?F_1=&space;2\cdot&space;\frac{precision\cdot&space;recall}{precision&plus;recall})

With the provided baseline method and the pretrained model, you can achieve the following result on the validation set:

| Validation Set | Precision | Recall | F1 Score |
| :--: | :--: | :--: | :--: |
| U-Net Baseline | 51.75 | 65.50 | 57.82 |

Note that the evaluation ranking is **ONLY** based on the **F1 score of the landslide category** in both validation and test phases.

## Submission Guide

                                                                                            
For both validation and test phases, participants should submit a `ZIP` file containing the prediction files for all test images. Each pixel in the prediction file corresponds to the class category with `1` for *landslide regions* and `0` for *non-landslide regions* (similar to the reference data of the training set).

Specifically, the predictions for each test image should be encoded as a `h5` file with the Byte (uint8) data type, and match the dimensions of the test images (i.e., `128×128`).


The submitted `ZIP` file in the validation phase should be structured as follows:
```
├── submission_name.zip     
    ├── mask_1.h5
    ├── mask_2.h5
    ├── ...
    ├── mask_245.h5
```

The submitted `ZIP` file in the test phase should be structured as follows:
```
├── submission_name.zip     
    ├── mask_1.h5
    ├── mask_2.h5
    ├── ...
    ├── mask_800.h5
```


## Awards and Prizes

- First-ranked team: Voucher or cash prize worth 5,000 EUR to the participant/team and one free IJCAI-ECAI 2022 conference registration

- Second-ranked team: Voucher or cash prize worth 3,000 EUR to the participant/team and one free IJCAI-ECAI 2022 conference registration

- Third-ranked team: Voucher or cash prize worth 2,000 EUR to the participant/team and one free IJCAI-ECAI 2022 conference registration

- Also special prizes for more selected submissions.



## Timeline 


- **April 1st (Phase 1):** Contest opens. Release training and validation data. The validation leaderboard starts to receive submissions.
- **June 12th (Phase 1):** Submit a short description of the methodology (1-2 pages) to  https://cloud.iarai.ac.at/index.php/s/sYQgdHryGMPQsHa according to the IJCAI template.
- **June 15th (Phase 2):** Release test data. The validation leaderboard closes and the test leaderboard starts to receive submissions.
- **June 20th (Phase 2):** The test leaderboard stops accepting submissions.
- **June 25th:** Winner announcement. Invitations to present at the Special Competition Session at the CDCEO workshop.
- **July 10th:** Full manuscript (4-pages, IJCAI formatted) submission deadline and pre-recorded presentation video deadline.



## Citation
Please cite the following paper if you use the data or the codes: 

```
@article{tbd}
```

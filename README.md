# Group9-Final-Project
### Sports Image Classification Using Deeplearning PyTorch Pretrained models ResNet series and YOLO

<div align="center">

![Alt text](https://t3.ftcdn.net/jpg/02/78/42/76/360_F_278427683_zeS9ihPAO61QhHqdU1fOaPk2UClfgPcW.jpg)

</div>

## Overview

Welcome to the Sports Image Classification Project, a deep learning endeavor aimed at accurately categorizing sports images using PyTorch and pre-trained models. This repository showcases the implementation and evaluation of ResNet50, ResNet101, ResNet152, and YOLO8x.pt models. The project meticulously measures training time, loss, and validation, training, and testing accuracies. The dataset is sourced from Kaggle and loaded using the Kaggle Python library. Additionally, the trained models are persistently saved for future applications.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Enhancements](#future-enhancements)
- [Installation](#installation)
- [License](#license)
- [Contact](#contact)

## Introduction

The Sports Image Classification Project addresses the intricate task of classifying sports images based on visual cues. Leveraging the robustness of deep learning models, we delve into the performance of varied pre-trained models for this classification endeavor.

## Dataset

Our dataset stems from Kaggle, encompassing an expansive collection of sports images spanning diverse categories. To ensure model efficacy, the dataset is meticulously preprocessed and meticulously divided into training, validation, and testing subsets. Each image is meticulously labeled, thereby facilitating supervised training. [link](https://www.kaggle.com/datasets/gpiosenka/sports-classification)

You can also download dataset and make dataset ready by applying following procedure.

1. Install Kaggle Library

    ```
    pip install kaggle
    ```
2. Download Data Using kaggle command

    ```python
        if "sports-classification" in os.listdir():
            print('Data Already Exist')
        else:
            print("Downloading Data From Kaggle")
            !kaggle datasets download -d gpiosenka/sports-classification
    ```

## Models

The following pre-trained models are harnessed for our sports image classification project:

1. ResNet50
2. ResNet101
3. ResNet152
4. YOLO8x.pt

[For More Info - ResNet](https://pytorch.org/hub/pytorch_vision_resnet/) |  [For YOLO](https://docs.ultralytics.com/tasks/classify/)

## Training

Our models undergo rigorous training using PyTorch and the Adam optimizer. Our focus extends beyond mere accuracy, encompassing detailed tracking of training time, loss evolution, and accuracy trajectory. Training is expedited through GPU acceleration.

## Evaluation

To gauge model performance, we subject the trained models to a comprehensive evaluation on our validation dataset. The evaluation encompasses validation accuracy and loss metrics, revealing each model's aptitude for generalization.


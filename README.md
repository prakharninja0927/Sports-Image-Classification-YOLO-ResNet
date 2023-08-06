# Group9-Final-Project
### Sports Image Classification Using Deeplearning PyTorch Pretrained models ResNet series and YOLO

<div align="center">

![Alt text](https://t3.ftcdn.net/jpg/02/78/42/76/360_F_278427683_zeS9ihPAO61QhHqdU1fOaPk2UClfgPcW.jpg)

</div>

## Group9 
1. Prakhar Patel	101413720
2. Nihitha Patcha	101446620
3. Archana Wasti Dahal	101465212
4. Vijay Karthik Bethapudi	101442692

[Repoort link](https://prakharninja0927.github.io/group9-final-project/)

## Overview

Welcome to the Sports Image Classification Project, a deep learning endeavor aimed at accurately categorizing sports images using PyTorch and pre-trained models. This repository showcases the implementation and evaluation of ResNet50, ResNet101, ResNet152, and YOLO8x.pt models. The project meticulously measures training time, loss, and validation, training, and testing accuracies. The dataset is sourced from Kaggle and loaded using the Kaggle Python library. Additionally, the trained models are persistently saved for future applications.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [Future Enhancements](#future-enhancements)

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

## install requirements.txt

```bash
pip install requirements.txt
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

## Sports Image Classification Applications

1. **Broadcasting and Media:** Sports image classification streamlines media production by automatically tagging and organizing images for live broadcasts, highlight reels, and social media sharing.

2. **Player Performance Analysis:** Coaches and analysts gain insights into player movements, gestures, and actions, aiding performance evaluation, tactical improvements, and future game strategies.

3. **Referee Assistance:** Real-time image analysis assists referees and umpires in making accurate decisions during crucial moments, ensuring fairness and minimizing controversies.

4. **Injury Prevention and Rehabilitation:** By tracking body language and rehabilitation progress, sports image classification contributes to injury prevention and facilitates effective recovery.

5. **Coaching and Training:** Athletes benefit from personalized coaching as their form and technique are analyzed, leading to better training routines and skill development.

6. **Sports Analytics:** Visual data, including player positioning and ball trajectory, enriches sports analytics, providing deeper insights into game statistics and strategies.

7. **Sponsorship and Branding:** Brands can assess sponsorship impact by identifying logo appearances, aiding sponsorship decisions and measuring advertising effectiveness.

8. **Security and Crowd Management:** Image classification enhances security at sports events by monitoring crowds and identifying potential risks, ensuring a safe environment.

9. **Fan Engagement:** Augmented reality applications driven by image classification offer interactive fan experiences, adding an exciting dimension to spectator involvement.

Sports image classification revolutionizes broadcasting, coaching, security, and fan engagement, contributing to a more dynamic and immersive sports ecosystem.

## Conclusion

In the Sports Image Classification Project, we successfully implement an end-to-end system using diverse pre-trained models. This project underscores the potential of deep learning in discerning nuanced features within sports images. The model selection and hyperparameter tuning process significantly influence final outcomes.

## Future Enhancements

Prospective enhancements to our project include:

1. Integration of advanced data augmentation techniques to bolster model robustness.
2. Targeted fine-tuning of pre-trained models on our sports dataset, amplifying classification accuracy.
3. Exploring alternative architectures and transfer learning strategies to further elevate model performance.






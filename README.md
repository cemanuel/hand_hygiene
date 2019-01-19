# Overview

### Supervised Learning for Hand-Hygiene Binary Detection
Due to the recent progress of cost-effective depth sensors, many tasks in smart hospitals have been automated through AI-assisted solutions. One of these tasks, automating hand-hygiene compliance, can be used to prevent hospital acquired infections. In this paper, convolutional networks (CNNs) are used to classify top-view depth images as “using dispenser” or “not using dispenser.” One model is trained per sensor.

### WorkFlow

![Alt Text](https://github.com/cemanuel/hand_hygiene/blob/master/workflow.png)

1.) Segmentation
Otsu thresholding is used to separate the background from the objects in the foreground. One can view examples of these segmentation results, with threshold values, in the following notebook: viewing_processed_images.ipynb

2.) Classifiation
These segmentations are fed to a convolutional neural network that is trained to output whether an image shows the activity of hand washing or not.


### Discussion
Due to the recent progress of cost-effective depth sensors, many tasks in smart hospitals have been automated through AI-assisted solutions. One of these tasks, automating hand-hygiene compliance, can be used to prevent hospital acquired infections. In this paper, convolutional networks (CNNs) are used to classify top-view depth images as “using dispenser” or “not using dispenser.” One model is trained per sensor. 


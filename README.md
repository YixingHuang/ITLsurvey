# Incremental Transfer Learning (ITL) Survey

This is the code repository for our paper:

## A Survey of Incremental Transfer Learning: Combining Peer-to-Peer Federated Learning and Domain Incremental Learning for Multicenter Collaboration

This part investigates the efficacy of continual learning methods for classification tasks in multicenter collaboration. It is evaluated on two datasets: the Tiny ImageNet dataset and the retinal dataset.
For segmentation tasks, due to the massive changes of the framework including the network architecture, evaluation metrics, and training losses, we use a [separate repository](https://github.com/YixingHuang/ITLsurveySegmentation) for such segmentation tasks.


Incremental Transfer Learning shares trained models in a peer-to-peer federated learning manner and uses continual learning to avoid the forgetting problem caused by domain gaps. It is a more practical choice for multicenter collaboration compared with center-to-peer federated learning, because of the low communication cost, low financial cost (no need for central server), low technical difficulty and easy for management.

The framework of incremental transfer learning is illustrated in the following figure:

[logo](https://github.com/YixingHuang/ITLsurvey/blob/main/image857-8.png "Incremental Transfer Learning")

## Acknolowdgement
This repository is developed based on the [task-incremental learning respository](https://github.com/Mattdl/CLsurvey) of the [TPAMI survey paper](https://ieeexplore.ieee.org/abstract/document/9349197).

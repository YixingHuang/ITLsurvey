# Incremental Transfer Learning (ITL) Survey

This is the code repository for our paper:

## A Survey of Incremental Transfer Learning: Combining Peer-to-Peer Federated Learning and Domain Incremental Learning for Multicenter Collaboration

## [Preprint](https://arxiv.org/abs/2309.17192)

This part investigates the efficacy of continual learning methods for classification tasks in multicenter collaboration. It is evaluated on two datasets: the Tiny ImageNet dataset and the retinal dataset.
For segmentation tasks, due to the massive changes of the framework including the network architecture, evaluation metrics, and training losses, we use a [separate repository](https://github.com/YixingHuang/ITLsurveySegmentation) for such segmentation tasks.


Incremental Transfer Learning shares trained models in a peer-to-peer federated learning manner and uses continual learning to avoid the forgetting problem caused by domain gaps. It is a more practical choice for multicenter collaboration compared with center-to-peer federated learning, because of the low communication cost, low financial cost (no need for central server), low technical difficulty and easy for management.

The framework of incremental transfer learning is illustrated in the following figure:

![Incremental Transfer Learning](https://github.com/YixingHuang/ITLsurvey/blob/main/image857-8.png "Incremental Transfer Learning")

## Methods
The following classic regularization-based continual learning methods are investigated:
- Synaptic Intelligence (SI)
- Elastic Weight Consollidation (EWC)
- Memory Aware Synapses (MAS)
- Learning Without Forgetting (LWF)
- Encoder-Based Lifelong Learning (EBLL)
- Incremental Moment Matching (IMM): mean-IMM and mode-IMM
  
They are compared with baselines without conitnual learning methods:

- Fine-Tuning (FT)
- Independant Training (IT)
- Joint training (Joint)

All the above methods are investigated using single weight transfer (SWT) and cyclic weight transfer (CWT).

## Optimizers
The state-of-the-art continual learning methods typically use the SGD optimizer. In our work, we propose to use adaptive optimizers such as the Adam for better performance and ease of learning rate choice.

## Datasets
### Three datasets are used for this survey.
- The Tiny ImageNet dataset: this is available from the [CS231n course source](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
- The retinal image dataset: this is from two data sources: the [retinal fundus multi-disease image dataset (RFMiD)](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification) and the [Kaggle diabetic retinopathy detection dataset (KDRDD)](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data).
- The glioma dataset: This is from three data sources: the [BraTS 2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html), the [UCSF-PDGM dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=119705830), and the [UPenn-GBM dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642).

## Acknolowdgement
This repository is developed based on the [task-incremental learning respository](https://github.com/Mattdl/CLsurvey) of the [TPAMI survey paper](https://ieeexplore.ieee.org/abstract/document/9349197).

The following major modifications have been made to adapt the existing continual learning framework for our incremental transfer learning:
- Single-head setting instead of multi-head setting, which drastically reduces the forgetting problem and is more natural for single-task multicenter collaboration scenarios.
- Add the Adam optimizer for higher performance and easier choice of learning rates, compared with the SGD optimizer. All the algorithms need to be modified correspondingly.
- Add cyclic weight transfer to the framework, as cyclic weight transfer can achieve better performance than single weight transfer and existing continual learning frameworks all use single weight transfer only.
- Reload optimizer: the parameters of the Adam optimizer (including the learning rate) will be reloaded to have a smooth transition from center to center.
- Overfitting monitor: The monitoring of overfitting is very important to avoid drastic performance drop after training in one center.
- Data preprocessing pipeline to get independent and identically distributed (IID) data and non-IID data.
- Add the evaluation metric of monotonicity, as we want the model performance can increase stably and monotonically.
- ...

  

# Imbalanced_SAM

This is an official implementation of "ImbSAM: A Closer Look at Sharpness-Aware Minimization in Class-Imbalanced Recognition".

## Abstract
Class imbalance is a common challenge in real-world recognition tasks, where the majority of classes have few samples, also known as tail classes. We address this challenge with the perspective of generalization and empirically find that the promising Sharpness-Aware Minimization (SAM) fails to address generalization issues in class-imbalanced recognition.  Through investigating this specific type of task, we identify that its generalization bottleneck primarily lies in the severe overfitting for tail classes with limited training data. To overcome this bottleneck, we leverage class priors to restrict the generalization scope of the class-agnostic SAM and propose a class-aware smoothness optimization algorithm named Imbalanced-SAM (ImbSAM). With the guidance of class priors, our ImbSAM specifically improves generalization targeting tail classes. We also verify the efficacy of ImbSAM on two prototypical applications of class-imbalanced recognition: long-tailed classification and semi-supervised anomaly detection, where our ImbSAM demonstrates remarkable performance improvements for tail classes and anomaly.

![](./imgs/intro.png)

## Enviroment
- PyTorch 

## Thanks to

- [ASAM](https://github.com/SamsungLabs/ASAM)
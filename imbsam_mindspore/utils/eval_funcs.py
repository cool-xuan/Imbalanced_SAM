from __future__ import absolute_import, division, print_function

import numpy as np

import mindspore as ms
import mindspore.ops as ops

import math
import sklearn
from utils.dataset_CIFAR100LT import *
    
def print_accuracy(model, dataloaders, new_labelList, test_aug = True, save_dir = None):
    model.set_train(False)
    
    if test_aug:
        model = horizontal_flip_aug(model)

    predList = np.array([])
    grndList = np.array([])

    for sample in dataloaders['test'].create_dict_iterator():
        images, labels = sample['image'], sample['label']
        labels = labels.long().view(-1).asnumpy()
        logits = model(images)
        softmaxScores = ops.softmax(logits, axis=1)   

        predLabels = softmaxScores.argmax(axis=1).float().squeeze().asnumpy()
        predList = np.concatenate((predList, predLabels))    
        grndList = np.concatenate((grndList, labels))

    
    confMat = sklearn.metrics.confusion_matrix(grndList, predList)

    # normalize the confusion matrix
    a = confMat.sum(axis=1).reshape((-1,1))
    confMat = confMat / a

    acc_avgClass = 0
    for i in range(confMat.shape[0]):
        acc_avgClass += confMat[i,i]

    acc_avgClass /= confMat.shape[0]
    print('acc avgClass: ', "{:.1%}".format(acc_avgClass))

    breakdownResults = shot_acc(predList, grndList, np.array(new_labelList), many_shot_thr=100, low_shot_thr=20, acc_per_cls=False)
    print('Many:', "{:.1%}".format(breakdownResults[0]), 'Medium:', "{:.1%}".format(breakdownResults[1]), 'Few:', "{:.1%}".format(breakdownResults[2]))
    
    if save_dir is not None:
        log_filename = os.path.join(save_dir, 'train.log')
        fn = open(log_filename,'a')
        # fn.write('\n'.format(acc_avgClass))
        fn.write('ACC avgClass:{:.1%} \t Many: {:.1%} \t Medium: {:.1%} \t Few: {:.1%}\n'.format(acc_avgClass, *breakdownResults))
        fn.close()


def horizontal_flip_aug(model):
    def aug_model(data):
        logits = model(data)
        h_logits = model(data.flip((3,)))
        return (logits+h_logits)/2
    return aug_model

def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    # This function is excerpted from a publicly available code [commit 01e52ed, BSD 3-Clause License]
    # https://github.com/zhmiao/OpenLongTailRecognition-OLTR/blob/master/utils.py
    
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, ms.Tensor):
        preds = preds.asnumpy()
        labels = labels.asnumpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))    
 
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

# -*- coding:utf-8 -*-
from sklearn import metrics
from sklearn.metrics import classification_report
def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    pred = pred[0].cpu()
    target = target.cpu()
    recall = metrics.recall_score(target, pred, average='micro')
    f1 = metrics.f1_score(target, pred, average='weighted')

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, recall, f1

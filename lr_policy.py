# -*- coding: gbk -*-

def lr_decay(lr, n_epoch_now, n_epoch):
    if n_epoch_now > 0 and n_epoch_now % n_epoch == 0:
        return lr * 0.8
    return lr

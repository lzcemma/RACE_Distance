"""
    @file: util.py
    @author: Li Chou
    @desc: federated learning
"""

import re


## partition data into batches ##
def partition_list_by_size(_l, _n):
    return [_l[b * _n : (b + 1) * _n] for b in range((len(_l) + _n - 1) // _n)]


## get sample weight per each device (from sampled set of devices) ##
def get_norm_weights_devices(_devices):
    norm_weights = {}
    Z1 = 0.0
    device_ids = []
    for dev_i in _devices:
        id = dev_i.get_id()
        Z1 += len(dev_i.localdata_idx)
        device_ids.append(id)
    for dev_i in _devices:
        norm_weights[dev_i.get_id()] = float(len(dev_i.localdata_idx)) / Z1

    return norm_weights


def split_line(line):
    return re.findall(r"[\w']+|[.,!?;]", line)


def line_to_indices(line, word2id, max_words=25):
    unk_id = len(word2id)
    line_list = split_line(line)  # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id] * (max_words - len(indl))
    return indl


ALL_LETTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)


def word_to_indices(word, max_words=80):
    indices = []
    for c in word[:max_words]:
        indices.append(ALL_LETTERS.find(c))
    indices += [80] * (max_words - len(indices))
    return indices

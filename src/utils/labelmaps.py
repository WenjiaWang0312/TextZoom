from __future__ import absolute_import
import torch
import string


def get_vocabulary(voc_type, EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN'):
    '''
    voc_type: str: one of 'LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'
    '''
    voc = None
    types = ['digit', 'lower', 'upper', 'all']
    if voc_type == 'digit':
        voc = list(string.digits)
    elif voc_type == 'lower':
      voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'upper':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'all':
        voc = list(string.digits + string.ascii_letters + string.punctuation)
    else:
        raise KeyError('voc_type Error')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    return voc


## param voc: the list of vocabulary
def char2id(voc):
    return dict(zip(voc, range(len(voc))))


def id2char(voc):
    return dict(zip(range(len(voc)), voc))


def labels2strs(labels, id2char, char2id):
    # labels: batch_size x len_seq
    if labels.ndimension() == 1:
        labels = labels.unsqueeze(0)
    assert labels.dim() == 2
    labels = to_numpy(labels)
    strings = []
    batch_size = labels.shape[0]

    for i in range(batch_size):
        label = labels[i]
        string = []
        for l in label:
            if l == char2id['EOS']:
                break
            else:
                string.append(id2char[l])
        string = ''.join(string)
        strings.append(string)

    return strings


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

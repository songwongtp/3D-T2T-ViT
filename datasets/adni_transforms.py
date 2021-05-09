import torch

class ToTensor(object):
    def __call__(self, input):
        input = torch.from_numpy(input)
        return input
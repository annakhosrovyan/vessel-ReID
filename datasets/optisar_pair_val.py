import torch.utils.data as data
from torchvision.transforms import transforms as T

from .pretrain import Pretrain


class OptiSarPairVal(Pretrain):
    dataset_dir = "OptiSar_Pair_Val"

    def __init__(self, root="", verbose=True, pid_begin=0, **kwargs):
        super().__init__(root=root, verbose=False, pid_begin=pid_begin, **kwargs)

        if verbose:
            print("=> OptiSarPairVal Dataset loaded")
            if self.train_pair is not None:
                print("Number of RGB-SAR validation pairs: {}".format(len(self.train_pair)))
                print("  ----------------------------------------")

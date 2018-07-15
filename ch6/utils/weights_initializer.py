import torch

def xavier(m):
    """
    Initialize weights of the network using method described in Xavier Glorot, Yoshua Bengio, "Understanding the
    difficulty of trianing deep feedforward neural networks". Expected to be supplied to the `apply(...)` method on
    an instance of torch.nn.Module to be recursively applied to each module/layer present in the neural network.
    :param self:
    :param m: nn.Module
    :return:
    """
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

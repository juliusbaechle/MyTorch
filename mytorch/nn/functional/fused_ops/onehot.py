from mytorch import Tensor, eye

def onehot(tensor : Tensor, num_classes):
    output = eye(num_classes, device=tensor.device)
    return output[tensor]
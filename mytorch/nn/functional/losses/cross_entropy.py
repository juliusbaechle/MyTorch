import numpy as np
from mytorch import Tensor

def cross_entropy(logits : Tensor, targets : Tensor):
    max = np.max(logits.data, axis=-1, keepdims=True)
    exp = np.exp(logits.data - max)
    softmax = exp / exp.sum(axis=-1, keepdims=True)
    loss = np.sum(targets.data * softmax, axis=-1)
    loss = np.mean(-np.log(loss), axis=-1)

    def _cross_entropy_backward(grad):
        if logits.requires_grad:
            grad = grad * (softmax - targets.data)
            logits._add_grad(grad)

    requires_grad = logits.requires_grad and Tensor.build_graph_enabled()
    return Tensor(
        loss,
        requires_grad=requires_grad,
        grad_fn=_cross_entropy_backward if requires_grad else None,
        parents=logits if requires_grad else (),
        device=logits.device
    )
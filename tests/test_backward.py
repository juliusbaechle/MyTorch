from mytorch import Tensor, Array
import numpy as np

def test_backward():
    a = Tensor([1, 2], True, device="gpu:0")
    b = a * 2 # [2, 4]
    c = b - 1 # [1, 3]
    d = c.sum() # [4]
    d.backward()
    assert np.allclose(a.grad, Array([2, 2], "gpu:0"))
    for t in [b, c, d]: # check for memory leaks
        assert t.grad_fn is None
        assert t.grad is None

def test_backward_retain_graph():
    a = Tensor([1, 2], True, device="gpu:0")
    b = a * 2 # [2, 4]
    c = b - 1 # [1, 3]
    d = c.sum() # [4]
    d.backward(retain_graph=True)
    assert np.allclose(a.grad, Array([2, 2], "gpu:0"))
    for t in [b, c, d]:
        assert t.grad_fn is not None
        assert t.grad is not None
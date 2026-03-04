from mytorch.tensor import Tensor
from mytorch.array import Array
import numpy as np
import mytorch

def test_properties():
    tensor = Tensor([1, 2, 3], True, device="gpu:0")
    assert tensor.dtype == "float32"
    assert np.allclose(tensor.data, Array([1, 2, 3], "gpu:0"))
    assert tensor.device == "gpu:0"
    assert tensor.shape == (3,)
    assert tensor.ndim == 1
    assert tensor.is_leaf == True

def test_add():
    t1 = Tensor([1, 2, 3], True)
    t2 = Tensor([1], True)
    res = t1 + t2
    assert np.allclose(res.data, Array([2, 3, 4]))
    res.grad_fn(Array([1, 1, 1]))
    assert np.allclose(t2.grad, Array([3]))
    assert np.allclose(t1.grad, Array([1, 1, 1]))

def test_mul():
    t1 = Tensor([1, 2, 3], True)
    t2 = Tensor([2], True)
    res = t1 * t2
    assert np.allclose(res.data, Array([2, 4, 6]))
    res.grad_fn(Array([1, 1, 1]))
    assert np.allclose(t1.grad, Array([2, 2, 2]))
    assert np.allclose(t2.grad, Array([6]))

def test_matmul():
    t1 = Tensor([[1, 2]], True)
    t2 = Tensor([[1, 2], [1, 2]], True)
    res = t1 @ t2
    assert np.allclose(res.data, Array([[3, 6]]))
    res.grad_fn(Array([[1, 1]]))
    assert np.allclose(t1.grad, Array([[3, 3]]))
    assert np.allclose(t2.grad, Array([[1, 1], [2, 2]]))

def test_truediv():
    t1 = Tensor([2, 4, 6], True)
    t2 = Tensor([2], True)
    res = t1 / t2
    assert np.allclose(res.data, Array([1, 2, 3]))
    res.grad_fn(Array([1, 1, 1]))
    assert np.allclose(t1.grad, Array([0.5, 0.5, 0.5]))
    assert np.allclose(t2.grad, Array([-3]))

def test_pow():
    t1 = Tensor([np.e, np.e], True)
    t2 = Tensor(2, True)
    res = t1 ** t2
    assert np.allclose(res.data, Array([np.e**2, np.e**2]))
    res.grad_fn(Array([1, 1]))
    assert np.allclose(t1.grad, Array([2*np.e, 2*np.e]))
    assert np.allclose(t2.grad, Array([2*np.e**2]))

def test_exp():
    t1 = Tensor([1, 2], True)
    res = t1.exp()
    assert np.allclose(res.data, Array([np.e, np.e**2]))
    res.grad_fn(Array([1, 1]))
    assert np.allclose(t1.grad, Array([np.e, np.e**2]))

def test_log():
    t1 = Tensor([np.e, np.e**2], True)
    res = t1.log()
    assert np.allclose(res.data, Array([1, 2]))
    res.grad_fn(Array([1, 1]))
    assert np.allclose(t1.grad, Array([1/np.e, 1/np.e**2]))

def test_abs():
    t1 = Tensor([-3, 0, 3], True)
    res = t1.abs()
    assert np.allclose(res.data, Array([3, 0, 3]))
    res.grad_fn(Array([1, 1, 1]))
    assert np.allclose(t1.grad, Array([-1, 0, 1]))

def test_clamp():
    t1 = Tensor([-3, 0, 3], True)
    res = t1.clamp(0, 2)
    assert np.allclose(res.data, Array([0, 0, 2]))
    res.grad_fn(Array([1, 1, 1]))
    assert np.allclose(t1.grad, Array([0, 1, 0]))

def test_sin():
    t1 = Tensor([0, np.pi/2], True)
    res = t1.sin()
    assert np.allclose(res.data, Array([0, 1]), atol=1e-5)
    res.grad_fn(Array([1, 1]))
    assert np.allclose(t1.grad, Array([1, 0]), atol=1e-5)

def test_cos():
    t1 = Tensor([0, np.pi/2], True)
    res = mytorch.cos(t1)
    assert np.allclose(res.data, Array([1, 0]), atol=1e-5)
    res.grad_fn(Array([1, 1]))
    assert np.allclose(t1.grad, Array([0, -1]), atol=1e-5)

def test_tan():
    t1 = Tensor([0, np.pi/4], True)
    res = mytorch.tan(t1)
    assert np.allclose(res.data, Array([0, 1]))
    res.grad_fn(Array([1, 1]))
    assert np.allclose(t1.grad, Array([1, 2]))

def test_compare_equal():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([1, 0, 3])
    assert t1 == t2

def test_getitem():
    t1 = Tensor([[1, 2], [3, 4]], True)
    res = t1[0, 1]
    assert np.allclose(res.data, Array([2]))
    res.grad_fn(Array([1]))
    assert np.allclose(t1.grad, Array([[0, 1], [0, 0]]))

def test_transpose():
    t1 = Tensor([[1, 2], [3, 4]], True)
    res = t1.transpose()
    assert np.allclose(res.data, Array([[1, 3], [2, 4]]))
    res.grad_fn(Array([[1, 2], [3, 4]]))
    assert np.allclose(t1.grad, Array([[1, 3], [2, 4]]))

def test_permute():
    t1 = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], True)
    res = t1.permute(2, 0, 1)
    assert np.allclose(res.data, Array([[[1, 3], [5, 7]], [[2, 4], [6, 8]]]))
    res.grad_fn(Array([[[1, 3], [5, 7]], [[2, 4], [6, 8]]]))
    assert np.allclose(t1.grad, Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

def test_reshape():
    t1 = Tensor([[1, 2], [3, 4]], True)
    res = t1.reshape(4)
    assert np.allclose(res.data, Array([1, 2, 3, 4]))
    res.grad_fn(Array([1, 2, 3, 4]))
    assert np.allclose(t1.grad, Array([[1, 2], [3, 4]]))

def test_flatten():
    t1 = Tensor([[1, 2], [3, 4]], True)
    res = t1.flatten()
    assert np.allclose(res.data, Array([1, 2, 3, 4]))

def test_chunk():
    t1 = Tensor([[1, 2], [3, 4]], True)
    res0, res1 = t1.chunk(2, dim=0)
    assert np.allclose(res0.data, Array([[1, 2]]))
    assert np.allclose(res1.data, Array([[3, 4]]))
    res0.grad_fn(Array([[1, 2]]))
    res1.grad_fn(Array([[3, 4]]))
    assert np.allclose(t1.grad, Array([[1, 2], [3, 4]]))

def test_sum():
    t1 = Tensor([[1, 2], [3, 4]], True)
    res = t1.sum()
    assert np.allclose(res.data, Array([10]))
    res.grad_fn(Array([1]))
    assert np.allclose(t1.grad, Array([[1, 1], [1, 1]]))

def test_cumsum():
    t1 = Tensor([[1, 2], [3, 4]], True)
    res = t1.cumsum(dim=0)
    assert np.allclose(res.data, Array([[1, 2], [4, 6]]))
    res.grad_fn(Array([[1, 2], [1, 2]]))
    assert np.allclose(t1.grad, Array([[2, 4], [1, 2]]))

def test_mean():
    t1 = Tensor([[1, 2], [3, 4]], True)
    res = t1.mean()
    assert np.allclose(res.data, Array([2.5]))
    res.grad_fn(Array([1], dtype="float32"))
    print(t1.grad)
    assert np.allclose(t1.grad, Array([[0.25, 0.25], [0.25, 0.25]]))

def test_var():
    t1 = Tensor([[1, 2], [3, 4]], True)
    res = t1.var()
    assert np.allclose(res.data, Array([1.25]))
    res.grad_fn(Array([1], dtype="float32"))
    assert np.allclose(t1.grad, Array([[-0.75, -0.25], [0.25, 0.75]]))

def test_max():
    t1 = Tensor([[1, 2], [3, 4]], True)
    res = t1.max(dim=1)
    assert np.allclose(res.data, Array([2, 4]))
    assert not np.allclose(res.data, Array([1, 2]))
    res.grad_fn(Array([1, 1]))
    assert np.allclose(t1.grad, Array([[0, 1], [0, 1]]))

def test_max_scalar():
    t1 = Tensor([[1, 2], [3, 4]], True)
    res = t1.max()
    assert np.allclose(res.data, Array([4]))
    res.grad_fn(Array([1]))
    assert np.allclose(t1.grad, Array([[0, 0], [0, 1]]))

def test_argmax():
    t1 = Tensor([[1, 2], [3, 4]], True)
    res = t1.argmax(dim=1)
    assert np.allclose(res.data, Array([1, 1]))

def test_masked_fill():
    t1 = Tensor([[1, 2], [3, 4]], True)
    mask = Tensor([[True, False], [False, True]])
    res = t1.masked_fill(mask, 0)
    assert np.allclose(res.data, Array([[0, 2], [3, 0]]))
    res.grad_fn(Array([[1, 1], [1, 1]]))
    assert np.allclose(t1.grad, Array([[0, 1], [1, 0]]))

def test_sort():
    t1 = Tensor([[3, 1], [4, 2]], True)
    res = t1.sort(dim=1)
    assert np.allclose(res.data, Array([[1, 3], [2, 4]]))
    res.grad_fn(Array([[0, 1], [0, 1]]))
    assert np.allclose(t1.grad, Array([[1, 0], [1, 0]]))

def test_concat():
    t1 = Tensor([[1, 2]], True)
    t2 = Tensor([[3, 4]], True)
    res = mytorch.concatenate([t1, t2], dim=0)
    assert np.allclose(res.data, Array([[1, 2], [3, 4]]))
    res.grad_fn(Array([[1, 1], [2, 2]]))
    assert np.allclose(t1.grad, Array([[1, 1]]))
    assert np.allclose(t2.grad, Array([[2, 2]]))

def test_stack():
    t1 = Tensor([1, 2], True)
    t2 = Tensor([3, 4], True)
    res = mytorch.stack([t1, t2], dim=0)
    assert np.allclose(res.data, Array([[1, 2], [3, 4]]))
    res.grad_fn(Array([[1, 1], [2, 2]]))
    assert np.allclose(t1.grad, Array([1, 1]))
    assert np.allclose(t2.grad, Array([2, 2]))

def test_zeros():
    res = mytorch.zeros((2, 3), device="cpu")
    assert np.allclose(res.data, Array([[0, 0, 0], [0, 0, 0]]))
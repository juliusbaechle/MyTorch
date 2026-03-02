from mytorch.array import Array
import numpy as np
import dpnp as dp
import dpctl
import pytest

def test_init_cpu():
    array = Array([], device="cpu")
    assert(array.device == "cpu:0")

def test_init_gpu():
    array = Array([], device="gpu:1")
    assert(array.device == "gpu:1")

def test_init__with_dpnp_array():
    dp_arr = dp.array([1, 2, 3])
    array = Array(dp_arr)
    assert(array.device == "gpu:0")

def test_init__with_np_array():
    dp_arr = np.array([1, 2, 3])
    array = Array(dp_arr)
    assert(array.device == "cpu:0")

def test_init__with_array():
    array1 = Array([1.0], device="gpu:0")
    array2 = Array(array1)
    assert (array2.device == "gpu:0")

def test_init__with_sycl_device():
    device = dpctl.SyclDevice("gpu:1")
    array = Array([], device=device)
    assert (array.device == "gpu:1")

def test_init__convert_to_float32():
    dp_arr = dp.array([1, 2, 3], device="cpu")
    array = Array(dp_arr, dtype="float32")
    assert(array.device == "cpu:0")

def test_to_device():
    array = Array([1, 2, 3], dtype="float32")
    assert(array.device == "cpu:0")
    array = array.to("gpu:0")
    assert(array.device == "gpu:0")

def test_print():
    array = Array([1.0, 2.0, 3.0])
    assert array.__repr__() == "Array([1.00000 2.00000 3.00000], dtype=float32, device='cpu:0')"

def test_concat():
    array1 = Array([1, 2, 3])
    array2 = Array([4, 5, 6])
    array3 = np.concatenate([array1, array2])
    expected = Array([1, 2, 3, 4, 5, 6])
    assert np.allclose(array3, b=expected)

def test_concat_multiple_devices():
    array1 = Array([1, 2, 3], "gpu:0")
    array2 = Array([4, 5, 6], "gpu:1")
    with pytest.raises(RuntimeError):
        np.concatenate([array1, array2])

def test_add():    
    array1 = Array([1, 2, 3], "gpu:0", "int32")
    array2 = Array([1, 2, 3], "gpu:0", "int32")
    result = array1 + array2
    assert(np.allclose(result, Array([2, 4, 6], "gpu:0", "int32")))

def test_iadd():
    array = Array([1, 2, 3], "gpu:0", "int32")
    array += dp.array([1, 2, 3], device="gpu:0")
    assert(np.allclose(array, Array([2, 4, 6], "gpu:0", "int32")))

def test_neg():    
    array = Array([1, 2, 3], "gpu:0", "int32")
    assert(np.allclose(-array, Array([-1, -2, -3], "gpu:0", "int32")))

def test_log():    
    array = Array([10, 100, 1000], "gpu:0")
    result = np.log10(array)
    assert(np.allclose(result, Array([1, 2, 3], "gpu:0")))

def test_getitem():    
    array = Array([10, 100, 1000], "gpu:0")
    assert(array[1] == 100)

#def test_throughput():    
#    arr1 = Array.rand((1000, 100000), device="gpu:0")
#    arr2 = Array.rand((100000, 1000), device="gpu:0")
#    for i in range(100):
#        arr = arr1 @ arr2
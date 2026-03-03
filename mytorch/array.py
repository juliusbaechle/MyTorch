from typing import Any
import dpnp as dp
import numpy as np # for array2string
import dpctl

class Array:
    _binary_ufuncs = {
        "__add__": "add", "__radd__": "add",
        "__sub__": "subtract", "__rsub__": "subtract",
        "__mul__": "multiply", "__rmul__": "multiply",
        "__truediv__": "true_divide", "__rtruediv__": "true_divide",
        "__floordiv__": "floor_divide", "__rfloordiv__": "floor_divide",
        "__matmul__": "matmul", "__rmatmul__": "matmul",
        "__pow__": "power", "__rpow__": "power",
        "__mod__": "remainder", "__rmod__": "remainder",
        "__and__": "bitwise_and", "__rand__": "bitwise_and",
        "__or__": "bitwise_or", "__ror__": "bitwise_or",
        "__xor__": "bitwise_xor", "__rxor__": "bitwise_xor",
        "__lt__": "less", "__le__": "less_equal",
        "__gt__": "greater", "__ge__": "greater_equal",
        "__eq__": "equal", "__ne__": "not_equal",
    }

    _inplace_ops = {
        "__iadd__": "add",
        "__isub__": "subtract",
        "__imul__": "multiply",
        "__itruediv__": "true_divide",
        "__ifloordiv__": "floor_divide",
        "__imatmul__": "matmul",
        "__ipow__": "power",
        "__imod__": "remainder",
        "__iand__": "bitwise_and",
        "__ior__": "bitwise_or",
        "__ixor__": "bitwise_xor",
    }

    _unary_ufuncs = {
        "__neg__": "negative",
        "__pos__": "positive",
        "__abs__": "absolute",
        "__invert__": "invert",
    }

    def __init__(self, data, device : str | dpctl.SyclDevice = None, dtype : str = None):
        src_dtype = getattr(data, "dtype", "float32")
        if dtype is None:
            tgt_dtype = src_dtype
            if src_dtype == "float64":
                tgt_dtype = "float32"
            if src_dtype == "int64":
                tgt_dtype = "int32"
        else:
            tgt_dtype = dtype

        if isinstance(data, dp.ndarray):
            self._array = data.astype(tgt_dtype)
        elif isinstance(data, Array):
            self._array = data._array.astype(tgt_dtype)
        else:
            self._array = dp.array(data, device="cpu", dtype=tgt_dtype)

        src_dev = getattr(data, "device", "cpu")
        if isinstance(src_dev, dpctl.tensor.Device):
            src_dev = src_dev.sycl_device.filter_string
        
        tgt_dev = device if device is not None else src_dev
        if isinstance(device, dpctl.SyclDevice):
            tgt_dev = device.filter_string
        
        if src_dev != tgt_dev:
            self._array = dp.asarray(self._array, device=dpctl.SyclDevice(tgt_dev))
    
    def to(self, device : str):
        if self.device == device: 
            return self
        
        device = dpctl.SyclDevice(device)
        arr = dp.asarray(self._array, device=device)
        return Array(arr, device, dtype=self.dtype)
    
    @property
    def device(self):
        s = self._array.device.sycl_device.filter_string
        return s[s.index(':') + 1:] 
    
    @property
    def dtype(self):
        return self._array.dtype
    
    @property
    def shape(self):
        return self._array.shape
    
    @property
    def ndim(self):
        return self._array.ndim
    
    @property
    def size(self):
        return self._array.size
    
    @property
    def T(self):
        return Array(self._array.T)
    
    def _coerce_other(self, other):
        if isinstance(other, Array):
            return other._array, other.device
        if isinstance(other, dp.ndarray):
            dev = other.device.sycl_device.filter_string
            return other, dev[dev.index(':') + 1:]
        return other, None

    @staticmethod
    def _make_binary_op(ufunc_name : str, reflect=False):
        def op(self : Array, other : Array | dp.ndarray):
            other_arr, other_dev = self._coerce_other(other)
            if other_dev is not None and other_dev != self.device:
                raise RuntimeError(f"Expected all arrays to be on the "
                 f"same device, but found at least two devices, "
                 f"{self.device} and {other_dev}!")
            
            func = getattr(dp, ufunc_name)
            if reflect:
                result = func(other_arr, self._array)
            else:
                result = func(self._array, other_arr)

            return Array(result, device=self.device)
        return op

    @staticmethod
    def _make_unary_op(ufunc_name):
        def op(self : Array):
            func = getattr(dp, ufunc_name)
            result = func(self._array)
            return Array(result, device=self.device)
        return op

    @staticmethod
    def _make_inplace_op(ufunc_name):
        def op(self : Array, other):
            other_arr, other_dev = self._coerce_other(other)            
            if other_dev is not None and other_dev != self.device:
                raise RuntimeError(f"Expected all arrays to be on the "
                 f"same device, but found at least two devices, "
                 f"{self.device} and {other_dev}!")
            
            func = getattr(dp, ufunc_name)
            func(self._array, other_arr, out=self._array)
            return self
        return op

    def __len__(self):
        return len(self._array)
    
    def __repr__(self):
        s = np.array2string(
            self._array.asnumpy(), 
            separator=" ",
            precision=5,
            floatmode="fixed",
            max_line_width=80
        )

        lines = s.split("\n")
        if len(lines) > 1:
            indent = " " * len("Array(")
            s = lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])

        device_info = f", device='{self.device}'"

        return f"Array({s}, dtype={self.dtype}{device_info})"
    
    def astype(self, dtype):
        if self.dtype != dtype:
            self._array = self._array.astype(dtype)
        return self
    
    def asnumpy(self):
        return self._array.asnumpy()
    
    def __array_function__(self, func, types, args, kwargs):
        if not all(issubclass(t, Array) for t in types):
            return NotImplemented
        
        devices = set()
        def unpack_dp_array(x):
            if isinstance(x, Array):
                devices.add(x.device)
                return x._array
            elif isinstance(x, (list, tuple)):
                return type(x)(unpack_dp_array(y) for y in x)
            elif isinstance(x, dict):
                return {k: unpack_dp_array(v) for k,v in x.items()}
            else:
                return x
            
        unpacked_args = unpack_dp_array(args)
        unpacked_kwargs = unpack_dp_array(kwargs)

        if len(devices) > 1:
            raise RuntimeError(f"All inputs must be on the same device, found: {devices}")
        
        dp_func = getattr(dp, func.__name__, None)
        if dp_func == None:
            raise NotImplemented

        device = devices.pop() if devices else self.device
        result = dp_func(*unpacked_args, **unpacked_kwargs)
        if isinstance(result, (dp.ndarray, np.ndarray)) and result.size == 1:
            return result.asnumpy().item()
        if isinstance(result, (dp.ndarray, np.ndarray)):
            return Array(result, device=device)
        return result
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        arrays = []
        devices = set()

        for x in inputs:
            if isinstance(x, Array):
                arrays.append(x._array)
                devices.add(x.device)
            else:
                arrays.append(x)
                if isinstance(x, dp.ndarray):
                    devices.add(x.device.sycl_device.filter_string)

        if len(devices) > 1:
            raise RuntimeError(f"All inputs must be on the same device, found: {devices}")
        
        func = getattr(getattr(dp, ufunc.__name__, None), method, None)
        if func == None:
            raise NotImplementedError()

        device = devices.pop() if devices else self.device
        result = func(*arrays, **kwargs)
        if isinstance(result, (dp.ndarray, np.ndarray)) and result.size == 1:
            return result.asnumpy().item()
        if isinstance(result, (dp.ndarray, np.ndarray)):
            return Array(result, device=device)
        return result
    
    def __setitem__(self, idx, value):
        if isinstance(value, Array):
            value = value._array
        self._array[idx] = value
    
    def __getattr__(self, name):
        if hasattr(self._array, name):
            return getattr(self._array, name)
        raise AttributeError(f"'Array' object has no attribute '{name}'")
    
    def __getitem__(self, idx):
        def coerce_index(idx):
            if isinstance(idx, tuple):
                return tuple(coerce_index(i) for i in idx)
            if isinstance(idx, Array):
                return idx._array
            if hasattr(idx, "data") and isinstance(idx.data, Array):
                return idx.data._array
            return idx
        idx = coerce_index(idx)
        return self._array[idx]

    @staticmethod
    def zeros(shape, device="cpu", dtype="float32"):
        return Array(dp.zeros(shape, dtype=dtype, device=device))
    
    @staticmethod
    def ones(shape, device="cpu", dtype="float32"):
        return Array(dp.ones(shape, dtype=dtype, device=device))
    
    @staticmethod
    def empty(shape, device="cpu", dtype="float32"):
        return Array(dp.empty(shape, dtype=dtype, device=device))

    @staticmethod
    def full(shape, fill_value, device="cpu", dtype="float32"):
        return Array(dp.full(shape, fill_value, dtype=dtype, device=device))
        
    @staticmethod
    def arange(start, end=None, step=1, device="cpu", dtype="float32"):
        return Array(dp.arange(start, end, step, dtype=dtype, device=device))

    @staticmethod
    def linspace(start, end=None, num=50, device="cpu", dtype="float32"):
        return Array(dp.linspace(start, end, num, dtype=dtype, device=device))

    @staticmethod
    def eye(N, M=None, k=0, device="cpu", dtype="float32"):
        return Array(dp.eye(N, M, k, dtype=dtype, device=device))

    @staticmethod
    def randn(shape, device="cpu", dtype="float32"):
        return Array(dp.random.randn(*shape), dtype=dtype, device=device)
    
    @staticmethod
    def rand(shape, device="cpu", dtype="float32"):
        return Array(dp.random.rand(*shape), dtype=dtype, device=device)
    
    @staticmethod
    def randint(low, high, shape, device="cpu", dtype="int32"):
        return Array(dp.random.randint(low, high, size=shape, dtype=dtype), device=device)

    @staticmethod
    def tril(x, k=0, device="cpu", dtype="float32"):
        return Array(dp.tril(x, k=k), device=device, dtype=dtype)
    
    @staticmethod
    def triu(x, k=0, device="cpu", dtype="float32"):
        return Array(dp.triu(x, k=k), device=device, dtype=dtype)
    
    @staticmethod
    def zeros_like(other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or other.dtype
        return Array(dp.zeros_like(other._array, device=device, dtype=dtype))
    
    @staticmethod
    def ones_like(other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or other.dtype
        return Array(dp.ones_like(other._array, device=device, dtype=dtype))
    
    @staticmethod
    def empty_like(other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or other.dtype
        return Array(dp.empty_like(other._array, device=device, dtype=dtype))
    
    @staticmethod
    def full_like(other, fill_value, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or other.dtype
        return Array(dp.full_like(other._array, fill_value, device=device, dtype=dtype))

    @staticmethod
    def randn_like(other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or other.dtype
        return Array(dp.randn_like(other._array, device=device, dtype=dtype))

    @staticmethod
    def rand_like(other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or other.dtype
        return Array(dp.rand_like(other._array, device=device, dtype=dtype))


# Attach binary, unary, and inplace operations
for dunder, bfunc in Array._binary_ufuncs.items():
    reflect = dunder.startswith("__r")
    setattr(Array, dunder, Array._make_binary_op(bfunc, reflect=reflect))

for dunder, ufunc in Array._unary_ufuncs.items():
    setattr(Array, dunder, Array._make_unary_op(ufunc))

for dunder, ifunc in Array._inplace_ops.items():
    setattr(Array, dunder, Array._make_inplace_op(ifunc))
import numpy as np
import dpnp as dp
from mytorch import Tensor, Array

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self._persistent_buffers = set()
        self.training = True

    def __setattr__(self, name, value):
        if "_modules" not in self.__dict__:
            if isinstance(value, (Module, Tensor)):
                raise RuntimeError(
                    f"Cannot assign {type(value).__name__} to '{name}' "
                    "before calling supr().__init__() in your Module subclass" 
                )
            return object.__setattr__(self, name, value)
        
        if isinstance(value, Tensor):
            self._parameters[name] = value
        
        if isinstance(value, Module):
            self._modules[name] = value

        return object.__setattr__(self, name, value)

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix=""):
        for name, param in self._parameters.items():
            yield prefix + name, param
        for name, module in self._modules.items():
            yield from module.named_parameters(f"{prefix}{self.__name__}.")

    def register_buffer(self, name, tensor, persistent=True):
        if not isinstance(tensor, Tensor):
            raise TypeError("Buffers must be Tensors")
        
        self._buffers[name] = tensor
        if persistent:
            self._persistent_buffers.add(name)

        object.__setattr__(self, name, tensor)

    def named_buffers(self, prefix="", persistent_only=False):
        for name, param in self._buffers.items():
            if not persistent_only or name in self._persistent_buffers:
                yield f"{prefix}{name}", param
        for name, module in self._modules.items():
            yield from module.named_buffers(f"{prefix}{name}.", persistent_only)

    def to(self, device):
        for name, param in self._parameters.items():
            self._parameters[name] = param.to(device)
            object.__setattr__(self, name, self._parameters[name])
        for name, param in self._buffers.items():
            self._buffers[name] = param.to(device)
            object.__setattr__(self, name, self._buffers[name])
        for module in self._modules.values():
            module.to(device)
        return self
    
    def apply(self, fn):
        fn(self)
        for module in self._modules.values():
            module.apply(fn)
        return self
    
    def _extra_repr(self):
        return ""

    def _repr(self, indent=0):
        model_name = self.__class__.__name__
        ind = "    " * indent
        extra = self._extra_repr()
        if not self._modules:
            return f"{ind}{model_name}({extra})\n"
        s = f"{ind}{model_name}(\n"
        for key, val in self._modules.items():
            s += f"{ind}  ({key}): {val._repr(indent + 1).lstrip()}"
        s += f"{ind})\n"
        return s
    
    def __repr__(self):
        return self._repr(indent=0).rstrip()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def state_dict(self):
        state = {}
        for name, param in self.named_parameters():
            state[name] = param.numpy()
        for name, buffer in self.named_buffers():
            state[name] = buffer.numpy()
        return state

    def load_state_dict(self, state_dict, strict=True, device="cpu"):
        missing_keys = []
        unexpected_keys = list(state_dict.keys())

        for name, param in self.named_parameters():
            if name in state_dict:
                try:
                    param.data = Array(state_dict[name], device)
                except:
                    print(f"Failed to load {name}. Expected {param.shape}, got {state_dict[name].shape}")
                    continue
                unexpected_keys.remove(name)
            else:
                missing_keys.append(name)

        for name, buffer in self.named_buffers():
            if name in state_dict:
                try:
                    buffer.data = Array(state_dict[name], device)
                except:
                    print(f"Failed to load {name}. Expected {buffer.shape}, got {state_dict[name].shape}")
                    continue
                unexpected_keys.remove(name)
            else:
                missing_keys.append(name)

        if strict:
            error_msgs = []
            if missing_keys:
                error_msgs.append(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                error_msgs.append(f"Unexpected keys: {unexpected_keys}")
            if error_msgs:
                raise RuntimeError("Error(s) in loading state_dict:\n" + "\n".join(error_msgs))
            else:
                return "<All Keys Matched Successfully>"
        else:
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                return "<All Keys Matched Successfully>"
            else:
                return missing_keys, unexpected_keys

    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()

    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()
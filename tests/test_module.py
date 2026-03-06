import mytorch.nn as nn
from mytorch import Tensor, Array

class FancyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.params1 = Tensor([1, 2], True, "gpu:0")
        self.register_buffer("buffer", Tensor([3, 4], True, "gpu:0"), False)        

def test_parameters():
    fancy = FancyLayer()
    assert len(list(fancy.parameters())) == 1
    assert len(list(fancy.named_parameters())) == 1

def test_buffers():
    fancy = FancyLayer()
    assert len(list(fancy.named_buffers())) == 1
    assert len(list(fancy.named_buffers("", True))) == 0

def test_print_module():
    assert FancyLayer().__repr__() == "FancyLayer()"
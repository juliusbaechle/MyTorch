from .tensor import Tensor
import numpy as np

def tensor2string(input : Tensor):
    data = input.data._array
    data_info = np.array2string(
        data.asnumpy(),
        separator=" ",
        precision=5,
        floatmode="fixed",
        max_line_width=80
    )

    lines = data_info.split("\n")
    if len(lines) > 1:
        indent = " " * len("tensor(")
        data_info = lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])

    grad_info = ""
    if input.requires_grad:
        if input.grad_fn is not None:
            grad_info = f", grad_fn={input.grad_fn.__name__}"
        else:
            grad_info = ", requires_grad=True"

    device_info = f", device={input.device}"

    return f"tensor({data_info}{grad_info}{device_info})"
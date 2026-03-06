from mytorch import Tensor

def linear(input, weight : Tensor, bias : Tensor = None) -> Tensor:
    w_T = weight.transpose()
    output = input @ w_T
    if bias is not None:
        output = output + bias
    return output
from mytorch import Tensor

def linear(input : Tensor, weight : Tensor, bias : Tensor = None) -> Tensor:
    output = input.data @ weight.data.T
    if bias is not None:
        output += bias.data
    
    def _linear_backward(grad):
        if weight.requires_grad:
            weight_grad = (input.data.T @ grad).T
            weight._add_grad(weight_grad)
        if bias.requires_grad:
            bias_grad = grad.sum(axis=0)
            bias._add_grad(bias_grad)
        if input.requires_grad:
            grad_input = grad @ weight.data
            input._add_grad(grad_input)

    requires_grad = (weight.requires_grad or input.requires_grad or (bias is not None and bias.requires_grad)) and Tensor.build_graph_enabled()
    return Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_linear_backward if requires_grad else None,
        parents=(input, weight, bias) if requires_grad else (),
        device=input.device
    )
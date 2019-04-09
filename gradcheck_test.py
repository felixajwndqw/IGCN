from torch.autograd import gradcheck
import torch
from igcn import GaborFunction

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
gabor = GaborFunction.apply
input = (
            torch.randn(4, 3, 10, 10, dtype=torch.double, requires_grad=True),
            torch.ones(2, 4, dtype=torch.double, requires_grad=True)
        )
test = gradcheck(gabor, input, eps=1e-6, atol=1e-4)
print(test)
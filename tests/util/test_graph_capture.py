import torch
from e3nn.o3 import Norm, Irreps

def test_submod_tracing() -> None:
    """Check that tracing actually occurs"""

    class MyModule(torch.nn.Module):
        def __init__(self, irreps_in):
            super().__init__()
            self.norm = Norm(irreps_in)

        def forward(self, x):
            norm = self.norm(x)
            if torch.any(norm > 7.):
                return norm
            else:
                return norm * 0.5

    irreps = Irreps("2x0e + 1x1o")
    mod = MyModule(irreps)

    torch._logging.set_logs(bytecode=True, graph=True)

    x = irreps.randn(2, -1)
    print('before: ', x)
    mod = torch.compile(mod)
    print('after: ', mod(x))
    print('\n')

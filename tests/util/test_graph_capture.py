import torch
from e3nn.o3 import Norm, Irreps
from e3nn.util.test import assert_no_graph_break

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


    x = irreps.randn(2, -1)
    assert_no_graph_break(mod)
    # print('before: ', x)
    # explanation = torch._dynamo.explain(mod,x)
    # print(explanation)
    # torch._logging.set_logs(bytecode=True, graph=True)
    # new_mod = torch.compile(mod)


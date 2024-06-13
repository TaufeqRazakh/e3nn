import pytest

import torch

from e3nn import o3
from e3nn.util.test import random_irreps, assert_no_graph_break

@pytest.mark.parametrize("irreps_in", ["", "5x0e", "1e + 2e + 4x1e + 3x3o"] + random_irreps(n=4))
@pytest.mark.parametrize("squared", [True, False])
def test_norm_compilations(irreps_in, squared) -> None:
    """Check whether norm compiles without graph breaks"""

    mod = o3.Norm(irreps_in, squared=squared)
    assert_no_graph_break(mod)
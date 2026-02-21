import torch
from torch.autograd import gradcheck
from functional import MHAFunction


def test_mha_gradcheck():
    B, S, Nh, D = 2, 4, 2, 8
    qry = torch.randn(B, S, Nh, D, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, S, Nh, D, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, S, Nh, D, dtype=torch.float64, requires_grad=True)
    key_pad_mask = torch.zeros(B, S, dtype=torch.bool)

    assert gradcheck(MHAFunction.apply, (qry, k, v, key_pad_mask), eps=1e-6, atol=1e-4)
    print("MHAFunction gradcheck passed")

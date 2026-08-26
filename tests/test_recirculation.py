import torch
from torch import nn

from korean_math_tdcs.recirculation.fixed import FixedRecirculation


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(3, 3, bias=False) for _ in range(3)])
        for layer in self.model.layers:
            nn.init.eye_(layer.weight)

    def forward(self, value):
        for layer in self.model.layers:
            value = layer(value)
        return value


def test_alpha_zero_is_exact_noop():
    model = ToyModel()
    first = torch.tensor([[[1.0, 2.0, 3.0]]])
    second = torch.tensor([[[3.0, 2.0, 1.0]]])
    expected_first, expected_second = model(first), model(second)
    with FixedRecirculation(model, source_layer=2, destination_layer=0, alpha=0.0):
        actual_first, actual_second = model(first), model(second)
    torch.testing.assert_close(actual_first, expected_first)
    torch.testing.assert_close(actual_second, expected_second)

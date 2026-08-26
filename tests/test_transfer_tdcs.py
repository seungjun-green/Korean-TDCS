import numpy as np
import pytest
import torch

from korean_math_tdcs.training.tdcs_trainer import tdcs_probabilities
from korean_math_tdcs.training.transfer import relative_transfer_from_vectors


def test_relative_transfer_orientation_and_diagonal():
    matrix = relative_transfer_from_vectors(
        [torch.tensor([1.0, 0.0]), torch.tensor([2.0, 1.0])]
    )
    np.testing.assert_allclose(np.diag(matrix), [1.0, 1.0])
    assert matrix[0, 1] == 2.0  # training on j=2 affects i=1
    assert matrix[1, 0] == 0.4  # the denominator belongs to affected row i=2


def test_tdcs_probability_simplex_and_previous_only_replay():
    transfer = np.eye(5)
    transfer[0, 2] = 0.1
    transfer[1, 2] = 0.2
    probabilities = tdcs_probabilities(transfer, current_level=3)
    assert np.all(probabilities >= 0)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[0] > 0 and probabilities[1] > 0 and probabilities[2] > 0
    assert probabilities[3] == 0 and probabilities[4] == 0


def test_harder_adjustment_reads_current_row():
    transfer = np.eye(5)
    transfer[0, 3] = 0.9
    probabilities = tdcs_probabilities(transfer, current_level=1, harder_fraction=0.2)
    assert probabilities[3] == pytest.approx(0.2)
    assert probabilities[0] == pytest.approx(0.8)

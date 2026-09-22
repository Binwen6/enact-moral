from __future__ import annotations

import unittest

import torch

from interventions import orthonormal_basis, remove_projection, replace_projection


class ProjectionTests(unittest.TestCase):
    def test_basis_is_orthonormal(self) -> None:
        vectors = torch.tensor([[1.0, 1.0], [1.0, -1.0], [0.0, 0.0]])
        basis = orthonormal_basis(vectors)
        torch.testing.assert_close(basis.T @ basis, torch.eye(2))

    def test_projection_replacement_preserves_orthogonal_component(self) -> None:
        basis = torch.tensor([[1.0], [0.0], [0.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])
        donor = torch.tensor([[5.0, 8.0, 9.0]])
        replaced = replace_projection(target, donor, basis)
        torch.testing.assert_close(replaced, torch.tensor([[5.0, 2.0, 3.0]]))

    def test_projection_removal_is_orthogonal(self) -> None:
        basis = torch.tensor([[0.0], [1.0], [0.0]])
        states = torch.tensor([[1.0, 2.0, 3.0], [4.0, -2.0, 6.0]])
        removed = remove_projection(states, basis)
        torch.testing.assert_close(removed @ basis, torch.zeros(2, 1))


if __name__ == "__main__":
    unittest.main()

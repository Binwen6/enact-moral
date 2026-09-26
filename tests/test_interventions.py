from __future__ import annotations

import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from interventions import LastTokenWrite, orthonormal_basis, remove_projection, replace_projection


def small_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
    )
    model = Qwen2ForCausalLM(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class LastTokenWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = small_model()
        self.inputs = torch.tensor([[3, 5, 7, 9], [2, 4, 6, 8]])

    def capture(self, tap: LastTokenWrite, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tap.mode = "capture"
        with torch.no_grad():
            logits = self.model(input_ids=inputs).logits[:, -1]
        tap.mode = "off"
        return tap.captured, logits

    def test_writing_own_state_is_identity(self) -> None:
        with LastTokenWrite(self.model, 2) as tap:
            state, clean = self.capture(tap, self.inputs)
            tap.mode = "write"
            tap.source = state
            with torch.no_grad():
                written = self.model(input_ids=self.inputs).logits[:, -1]
        torch.testing.assert_close(written, clean)

    def test_basis_write_changes_only_projection(self) -> None:
        basis = orthonormal_basis(torch.randn(16, 2, generator=torch.Generator().manual_seed(1)))
        # The observer hook is registered second, so it sees the written output
        with LastTokenWrite(self.model, 2) as tap, LastTokenWrite(self.model, 2) as observer:
            target, _ = self.capture(tap, self.inputs)
            source, _ = self.capture(tap, self.inputs.flip(0))
            tap.mode = "write"
            tap.source = source
            tap.basis = basis
            observer.mode = "capture"
            with torch.no_grad():
                self.model(input_ids=self.inputs)
        written = observer.captured
        torch.testing.assert_close(written @ basis, source @ basis, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(remove_projection(written, basis), remove_projection(target, basis), atol=1e-5, rtol=1e-5)

    def test_basis_receives_gradient(self) -> None:
        raw = torch.randn(16, 1, requires_grad=True)
        with LastTokenWrite(self.model, 2) as tap:
            source, _ = self.capture(tap, self.inputs.flip(0))
            tap.mode = "write"
            tap.source = source
            tap.basis = torch.linalg.qr(raw).Q
            logits = self.model(input_ids=self.inputs).logits[:, -1]
        logits[:, 0].sum().backward()
        self.assertIsNotNone(raw.grad)
        self.assertGreater(float(raw.grad.abs().sum()), 0.0)


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

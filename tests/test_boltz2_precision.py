"""Precision routing tests with a tiny model: no checkpoints or GPU required."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
from joltz import TrunkState

from mosaic.common import LossTerm
from mosaic.models.boltz2 import Boltz2


class Sampler(eqx.Module):
    precision: str = eqx.field(static=True)

    def sample(self, *, s_trunk, s_inputs, **kwargs):
        assert jax.config.jax_default_matmul_precision == self.precision
        return s_trunk @ s_inputs.T


class TinyBoltz2(eqx.Module):
    structure_module: Sampler

    def embed_inputs(self, features):
        assert jax.config.jax_default_matmul_precision == "float32"
        x = features["res_type"][0, :, 2:4]
        return SimpleNamespace(s_init=x, z_init=x, s_inputs=x,
                               relative_position_encoding=x)

    def trunk_iteration(self, state, embedding, features, *, key, deterministic):
        assert jax.config.jax_default_matmul_precision == "float32"
        return TrunkState(s=embedding.s_inputs, z=embedding.z_init), key

    def diffusion_conditioning(self, s, z, encoding, features):
        assert jax.config.jax_default_matmul_precision == "float32"
        return (s,) * 6

    def distogram_module(self, z):
        assert jax.config.jax_default_matmul_precision == "float32"
        return jnp.zeros((1, 2, 2, 1, 2))

    def confidence_module(self, **kwargs):
        assert jax.config.jax_default_matmul_precision == "float32"
        return SimpleNamespace(plddt=jnp.ones((1, 2)))


class CoordinateLoss(LossTerm):
    def __call__(self, sequence, output, *, key):
        # Also exercise confidence evaluation after the scoped sampler call.
        return output.structure_coordinates.sum() + output.plddt.sum(), {}


def make_model(tf32_sampling):
    raw = TinyBoltz2(Sampler("high" if tf32_sampling else "float32"))
    with patch("mosaic.models.boltz2.lb", return_value=raw):
        # Exercise the constructor default as well as the explicit opt-out.
        model = Boltz2() if tf32_sampling else Boltz2(tf32_sampling=False)
    assert model.model is raw
    return model


class Boltz2PrecisionTest(unittest.TestCase):
    def setUp(self):
        self.sequence = jnp.eye(20, dtype=jnp.float32)[:2]
        res_type = jnp.pad(self.sequence, ((0, 0), (2, 11)))[None]
        self.features = dict(res_type=res_type, msa=res_type[:, None],
                             profile=res_type, atom_pad_mask=jnp.ones((1, 2)))
        self.key = jax.random.key(0)

    def test_prediction_precision_is_local(self):
        for enabled in (True, False):
            with self.subTest(tf32_sampling=enabled):
                model = make_model(enabled)
                with jax.default_matmul_precision("float32"):
                    output = model.model_output(features=self.features, key=self.key)
                    self.assertEqual(output.tf32_sampling, enabled)
                    self.assertEqual(output.structure_coordinates.dtype, jnp.float32)
                    self.assertEqual(output.plddt.shape, (2,))
                    self.assertEqual(jax.config.jax_default_matmul_precision, "float32")

    def test_single_and_multisample_gradients_preserve_precision(self):
        @eqx.filter_jit
        def evaluate(sequence, loss):
            return eqx.filter_value_and_grad(
                lambda s: loss(s, key=self.key)[0]
            )(sequence)

        for enabled in (True, False):
            model = make_model(enabled)
            for builder in (model.build_loss, model.build_multisample_loss):
                with self.subTest(tf32_sampling=enabled, builder=builder.__name__):
                    loss = builder(loss=CoordinateLoss(), features=self.features)
                    self.assertEqual(loss.tf32_sampling, enabled)
                    with jax.default_matmul_precision("float32"):
                        value, gradient = evaluate(self.sequence, loss)
                        self.assertTrue(bool(jnp.isfinite(value)))
                        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))
                        self.assertGreater(float(jnp.linalg.norm(gradient)), 0)
                        self.assertEqual(gradient.dtype, jnp.float32)
                        graph = jax.make_jaxpr(lambda s: loss(s, key=self.key)[0])(
                            self.sequence
                        )
                        dots = [e for e in graph.jaxpr.eqns
                                if e.primitive.name == "dot_general"]
                        self.assertTrue(dots)
                        precision = (jax.lax.Precision.HIGH if enabled
                                     else jax.lax.Precision.HIGHEST)
                        for dot in dots:
                            self.assertEqual(dot.params["precision"], (precision,) * 2)
                        self.assertEqual(jax.config.jax_default_matmul_precision, "float32")


if __name__ == "__main__":
    unittest.main()

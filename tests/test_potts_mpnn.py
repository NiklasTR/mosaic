"""Check recovery semantics and directed-graph conditionals independently."""
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.common import TOKENS
from mosaic.losses.potts_mpnn import (PottsMPNNSequenceRecovery,
    TargetDistribution, recovery_from_probabilities, state_projection)
from mosaic.pottsmpnn import (V6_TOKENS, candidate_energies, gibbs_marginals,
                              incoming_edges)


def test_default_groups_all_states_and_excludes_unknown():
    projection = np.asarray(state_projection(V6_TOKENS))
    assert projection[20].sum() == 0
    assert projection.sum(0)[TOKENS.index('H')] == 4
    assert projection.sum(0)[TOKENS.index('D')] == 4
    assert projection.sum(0)[TOKENS.index('E')] == 4
    assert projection[0, TOKENS.index('A')] == 1
    assert projection.sum() == 29


def test_selected_state_keeps_rejected_mass_and_stops_provider_gradient():
    q = jax.nn.one_hot(jnp.array([TOKENS.index('H'), TOKENS.index('A')]), 20)
    p = jnp.zeros((2, 30)).at[0, 8].set(.6).at[0, 21].set(.2).at[0, 22].set(.2).at[1, 0].set(1.)
    projection = state_projection(V6_TOKENS, {'H': ('HIS-P',)})
    loss, aux = recovery_from_probabilities(q, p, projection, jnp.array([True, False]))
    assert float(loss) == pytest.approx(-.2)
    assert float(aux['accepted_mass']) == pytest.approx(.2)
    dq, dp = jax.grad(lambda a, b: recovery_from_probabilities(a, b, projection,
                      jnp.array([True, False]))[0], argnums=(0, 1))(q, p)
    assert float(dq[0, TOKENS.index('H')]) == pytest.approx(-.2)
    np.testing.assert_array_equal(dq[1], 0)
    np.testing.assert_array_equal(dp, 0)
    positional = jnp.stack((projection, state_projection(V6_TOKENS)))
    assert float(recovery_from_probabilities(q, p, positional)[0]) == pytest.approx(-.6)


@pytest.mark.parametrize('selection', [{'H': 'HIS-P'}, {'H': ('ASP-P',)}, {'H': ()}, {'X': ('UNK',)}])
def test_invalid_selection_rejected(selection):
    with pytest.raises(ValueError):
        state_projection(V6_TOKENS, selection)


def test_conditional_energies_match_exact_hamiltonian_on_asymmetric_graph():
    # Node zero has three incoming edges but K=2. Include nonreciprocal 3->0.
    edges = jnp.array([[0, 1], [1, 0], [2, 0], [3, 0]])
    table = jnp.asarray(np.random.default_rng(3).normal(size=(4, 2, 3, 3)), dtype=jnp.float32)
    states = jnp.array([0, 1, 2, 0])
    def energy(s):
        return table[jnp.arange(4)[:, None], jnp.arange(2)[None], s[:, None], s[edges]].sum()
    incoming = incoming_edges(edges)
    assert int(incoming[2][0].sum()) == 3
    for position in range(4):
        conditional = candidate_energies(table, edges, states, position, incoming)
        exact = jnp.stack([energy(states.at[position].set(token)) for token in range(3)])
        np.testing.assert_allclose(conditional - conditional[0], exact - exact[0], atol=2e-6)


def test_gibbs_single_site_matches_exact_softmax_and_clamps_states_under_jit():
    table = jnp.array([[[[1., 0., 0.], [0., 2., 0.], [0., 0., -1.]]]])
    edges = jnp.array([[0]])
    allowed = jnp.array([[[True, False, True]], [[False, True, False]]])
    p = jax.jit(lambda k: gibbs_marginals(table, edges, allowed, jnp.zeros((1, 3)),
        key=k, temperature=1., burn_in=1, num_sweeps=2))(jax.random.key(4))
    np.testing.assert_allclose(p[0, 0], jax.nn.softmax(jnp.array([-1., -jnp.inf, 1.])), atol=1e-6)
    np.testing.assert_array_equal(p[1, 0], [0., 1., 0.])


class ZeroEncoder(eqx.Module):
    token_names: tuple = eqx.field(static=True, default=V6_TOKENS)
    def encode(self, backbone, residue_idx, chain_id, *, key):
        length = backbone.shape[0]
        return jnp.zeros((length, 1, 30, 30)), jnp.arange(length)[:, None]


def output_for(target):
    full = jax.nn.one_hot(jnp.array([0, TOKENS.index(target)]), 20)
    return SimpleNamespace(full_sequence=full, backbone_coordinates=jnp.zeros((2, 4, 3)),
                           residue_idx=jnp.array([0, 0]), asym_id=jnp.array([0, 1]))


def test_scaffold_bias_target_identity_and_state_validation():
    model = ZeroEncoder()
    q = jax.nn.one_hot(jnp.array([0]), 20)
    loss = PottsMPNNSequenceRecovery(model,
        target=TargetDistribution.from_sequences({'target': 'H'}, [1.]),
        num_chains=1, burn_in=0, num_sweeps=1,
        target_states={'H': ('HIS-P',)}, design_mask=[False], scaffold_aa_idx=[0])
    p, _, _ = loss.chain_probabilities(q, output_for('H'), key=jax.random.key(0))
    np.testing.assert_array_equal(p[0, 0], jax.nn.one_hot(0, 30))
    value, aux = loss(q, output_for('H'), key=jax.random.key(0))
    assert float(value) == 0
    with pytest.raises(ValueError, match='must match'):
        loss(q, output_for('D'), key=jax.random.key(0))
    biased = PottsMPNNSequenceRecovery(model,
        target=TargetDistribution.from_sequences({'target': 'C'}, [1.]),
        num_chains=1, burn_in=0, num_sweeps=1,
        bias=jnp.zeros((1, 20)).at[:, TOKENS.index('C')].set(-1e6))
    p, _, _ = biased.chain_probabilities(q, output_for('C'), key=jax.random.key(0))
    assert float(p[0, 0, 4]) == 0
    np.testing.assert_allclose(p.sum(-1), 1, atol=1e-6)


def test_loss_composes_and_runs_with_jit_gradient():
    term = PottsMPNNSequenceRecovery(ZeroEncoder(),
        target=TargetDistribution.from_sequences({'target': 'H'}, [1.]),
        num_chains=1, burn_in=0, num_sweeps=1)
    q = jnp.ones((1, 20)) / 20
    output = output_for('H')
    combo = 10. * term + .5 * term
    (value, aux), gradient = eqx.filter_jit(eqx.filter_value_and_grad(
        lambda x: combo(x, output, key=jax.random.key(1)), has_aux=True))(q)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    assert len(aux) == 2
    assert float(gradient[0, TOKENS.index('H')]) == pytest.approx(-10.5 * 4 / 29)


def test_whole_sequence_mixture_preserves_correlations_and_weights():
    target = TargetDistribution.from_sequences({'human': 'AGA', 'mouse': 'GAG'}, [.7, .3])
    identities, weights, components = target.sample(jax.random.key(2), num_chains=5)
    expected = np.array([[TOKENS.index(a) for a in s] for s in ('AGA', 'GAG')])
    np.testing.assert_array_equal(identities, expected[np.asarray(components)])
    assert float(weights[components == 0].sum()) == pytest.approx(.7)
    assert float(weights[components == 1].sum()) == pytest.approx(.3)
    independent = TargetDistribution.independent(target.mean_probabilities)
    draws, _, _ = independent.sample(jax.random.key(2), num_chains=32)
    assert any(not any(np.array_equal(row, sequence) for sequence in expected) for row in np.asarray(draws))


def test_soft_canonical_identity_probabilities_are_not_multiplied_by_substate_count():
    p = np.zeros((1, 20), np.float32)
    p[0, TOKENS.index('H')], p[0, TOKENS.index('A')] = .5, .5
    target = TargetDistribution.independent(p)
    identities, weights, _ = target.sample(jax.random.key(9), num_chains=4096)
    assert np.mean(np.asarray(identities) == TOKENS.index('H')) == pytest.approx(.5, abs=.025)
    assert float(weights.sum()) == pytest.approx(1.)


class PairEncoder(eqx.Module):
    token_names: tuple = eqx.field(static=True, default=V6_TOKENS)
    def encode(self, backbone, residue_idx, chain_id, *, key):
        # A/G binder coupled to an A/G target; only G-binder/A-target costs 4.
        table = jnp.zeros((2, 2, 30, 30)).at[0, 1, 7, 0].set(4.)
        return table, jnp.array([[0, 1], [1, 0]])


def test_recovery_is_exact_weighted_probability_mixture_on_tiny_system():
    target = TargetDistribution.from_sequences({'human': 'A', 'mouse': 'G'}, [.5, .5])
    output = output_for('A')
    output.full_sequence = output.full_sequence.at[1:].set(target.mean_probabilities)
    bias = jnp.full((1, 20), -1e6).at[:, TOKENS.index('A')].set(0.).at[:, TOKENS.index('G')].set(0.)
    loss = PottsMPNNSequenceRecovery(PairEncoder(), target=target, bias=bias,
        num_chains=5, burn_in=0, num_sweeps=1)
    q = jax.nn.one_hot(jnp.array([TOKENS.index('A')]), 20)
    value, aux = loss(q, output, key=jax.random.key(5))
    human = 1 / (1 + np.exp(-4.))
    expected = .5 * human + .5 * .5
    assert float(value) == pytest.approx(-expected, abs=1e-6)
    assert float(aux['target_sequence_recovery']['human']) == pytest.approx(human, abs=1e-6)
    assert float(aux['target_sequence_recovery']['mouse']) == pytest.approx(.5, abs=1e-6)
    assert abs(expected - 1 / (1 + np.exp(-2.))) > .1


def test_target_state_filters_preserve_nonhistidine_support():
    from unittest.mock import patch
    target = TargetDistribution.from_sequences({'histidine': 'H', 'alanine': 'A'}, [.5, .5])
    output = output_for('H')
    output.full_sequence = output.full_sequence.at[1:].set(target.mean_probabilities)
    captured = {}
    def sampler(table, edges, allowed, token_bias, **kwargs):
        captured['allowed'] = np.asarray(allowed)
        return allowed / allowed.sum(-1, keepdims=True)
    loss = PottsMPNNSequenceRecovery(ZeroEncoder(), target=target,
        target_states={'H': ('HIS-P',)}, num_chains=2)
    with patch('mosaic.losses.potts_mpnn.gibbs_marginals', side_effect=sampler):
        loss(jnp.ones((1, 20)) / 20, output, key=jax.random.key(1))
    np.testing.assert_array_equal(captured['allowed'][0, 1], np.arange(30) == V6_TOKENS.index('HIS-P'))
    np.testing.assert_array_equal(captured['allowed'][1, 1], np.arange(30) == V6_TOKENS.index('ALA'))


def test_soft_target_loss_is_finite_under_jit_gradient():
    p = np.full((1, 20), 1 / 20, np.float32)
    target = TargetDistribution.independent(p)
    output = output_for('H')
    output.full_sequence = output.full_sequence.at[1:].set(target.mean_probabilities)
    term = PottsMPNNSequenceRecovery(ZeroEncoder(), target=target,
        num_chains=4, burn_in=0, num_sweeps=1)
    (value, aux), gradient = eqx.filter_jit(eqx.filter_value_and_grad(
        lambda q: term(q, output, key=jax.random.key(1)), has_aux=True))(jnp.ones((1, 20)) / 20)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    assert float(gradient[0, TOKENS.index('H')]) == pytest.approx(-4 / 29)


@pytest.mark.parametrize('probabilities,weights,names', [
    (np.ones((1, 20)), [1.], ['target']),
    (np.zeros((1, 1, 20)), [1.], ['target']),
    (np.full((1, 1, 20), np.nan), [1.], ['target']),
    (np.full((1, 1, 20), .05), [.5], ['target']),
    (np.full((2, 1, 20), .05), [.5, .5], ['same', 'same']),
])
def test_invalid_target_distributions_rejected(probabilities, weights, names):
    with pytest.raises(ValueError):
        TargetDistribution(probabilities=probabilities, weights=weights, names=names)


def test_every_target_component_requires_a_chain():
    target = TargetDistribution.from_sequences({'human': 'A', 'mouse': 'G'}, [.5, .5])
    with pytest.raises(ValueError, match='num_chains'):
        PottsMPNNSequenceRecovery(ZeroEncoder(), target=target, num_chains=1)


def test_fractional_target_state_filters_are_rejected():
    target = TargetDistribution.from_sequences({'target': 'H'}, [1.])
    projection = np.asarray(state_projection(V6_TOKENS))[None].copy()
    projection[0, V6_TOKENS.index('HIS-P'), TOKENS.index('H')] = .5
    with pytest.raises(ValueError, match='zero or one'):
        PottsMPNNSequenceRecovery(ZeroEncoder(), target=target, target_states=projection)

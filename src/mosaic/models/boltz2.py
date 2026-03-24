from mosaic.structure_prediction import (
    PolymerType,
    StructurePredictionModel,
    TargetChain,
    StructurePrediction,
)

from mosaic.losses.structure_prediction import (
    BinderTargetIPSAE,
    IPTMLoss,
    TargetBinderIPSAE,
)

from mosaic.losses.boltz2 import (
    load_boltz2 as lb,
    load_features_and_structure_writer,
    set_binder_sequence,
    Boltz2Loss,
    Boltz2Output,
    MultiSampleBoltz2Loss
)

import json
from pathlib import Path
from jaxtyping import Array, Float, PyTree
import equinox as eqx
import jax
import jax.numpy as jnp

import numpy as np
import gemmi

from tempfile import NamedTemporaryFile

def pad_atom_features(features: dict, pad_to: int):

    n_atoms = features["atom_pad_mask"].shape[-1]
    assert pad_to >= n_atoms

    def pad(v):
        pad_width = tuple((0, pad_to-n_atoms) if d == n_atoms else (0,0) for d in v.shape)
        return jnp.pad(v, pad_width)

    return jax.tree.map(pad, features)

def _prefix():
    return """version: 1
sequences:"""


def chain_yaml(chain_name: str, chain: TargetChain) -> str:
    raw = f"""  - {chain.polymer_type.lower()}:
        id: [{chain_name}]
        sequence: {chain.sequence}"""
    if chain.msa_path is not None:
        p = Path(chain.msa_path).expanduser().resolve()
        raw += f"\n        msa: {json.dumps(str(p))}"
    elif not chain.use_msa:
        raw += """
        msa: empty"""
    if chain.cyclic_peptide and chain.polymer_type == PolymerType.PROTEIN:
        raw += """
        cyclic: true"""

    return raw


def target_only_features(
    chains: list[TargetChain],
    *,
    preprocess_out_dir: Path | None = None,
    input_stem: str = "input",
):
    yaml = "\n".join(
        [_prefix()]
        + [
            chain_yaml(chain_id, c)
            for chain_id, c in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
        ]
    )

    tf, template_yaml = build_template_yaml("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
    if tf is not None:
        yaml += template_yaml

    features, writer = load_features_and_structure_writer(
        yaml,
        preprocess_out_dir=preprocess_out_dir,
        input_stem=input_stem,
    )
    if tf is not None: # make sure we actually got a template
        assert np.sum(features["template_mask"]) > 0
    return (features, writer)



def build_template_yaml(chain_names: str, chains: list[TargetChain]):
    # boltz wants perfect .cifs :( 
    templates = {
        chain_id: c.template_chain
        for chain_id, c in zip(chain_names, chains)
        if c.template_chain != None
    }
    if len(templates) > 0:
        st = gemmi.Structure()
        model = gemmi.Model("0")
        entities = []

        for chain_id, chain in templates.items():
            chain.name = chain_id
            ent = gemmi.Entity(chain_id)
            ent.entity_type = gemmi.EntityType.Polymer
            ent.polymer_type = gemmi.PolymerType.PeptideL
            ent.subchains = [chain_id]
            ent.full_sequence = [r.name for r in chain]
            entities.append(ent)
            for r in chain:
                r.subchain = chain_id
            model.add_chain(chain)

        st.add_model(model)
        st.entities = gemmi.EntityList(entities)
        st.assign_subchains()
        st.setup_entities()
        st.ensure_entities()
        st.assign_label_seq_id()

        tf = NamedTemporaryFile(suffix=".cif")

        template_yaml = f"""
        
templates:
  - cif: {tf.name}
    chain_id: [{', '.join(k for k in templates)}]
    template_id: [{', '.join(k for k in templates)}]
"""
        
        st.setup_entities()
        doc = st.make_mmcif_document()
        doc.write_file(tf.name)
        return tf, template_yaml
    else:
        return None, None

def binder_features(
    chains: list[TargetChain],
    *,
    binder_length: int | None = None,
    binder_sequence: str | None = None,
    cyclic_binder: bool = False,
    preprocess_out_dir: Path | None = None,
    input_stem: str = "input",
):
    from mosaic.binder_scaffold import resolve_binder_sequence

    seq, _ = resolve_binder_sequence(binder_length, binder_sequence)
    binder = TargetChain(
        sequence=seq,
        use_msa=False,
        cyclic_peptide=cyclic_binder,
    )
    return target_only_features(
        [binder] + chains,
        preprocess_out_dir=preprocess_out_dir,
        input_stem=input_stem,
    )


class Boltz2(StructurePredictionModel):
    model: eqx.Module

    def __init__(self, cache_path: Path | None = None):
        self.model = lb(cache_path) if cache_path is not None else lb()

    @staticmethod
    def target_only_features(
        chains: list[TargetChain],
        *,
        preprocess_out_dir: Path | None = None,
        input_stem: str = "input",
    ):
        return target_only_features(
            chains,
            preprocess_out_dir=preprocess_out_dir,
            input_stem=input_stem,
        )

    @staticmethod
    def binder_features(
        chains: list[TargetChain],
        *,
        binder_length: int | None = None,
        binder_sequence: str | None = None,
        cyclic_binder: bool = False,
        preprocess_out_dir: Path | None = None,
        input_stem: str = "input",
    ):
        return binder_features(
            chains,
            binder_length=binder_length,
            binder_sequence=binder_sequence,
            cyclic_binder=cyclic_binder,
            preprocess_out_dir=preprocess_out_dir,
            input_stem=input_stem,
        )

    def build_loss(
        self,
        *,
        loss,
        features,
        recycling_steps=1,
        sampling_steps=None,
        binder_design_mask=None,
    ):
        return Boltz2Loss(
            joltz2=self.model,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps if sampling_steps is not None else 25,
            loss=loss,
            deterministic=True,
            binder_design_mask=binder_design_mask,
        )

    def build_multisample_loss(
        self,
        *,
        loss,
        features,
        recycling_steps=1,
        num_samples: int = 4,
        sampling_steps=None,
        reduction=jnp.mean,
        binder_design_mask=None,
    ):
        return MultiSampleBoltz2Loss(
            joltz2=self.model,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps if sampling_steps is not None else 25,
            loss=loss,
            deterministic=True,
            num_samples=num_samples,
            reduction=reduction,
            binder_design_mask=binder_design_mask,
        )


    def model_output(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        sampling_steps=None,
        key,
        binder_design_mask=None,
    ):
        if PSSM is not None:
            features = set_binder_sequence(
                PSSM, features, design_mask=binder_design_mask
            )

        return Boltz2Output(
            joltz2=self.model,
            features=features,
            recycling_steps=recycling_steps,
            num_sampling_steps=sampling_steps if sampling_steps is not None else 25,
            key=key,
            deterministic=True,
        )

    @eqx.filter_jit
    def _coords_and_confidences(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        sampling_steps=None,
        key,
    ):
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            key=key,
        )

        coords = output.structure_coordinates
        pae = output.pae
        plddt = output.plddt
        if PSSM is None:
            PSSM = jnp.zeros((0, 20))
        iptm = -IPTMLoss()(PSSM, output, key=jax.random.key(0))[0]
        _, aux_bt = BinderTargetIPSAE()(PSSM, output, key=jax.random.key(40))
        _, aux_tb = TargetBinderIPSAE()(PSSM, output, key=jax.random.key(41))
        bt_ipsae = aux_bt["bt_ipsae"]
        tb_ipsae = aux_tb["tb_ipsae"]
        return coords, pae, plddt, iptm, bt_ipsae, tb_ipsae

    def predict(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        writer: any,
        recycling_steps=1,
        sampling_steps=None,
        key,
    ):
        coords, pae, plddt, iptm, bt_ipsae, tb_ipsae = self._coords_and_confidences(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            key=key,
        )

        return StructurePrediction(
            st=writer(coords),
            plddt=plddt,
            pae=pae,
            iptm=iptm,
            bt_ipsae=bt_ipsae,
            tb_ipsae=tb_ipsae,
        )

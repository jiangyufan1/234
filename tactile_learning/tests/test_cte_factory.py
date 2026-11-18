import torch

from tactile.config import load_config, resolve_config
from tactile.models.classifier import create_model
from tactile.models.cte_model import CTETactileClassifier
from tactile.models.classifier import TactileClassifier


def test_cte_variant_is_registered():
    config = resolve_config(load_config(variant="cte_hgnn"))

    assert config["model"]["architecture"] == "cte_hgnn"
    assert config["training"]["temporal_shuffle"] is False

    model = create_model(5, config["model"])
    assert isinstance(model, CTETactileClassifier)


def test_original_factory_still_creates_baseline():
    config = load_config(variant="hierarchical_gnn")
    model = create_model(5, config["model"])

    assert isinstance(model, TactileClassifier)


def test_cte_auxiliary_loss_is_finite():
    config = load_config(variant="cte_hgnn")
    config["model"].update({
        "feature_dim": 8,
        "backbone": "tiny",
        "cte_num_atoms": 4,
        "dropout": 0.0,
    })

    model = create_model(5, config["model"])

    B, N, Tr, Tf, H, W = 2, 3, 4, 5, 32, 32

    batch = {
        "rgb": torch.rand(B, N, Tr, 3, H, W),
        "rgb_mask": torch.ones(B, N, Tr, dtype=torch.bool),
        "rgb_times": torch.arange(Tr).float().unsqueeze(0).expand(B, Tr) * .02,
        "flow": torch.rand(B, N, Tf, 2, H, W),
        "flow_mask": torch.ones(B, N, Tf, dtype=torch.bool),
        "flow_dt": torch.full((B, Tf), .02),
        "flow_start": torch.arange(Tf).float().unsqueeze(0).expand(B, Tf) * .02,
        "flow_end": torch.arange(Tf).float().unsqueeze(0).expand(B, Tf) * .02 + .02,
        "node_mask": torch.ones(B, N, dtype=torch.bool),
        "label": torch.tensor([0, 2]),
    }

    logits, aux = model.forward_with_aux(batch)

    ce = torch.nn.functional.cross_entropy(logits, batch["label"])

    aux_loss = sum(
        value for key, value in aux.items()
        if key in (
            "trend_smoothness",
            "event_sparsity",
            "decomposition_reconstruction",
        )
        or (
            key.startswith("flow_")
            and key[5:] in (
                "trend_smoothness",
                "event_sparsity",
                "decomposition_reconstruction",
            )
        )
    )

    total = ce + aux_loss

    assert torch.isfinite(total)
    total.backward()

    assert model.encoder.rgb_encoder.network[0].weight.grad is not None
    assert model.encoder.flow_encoder.network[0].weight.grad is not None


def test_cte_rejects_temporal_shuffle():
    config = load_config(variant="cte_hgnn")
    config["training"]["temporal_shuffle"] = True

    try:
        resolve_config(config)
    except ValueError as exc:
        assert "temporal_shuffle" in str(exc)
    else:
        raise AssertionError("CTE-HGNN must reject temporal shuffling")

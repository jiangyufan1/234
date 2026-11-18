import torch

from tactile.models.cte_model import CTETactileClassifier


def make_batch():
    B, N, Tr, Tf = 2, 3, 4, 5
    H, W = 32, 32

    rgb_times = torch.tensor([
        [0.01, 0.03, 0.05, 0.07],
        [0.02, 0.04, 0.06, 0.08],
    ])

    flow_start = torch.tensor([
        [0.00, 0.02, 0.04, 0.06, 0.08],
        [0.01, 0.03, 0.05, 0.07, 0.09],
    ])

    flow_end = flow_start + 0.02

    return {
        "rgb": torch.rand(B, N, Tr, 3, H, W),
        "rgb_mask": torch.ones(B, N, Tr, dtype=torch.bool),
        "rgb_times": rgb_times,
        "flow": torch.rand(B, N, Tf, 2, H, W),
        "flow_mask": torch.ones(B, N, Tf, dtype=torch.bool),
        "flow_dt": torch.full((B, Tf), 0.02),
        "flow_start": flow_start,
        "flow_end": flow_end,
        "node_mask": torch.ones(B, N, dtype=torch.bool),
        "label": torch.tensor([0, 2]),
    }


def test_raw_input_forward():
    model = CTETactileClassifier(
        num_classes=5,
        config={
            "feature_dim": 8,
            "backbone": "tiny",
            "pretrained": False,
            "dropout": 0.0,
            "cte_num_atoms": 4,
        },
    )

    logits = model(make_batch())

    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()


def test_raw_input_backward_reaches_both_encoders():
    model = CTETactileClassifier(
        num_classes=5,
        config={
            "feature_dim": 8,
            "backbone": "tiny",
            "pretrained": False,
            "dropout": 0.0,
            "cte_num_atoms": 4,
        },
    )

    batch = make_batch()
    logits = model(batch)

    loss = torch.nn.functional.cross_entropy(
        logits, batch["label"]
    )
    loss.backward()

    rgb_grad = sum(
        p.grad.abs().sum()
        for p in model.encoder.rgb_encoder.parameters()
        if p.grad is not None
    )

    flow_grad = sum(
        p.grad.abs().sum()
        for p in model.encoder.flow_encoder.parameters()
        if p.grad is not None
    )

    assert rgb_grad > 0
    assert flow_grad > 0


def test_encoded_sequences_keep_independent_lengths():
    model = CTETactileClassifier(
        num_classes=5,
        config={
            "feature_dim": 8,
            "backbone": "tiny",
            "pretrained": False,
            "cte_num_atoms": 4,
        },
    )

    batch = make_batch()
    rgb, flow = model.encoder(batch)

    assert rgb.shape == (2, 3, 4, 8)
    assert flow.shape == (2, 3, 5, 8)

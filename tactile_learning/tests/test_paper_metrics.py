import torch
from tactile.metrics import classification_metrics
from tactile.evaluation.early_recognition import earliest_confident_time, early_recognition_metrics

def test_classification_paper_metrics():
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"])
    assert m["accuracy"] == 0.75
    assert abs(m["macro_precision"] - (1 + 2/3)/2) < 1e-8
    assert abs(m["macro_recall"] - (0.5 + 1)/2) < 1e-8
    assert abs(m["macro_f1"] - (2/3 + 0.8)/2) < 1e-8
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]

def test_early_recognition_and_ttcr():
    logits_ratio = torch.tensor([
        [[3., 0.], [0., 3.]],
        [[3., 0.], [3., 0.]],
        [[3., 0.], [3., 0.]],
    ])
    logits_time = torch.tensor([
        [[3., 0.], [4., 0.], [5., 0.]],
        [[0., 3.], [4., 0.], [5., 0.]],
    ])
    times = torch.tensor([[.1, .2, .3], [.1, .2, .3]])
    result = early_recognition_metrics(
        logits_ratio, [0, 0], [.25, .5, 1.], ["a", "b"],
        logits_by_time=logits_time, times=times, confidence=.7, patience=2,
    )
    assert len(result["by_observation_ratio"]) == 3
    assert result["auc_or_accuracy"] >= 0
    assert result["ttcr"]["successful_samples"] == 2

def test_ttcr_requires_correct_stable_prediction():
    logits = torch.tensor([[0., 4.], [0., 5.], [5., 0.], [6., 0.]])
    times = torch.tensor([.1, .2, .3, .4])
    assert abs(earliest_confident_time(logits, times, .8, 2, target=0) - .4) < 1e-6

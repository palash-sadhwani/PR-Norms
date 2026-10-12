from app.services.evaluation import evaluate_labels


def test_evaluation_metrics_and_undefined_denominators():
    metrics = evaluate_labels(
        [
            (True, "confirmed"),
            (True, "dismissed"),
            (False, "confirmed"),
            (False, "dismissed"),
        ]
    )
    assert metrics.sample_count == 4
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_negatives == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.false_positive_rate == 0.5

    empty = evaluate_labels([])
    assert empty.precision is None
    assert empty.recall is None
    assert empty.f1 is None
    assert empty.false_positive_rate is None

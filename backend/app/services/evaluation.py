from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationMetrics:
    sample_count: int
    positive_label_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    false_positive_rate: float | None


def evaluate_labels(
    labels: Iterable[tuple[bool, str]],
) -> EvaluationMetrics:
    positive_labels = {"confirmed", "accepted"}
    samples = list(labels)
    tp = sum(
        predicted and decision in positive_labels for predicted, decision in samples
    )
    fp = sum(
        predicted and decision not in positive_labels for predicted, decision in samples
    )
    fn = sum(
        not predicted and decision in positive_labels for predicted, decision in samples
    )
    tn = sum(
        not predicted and decision not in positive_labels
        for predicted, decision in samples
    )
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    false_positive_rate = fp / (fp + tn) if fp + tn else None
    return EvaluationMetrics(
        sample_count=len(samples),
        positive_label_count=sum(
            decision in positive_labels for _, decision in samples
        ),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=false_positive_rate,
    )

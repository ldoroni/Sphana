"""Unit tests for custom loss functions."""

import torch

from sphana_trainer.training.embedding import multiple_negatives_loss
from sphana_trainer.training.gnn import listnet_loss


def test_multiple_negatives_loss_zero_when_identical():
    anchor = torch.eye(2)
    positive = torch.eye(2)
    loss = multiple_negatives_loss(anchor, positive, temperature=0.1)
    assert loss.item() < 1e-3


def test_listnet_loss_prefers_higher_labels():
    scores = torch.tensor([1.0, 0.5])
    labels = torch.tensor([1.0, 0.1])
    higher_loss = listnet_loss(scores, labels, temperature=1.0)
    swapped_loss = listnet_loss(scores.flip(0), labels, temperature=1.0)
    assert higher_loss < swapped_loss


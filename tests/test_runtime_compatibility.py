import pytest

torch = pytest.importorskip("torch")

from vapo.affordance.utils.losses import compute_dice_loss


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable.",
            ),
        ),
    ],
)
@pytest.mark.parametrize("num_classes", [1, 2])
def test_dice_loss_keeps_one_hot_targets_on_input_device(device, num_classes):
    logits = torch.zeros((1, num_classes, 4, 4), device=device)
    labels = torch.zeros((1, 1, 4, 4), dtype=torch.long, device=device)

    loss = compute_dice_loss(labels, logits)

    assert loss.device == logits.device

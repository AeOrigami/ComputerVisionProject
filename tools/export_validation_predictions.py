"""Export existing validation loaders/models without training, splitting, or fusion.

Call export_validation_predictions() from the existing v3-validation kernel.
This module deliberately cannot construct a split or run a notebook from scratch.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import SequentialSampler


def export_validation_predictions(
    *, val_df, adn_model, csn_model, adn_val_loader, csn_val_loader,
    adn_checkpoint, csn_checkpoint, adn_class_to_index, csn_class_to_index,
    adn_indices, csn_indices, label_col, sensitivity_col, training_run_id,
    split_fingerprint, device, use_amp, output_path,
):
    """One inference pass per model, reusing the live ordered validation loaders."""
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Export already exists; refusing to overwrite: {output_path}")
    frame = val_df.copy(deep=True)
    assert len(frame) > 0
    assert {"image_id", "image_path", label_col, sensitivity_col}.issubset(frame.columns)
    assert frame["image_path"].is_unique and frame["image_id"].is_unique
    assert frame[["image_id", "image_path", label_col, sensitivity_col]].notna().all().all()
    for loader in (adn_val_loader, csn_val_loader):
        assert len(loader.dataset) == len(frame)
        assert loader.dataset.frame.equals(frame), "Loader must preserve the existing val_df exactly."
        assert isinstance(loader.sampler, SequentialSampler), "Validation cannot shuffle."
        assert not loader.drop_last
        assert loader.dataset.sample_weights is None
    assert len(adn_val_loader) == len(csn_val_loader)
    assert adn_val_loader.batch_size == csn_val_loader.batch_size
    assert adn_val_loader.dataset.target_col == label_col
    assert csn_val_loader.dataset.target_col == sensitivity_col
    assert adn_val_loader.dataset.class_to_index == adn_class_to_index
    assert csn_val_loader.dataset.class_to_index == csn_class_to_index
    assert adn_val_loader.dataset.frame["image_path"].equals(csn_val_loader.dataset.frame["image_path"])
    assert repr(adn_val_loader.dataset.transform) == repr(csn_val_loader.dataset.transform)

    checkpoints = (Path(adn_checkpoint).resolve(), Path(csn_checkpoint).resolve())
    assert checkpoints[0].parent == checkpoints[1].parent == output_path.parent
    # Check every tensor against the existing best checkpoints without altering a
    # model, checkpoint, training run ID, or validation metric.
    for task, model, path, mapping in (
        ("ADN", adn_model, checkpoints[0], adn_class_to_index),
        ("CSN", csn_model, checkpoints[1], csn_class_to_index),
    ):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["task"] == task
        assert checkpoint["training_run_id"] == training_run_id
        assert checkpoint["split_fingerprint"] == split_fingerprint
        assert checkpoint["selection_metric"] == "validation_macro_f1"
        assert checkpoint["num_classes"] == len(mapping)
        assert checkpoint["class_to_index"] == {str(value): int(index) for value, index in mapping.items()}
        state = model.state_dict()
        assert state.keys() == checkpoint["model_state_dict"].keys()
        for key, value in state.items():
            assert torch.equal(value.detach().cpu(), checkpoint["model_state_dict"][key]), (
                f"{task} model tensor differs from selected checkpoint: {key}"
            )
        del checkpoint, state

    adn_model.eval()
    csn_model.eval()
    assert not adn_model.training and not csn_model.training
    adn_labels, csn_labels, adn_probabilities, csn_probabilities = [], [], [], []
    with torch.inference_mode():
        # Each existing loader is traversed exactly once. No dataset is rebuilt.
        for adn_batch, csn_batch in zip(adn_val_loader, csn_val_loader, strict=True):
            adn_images, adn_targets = adn_batch
            csn_images, csn_targets = csn_batch
            assert torch.equal(adn_images, csn_images), "Ordered validation image batches must match."
            with autocast(enabled=use_amp):
                adn_logits = adn_model(adn_images.to(device, non_blocking=True))
                csn_logits = csn_model(csn_images.to(device, non_blocking=True))
            # Same forward precision as validation; save softmax at float32
            # precision rather than rounding exported probabilities to float16.
            adn_probabilities.append(torch.softmax(adn_logits.float(), dim=1).cpu().numpy())
            csn_probabilities.append(torch.softmax(csn_logits.float(), dim=1).cpu().numpy())
            adn_labels.append(adn_targets.numpy())
            csn_labels.append(csn_targets.numpy())

    adn_probabilities = np.concatenate(adn_probabilities)
    csn_probabilities = np.concatenate(csn_probabilities)
    assert np.array_equal(np.concatenate(adn_labels), frame[label_col].map(adn_class_to_index).to_numpy())
    assert np.array_equal(np.concatenate(csn_labels), frame[sensitivity_col].map(csn_class_to_index).to_numpy())
    assert adn_probabilities.shape == (len(frame), len(adn_class_to_index))
    assert csn_probabilities.shape == (len(frame), len(csn_class_to_index))
    for probabilities in (adn_probabilities, csn_probabilities):
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert np.allclose(probabilities.astype(np.float64).sum(axis=1), 1, rtol=0, atol=1e-6)

    result = frame[["image_id", "image_path"]].copy()
    result["true_real_fake"] = frame[label_col].to_numpy()
    result["true_sensitivity"] = frame[sensitivity_col].to_numpy()
    for name in ("real", "fake"):
        result[f"p_{name}"] = adn_probabilities[:, adn_indices[name]].astype(np.float64)
    for name in ("low", "medium", "high"):
        result[f"p_{name}"] = csn_probabilities[:, csn_indices[name]].astype(np.float64)
    result["split"] = "validation"
    result["training_run_id"] = training_run_id
    result["split_fingerprint"] = split_fingerprint
    result["adn_checkpoint"] = str(checkpoints[0])
    result["csn_checkpoint"] = str(checkpoints[1])
    assert len(result) == len(frame) == len(val_df)
    assert result["image_id"].equals(val_df["image_id"])
    assert result["image_path"].equals(val_df["image_path"])
    assert val_df.equals(frame), "The original validation dataframe must remain unchanged."
    adn_error = float(np.abs(result[["p_real", "p_fake"]].sum(axis=1) - 1).max())
    csn_error = float(np.abs(result[["p_low", "p_medium", "p_high"]].sum(axis=1) - 1).max())
    assert adn_error <= 1e-6 and csn_error <= 1e-6
    # Exclusive creation plus enough digits for an exact float32 round trip.
    with output_path.open("x", encoding="utf-8", newline="") as destination:
        result.to_csv(destination, index=False, float_format="%.17g")

    print(f"Validation sample count: {len(result)}")
    print(f"ADN checkpoint: {checkpoints[0]}")
    print(f"CSN checkpoint: {checkpoints[1]}")
    print(f"Maximum absolute ADN probability-sum error: {adn_error:.10g}")
    print(f"Maximum absolute CSN probability-sum error: {csn_error:.10g}")
    print("First 5 rows:")
    print(result.iloc[:5, :9].to_string(index=False))
    print(f"Saved: {output_path}")
    return result

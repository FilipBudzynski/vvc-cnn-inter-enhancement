import os
import matplotlib.pyplot as plt
from enhancer.config import Config
from enhancer.vtm_dataset import VTMDataset
import numpy as np
import yaml


def test_vtm_dataset():
    # Load the raw YAML directly to avoid the 'str' attribute error
    with open("enhancer/tests/test_config.yaml", "r") as f:
        raw_cfg = yaml.safe_load(f)
    
    # Access the dictionary keys directly
    dataset_cfg = raw_cfg['dataset']

    print("--- Initializing VTMDataset ---")
    dataset = VTMDataset(
        decoded_yuv_filepath=dataset_cfg['decoded_yuv_filepath'],
        original_yuv_filepath=dataset_cfg['original_yuv_file_path'],
        vtm_trace_path=dataset_cfg['vtm_trace_filepath'],
    )

    ds_len = len(dataset)
    print(f"Dataset length: {ds_len} frames")

    if ds_len == 0:
        print(
            "Error: No frames found. Check if CSV/Trace file is empty or formatted correctly."
        )
        return

    x, y, info = dataset[8]

    print(f"Dataset Item POC: {info['poc']} | Crop: {info['top']},{info['left']}")
    print(f"Input Shape: {x.shape}")  # Should be [10, 128, 128]
    print(f"Target Shape: {y.shape}")  # Should be [3, 128, 128]

    # 2. Extract channels for plotting
    # Channels 0,1,2 are Y, U, V
    dec_y = x[0].numpy()
    dec_u = x[1].numpy()
    dec_v = x[2].numpy()

    # Target Y for error calculation
    orig_y = y[0].numpy()

    # Metadata starts from index 3
    qp_map = x[3].numpy()
    pred_mode = x[4].numpy()
    depth_map = x[5].numpy()
    bound_map = x[6].numpy()
    mv_x = x[7].numpy()

    fig, axes = plt.subplots(2, 5, figsize=(22, 10))

    # Row 0: Pixel Data
    axes[0, 0].imshow(dec_y, cmap="gray")
    axes[0, 0].set_title("Decoded Luma (x[0])")

    axes[0, 1].imshow(dec_u, cmap="plasma")
    axes[0, 1].set_title("Upsampled U (x[1])")

    axes[0, 2].imshow(dec_v, cmap="plasma")
    axes[0, 2].set_title("Upsampled V (x[2])")

    diff = np.abs(dec_y - orig_y)
    axes[0, 3].imshow(diff, cmap="hot")
    axes[0, 3].set_title("Luma Error (Target y[0])")

    # Row 1: Metadata (Now part of the same tensor 'x')
    axes[1, 0].imshow(qp_map, cmap="nipy_spectral")
    axes[1, 0].set_title("QP Map (x[3])")

    axes[1, 1].imshow(pred_mode, cmap="tab10")
    axes[1, 1].set_title("PredMode (x[4])")

    axes[1, 2].imshow(depth_map, cmap="magma")
    axes[1, 2].set_title("Block Depth (x[5])")

    axes[1, 3].imshow(bound_map, cmap="gray")
    axes[1, 3].set_title("Tree Map (Boundary)")

    axes[1, 4].imshow(mv_x, cmap="coolwarm")
    axes[1, 4].set_title("Motion Vector X (x[6])")

    plt.tight_layout()
    output_path = "dataset_verification.png"
    plt.savefig(output_path)
    # plt.show()


if __name__ == "__main__":
    test_vtm_dataset()

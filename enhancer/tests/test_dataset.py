import torch

# If the error persists after pip install,
# you can tell Pyright to ignore it for now:
import matplotlib.pyplot as plt  # type: ignore
from enhancer.config import Config
from enhancer.vtm_dataset import VTMDataset

def test_vtm_dataset():
    # 1. Load config
    try:
        # Using the path you provided in previous turns
        cfg = Config.load("enhancer/tests/test_config.yaml") 
        dataset_cfg = cfg.dataset.train
    except Exception as e:
        print(f"Could not load config.yaml: {e}")
        return

    print("--- Initializing VTMDataset ---")
    dataset = VTMDataset(
        decoded_yuv_filepath=dataset_cfg.decoded_yuv_filepath,
        original_yuv_filepath=dataset_cfg.original_yuv_file_path,
        vtm_trace_path=dataset_cfg.vtm_trace_filepath,
        width=dataset_cfg.width,
        height=dataset_cfg.height,
    )

    ds_len = len(dataset)
    print(f"Dataset length: {ds_len} frames")

    if ds_len == 0:
        print("Error: No frames found. Check if CSV/Trace file is empty or formatted correctly.")
        return

    # 2. Get the first item
    dec, orig, meta, info = dataset[0]

    print(f"DEBUG: Y Channel Raw Min: {dec[0].min().item()}")
    print(f"DEBUG: Y Channel Raw Max: {dec[0].max().item()}")
    print(f"DEBUG: Y Channel Raw Mean: {dec[0].mean().item()}")

    # --- CRITICAL DEBUG PRINTS ---
    # If these are all 0.0, the files are not being read correctly
    print(f"\n--- Data Range Debug ---")
    print(f"Decoded Y (Luma) - Min: {dec[0].min():.4f}, Max: {dec[0].max():.4f}")
    print(f"Metadata Map 0  - Min: {meta[0].min():.4f}, Max: {meta[0].max():.4f}")
    print(f"Metadata Map 2 (Depth) - Unique values: {torch.unique(meta[2])}")

    print("\n--- Tensor Shape Verification ---")
    print(f"Decoded (YUV): {dec.shape}")  # Should be [3, 720, 1280]
    print(f"Metadata:     {meta.shape}") # Should be [7, 720, 1280]

    # 4. Visualization
    print("\n--- Plotting Samples ---")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Pixels (Y, U, V channels)
    # vmin=0, vmax=1 is vital if your data is normalized
    axes[0, 0].imshow(dec[0].numpy(), cmap="gray", vmin=dec[0].min(), vmax=dec[0].max())
    axes[0, 0].set_title("Decoded Luma (Y)")

    axes[0, 1].imshow(dec[1].numpy(), cmap="plasma", vmin=0, vmax=1)
    axes[0, 1].set_title("Upsampled U")

    axes[0, 2].imshow(dec[2].numpy(), cmap="plasma", vmin=0, vmax=1)
    axes[0, 2].set_title("Upsampled V")

    # Simple error visualization - normalized difference
    diff = torch.abs(dec[0] - orig[0]).numpy()
    axes[0, 3].imshow(diff, cmap="hot")
    axes[0, 3].set_title("Luma Error (Residuals)")

    # Row 2: Metadata Maps 
    # Metadata maps aren't always 0-1, so we let cmap auto-scale here
    axes[1, 0].imshow(meta[0].numpy(), cmap="viridis")
    axes[1, 0].set_title("QP Map")

    axes[1, 1].imshow(meta[1].numpy(), cmap="tab10")
    axes[1, 1].set_title("PredMode (Inter/Intra)")

    axes[1, 2].imshow(meta[2].numpy(), cmap="magma")
    axes[1, 2].set_title("lock Depth (Quadtree)")

    # Motion Vector X - Using coolwarm because MVs can be negative
    axes[1, 3].imshow(meta[3].numpy(), cmap="coolwarm")
    axes[1, 3].set_title("Motion Vector X")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_vtm_dataset()

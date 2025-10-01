import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path("images") / "output"

def show_array(arr, title):
    plt.figure()
    plt.title(title)
    if arr.ndim == 2:
        plt.imshow(arr, cmap="gray")
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        plt.imshow(arr)
    else:
        plt.imshow(arr[..., 0], cmap="gray")
    plt.axis("off")
    plt.show()  # Blocks until window closed

def main():
    run_dir = input("Enter run directory (e.g. 55067_5_1_5_20): ").strip()
    folder = BASE_DIR / run_dir
    if not folder.is_dir():
        print(f"Not found: {folder}")
        return
    files = sorted(folder.glob("*.npy"))
    if not files:
        print("No .npy files found.")
        return
    for f in files:
        try:
            arr = np.load(f)
        except Exception as e:
            print(f"Failed to load {f.name}: {e}")
            continue
        print(f"Viewing {f.name} shape={arr.shape} dtype={arr.dtype}")
        show_array(arr, f.name)
    print("Done.")

if __name__ == "__main__":
    main()

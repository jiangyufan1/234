"""Check dependencies and perform a small CUDA operation without opening a viewer."""
import importlib
import sys


def main():
    print("Python:", sys.executable, sys.version.split()[0])
    for name in ("numpy", "torch", "torchvision", "cv2", "mujoco"):
        try:
            module = importlib.import_module(name)
            print(f"{name}: {getattr(module, '__version__', 'available')}")
        except Exception as exc:
            print(f"{name}: UNAVAILABLE ({exc})")
    try:
        import torch
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            value = torch.ones(16, 16, device="cuda") @ torch.ones(16, 16, device="cuda")
            torch.cuda.synchronize()
            print("GPU:", torch.cuda.get_device_name(0), "operation:", value[0, 0].item())
    except Exception as exc:
        print("CUDA check failed:", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

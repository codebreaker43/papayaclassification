"""Train the TFLite MobileNetV2 model on papaya_curated_dataset (Multi-Angle & Multi-Background)."""
import os
import runpy

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    runpy.run_path(os.path.join(base_dir, "prepare_dataset_and_train.py"), run_name="__main__")

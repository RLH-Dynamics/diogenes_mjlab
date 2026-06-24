"""Development and inspection tools for the Diogenes MuJoCo model.

This package provides command-line utilities:
  - check_masses: Inspect compiled body masses and inertia tensors
  - inspect_joints: Interactive MuJoCo viewer for joint posing
  - export_onnx: Export trained actor checkpoint to ONNX format

Each module can be run as:
  python -m diogenes_mjlab.tools.check_masses
  python -m diogenes_mjlab.tools.inspect_joints
  python -m diogenes_mjlab.tools.export_onnx <checkpoint>
"""

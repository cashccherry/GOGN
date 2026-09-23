"""Deterministic inputs for small regression tests."""
import sys
from pathlib import Path
import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(42)
    np.random.seed(42)

"""Historical geometry generator; not needed to run experiments.

Uses the original seed and draw order. Existing directories deliberately cause
an error to avoid overwriting authoritative acquisition arrays.
"""
import os
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np

from gogn.objectives.FWI import *
from gogn.optimization.utils import line_search_interpolate, line_search_backtrack
from gogn.optimization.nonlinear import *

np.random.seed(1260)

src      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/good/srcs.npy")))
rec      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/good/recs.npy")))
invPars  = InversionPars(sigmak = 1e2)

bounds = [480e3/5, 4 * 480e3/5]

nrec = 300
nsrcs = [1,5,10,25,50]

rec = np.random.uniform(*bounds, (nrec,2,1))

for nsrc in nsrcs:
  src = np.random.uniform(*bounds, (nsrc,2,1))
#  print(src.shape)
#  print(rec.shape)
  os.mkdir(os.path.join(CWD_file, f"FWI/good{nsrc}"))
  np.save(os.path.join(CWD_file, f"FWI/good{nsrc}/srcs.npy"), src)
  np.save(os.path.join(CWD_file, f"FWI/good{nsrc}/recs.npy"), rec)

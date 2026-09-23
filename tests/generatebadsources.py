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

np.random.seed(67)

src0      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/bad/srcs.npy")))
rec0      = torch.tensor(np.load(os.path.join(CWD_file, f"FWI/bad/recs.npy")))
invPars  = InversionPars(sigmak = 1e2)

pbounds = [-50e3, 50e3]

nrec = 300
nsrcs = [1,5,10,25,50]

nbadsrc = src0.shape[0]

print(src0.shape)
print(rec0.shape)

rec = rec0 #np.random.uniform(*bounds, (nrec,2,1))

for nsrc in nsrcs:
  srclist = []
  for i in range(nsrc//nbadsrc):
    srclist.append(src0 + np.random.uniform(*pbounds,(nbadsrc,2,1)))
  srclist.append(src0[:nsrc%nbadsrc])
  src = torch.cat(srclist)
  print(src.shape)
  print(rec.shape)
  os.mkdir(os.path.join(CWD_file, f"FWI/bad{nsrc}"))
  np.save(os.path.join(CWD_file, f"FWI/bad{nsrc}/srcs.npy"), src)
  np.save(os.path.join(CWD_file, f"FWI/bad{nsrc}/recs.npy"), rec)

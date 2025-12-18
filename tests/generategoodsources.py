import os
import sys

import torch
import numpy as np

from sfgn.objectives.FWI import *
from sfgn.optimization.utils import line_search_interpolate, line_search_backtrack
from sfgn.optimization.nonlinear import *

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

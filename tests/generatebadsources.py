import os
import sys

import torch
import numpy as np

from sfgn.objectives.FWI import *
from sfgn.optimization.utils import line_search_interpolate, line_search_backtrack
from sfgn.optimization.nonlinear import *

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

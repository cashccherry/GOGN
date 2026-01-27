import os
import sys

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def percent_formatter(x, pos):
    return f"{x*100:+.0f}%"

from sfgn.objectives.FWI import *
from sfgn.optimization.utils import line_search_interpolate_grad, line_search_interpolate, line_search_backtrack
from sfgn.optimization.nonlinear import *


coverages    = ['good', 'bad', 'good25', 'bad25']
covs   = ['bad','good']
cov25s = ['bad25','good25']
sigma_noises = [1e-2,1e-1,1.]
algs         = ["CG", "LBFGS", "SFGN", "GNCG"]

final_obj = {}
final_err = {}
best_err  = {}

for coverage in coverages:
  for sigma_noise in sigma_noises:
    results = {}
    for alg in algs:
      # Initialize empty lists
      iterations = []
      times = []
      forwards = []
      backwards = []
      sims = []
      objectives = []
      gradnorms = []
      model_errors = []
      steplengths = []
      
      with open(os.path.join(os.getcwd(),'results', 'acoustic', f'{coverage}-{sigma_noise}-{alg}.csv'), 'r') as csvfile:
          reader = csv.DictReader(csvfile)
          
          for row_num, row in enumerate(reader, start=2):  # start=2 because header is row 1
              try:
                  iterations.append(int(row['Iteration']))
                  times.append(float(row['Time']))
                  forwards.append(int(row['Forwards']))
                  backwards.append(int(row['Backwards']))
                  sims.append(int(row['Forwards']) + int(row['Backwards']))
                  objectives.append(float(row['Objective']))
                  gradnorms.append(float(row['Gradnorm']))
                  model_errors.append(float(row['Model Error']))
                  steplengths.append(float(row['Steplength']))
              except (ValueError, KeyError) as e:
                  print(f"Warning: Issue with row {row_num}: {e}")
                  print(f"Row data: {row}")
      
      print(f"\nSuccessfully read {len(iterations)} rows")
    
      results[f"iterations-{alg}"] = iterations
      results[f"times-{alg}"] = times
      results[f"forwards-{alg}"] = forwards
      results[f"backwards-{alg}"] = backwards
      results[f"sims-{alg}"] = sims 
      results[f"objectives-{alg}"] = [o/objectives[0] for o in objectives]
      results[f"gradnorms-{alg}"] = [g/gradnorms[0] for g in gradnorms]
      results[f"model_errors-{alg}"] = [m/model_errors[0] for m in model_errors]
      results[f"steplengths-{alg}"] = steplengths
  
      final_obj[f"{coverage}-{sigma_noise}-{alg}"] = results[f"objectives-{alg}"][-1]
      final_err[f"{coverage}-{sigma_noise}-{alg}"]  = results[f"model_errors-{alg}"][-1]
      best_err[f"{coverage}-{sigma_noise}-{alg}"]  = min(results[f"model_errors-{alg}"])

#print(final_obj)
#print(final_err)
#print(best_err)



final_objective = [
[""] + sum([[f"Final Objective {alg}"] for alg in algs], [])
]
best_error = [
[""] + sum([[f"Best Model Error {alg}"] for alg in algs], [])
]
final_objective25 = [
[""] + sum([[f"Final Objective {alg}"] for alg in algs], [])
]
best_error25 = [
[""] + sum([[f"Best Model Error {alg}"] for alg in algs], [])
]


for coverage in covs:
  for sigma_noise in sigma_noises:
    row_f0 = [f"{coverage} - σ = {sigma_noise}"]
    row_m0 = [f"{coverage} - σ = {sigma_noise}"]

    for alg in algs:
      row_f0.append(f"{final_obj[f'{coverage}-{sigma_noise}-{alg}']:.4f}")
      row_m0.append(f"{best_err[f'{coverage}-{sigma_noise}-{alg}']:.4f}")

    final_objective.append(row_f0)
    best_error.append(row_f0)

for coverage in cov25s:
  for sigma_noise in sigma_noises:
    row_f0 = [f"{coverage} - σ = {sigma_noise}"]
    row_m0 = [f"{coverage} - σ = {sigma_noise}"]

    for alg in algs:
      row_f0.append(f"{final_obj[f'{coverage}-{sigma_noise}-{alg}']:.4f}")
      row_m0.append(f"{best_err[f'{coverage}-{sigma_noise}-{alg}']:.4f}")

    final_objective25.append(row_f0)
    best_error25.append(row_f0)




import csv

with open(os.path.join(os.getcwd(),'results', 'acoustic', f'final_objective.csv'), 'w') as csvfile:
  writer = csv.writer(csvfile)
  for row in final_objective:
    writer.writerow(row)

with open(os.path.join(os.getcwd(),'results', 'acoustic', f'best_error.csv'), 'w') as csvfile:
  writer = csv.writer(csvfile)
  for row in best_error:
    writer.writerow(row)


with open(os.path.join(os.getcwd(),'results', 'acoustic', f'final_objective25.csv'), 'w') as csvfile:
  writer = csv.writer(csvfile)
  for row in final_objective25:
    writer.writerow(row)

with open(os.path.join(os.getcwd(),'results', 'acoustic', f'best_error25.csv'), 'w') as csvfile:
  writer = csv.writer(csvfile)
  for row in best_error25:
    writer.writerow(row)




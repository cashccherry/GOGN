for coverage in bad good; do
  for sigma in 1e-2 1e-1 1; do
    for norm_p in 0.5 1 1.5 2 2.5 3; do
      mkdir figures/acousticTT/$coverage
      mkdir figures/acousticTT/$coverage/$sigma
      mkdir figures/acousticTT/$coverage/$sigma/$norm_p
      python tests/SFGNTT-test.py $coverage $sigma $norm_p
      python tests/plot-SFGNTT-test.py $coverage $sigma $norm_p
    done 
  done
done 

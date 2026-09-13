for coverage in bad good bad25 good25; do
  for sigma in 1e-2 1e-1 1; do
    mkdir figures/acousticTV/$coverage
    mkdir figures/acousticTV/$coverage/$sigma
    python tests/SFGN-TV-test.py $coverage $sigma
    python tests/plot-SFGN-TV-test.py $coverage $sigma
  done
done 

# Limitations and deviations

- Necessary upper bounds can establish impossibility but cannot prove that the
  reported optimizer actually attained a feasible value below those bounds.
- The reported numbers lie comfortably inside the feasible region, so this
  route does not resolve missing checkpoint or full-training evidence.
- Failed CPU optimization routes are deliberately excluded from falsification:
  they do not contradict the existence of the paper's reported runs.

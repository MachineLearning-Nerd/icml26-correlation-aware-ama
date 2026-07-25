# Limitations and deviations

- This is a valid full-scale CA-AMA, but it is separable across items and uses
  an analytically derived payment rather than the paper's learned 2048-menu
  joint AMenuNet and three-layer pCor network.
- The route directly tests whether the reported revenue regime is attainable
  under the exact distribution; it does not reproduce the authors' missing
  3x10 checkpoints or their optimizer trajectory.
- The paper/release transpose discrepancy prevents exact recovery of the 3x10
  menu and gamma configuration from public artifacts.

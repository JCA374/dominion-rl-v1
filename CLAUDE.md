# CLAUDE.md

`JCA374/dominion-rl-v1` (folder `trainers/rl-v1`): one network (PyTorch MLP, run in C) answers every Dominion
decision; behaviour cloning of the GA genome, then PPO against a league. No search at play time. README.md has the
commands; the live site (docs/, ml-project-site skill) shows the training.

- Rules only from `engine/` (submodule). `az-v1/` is a submodule for az-v1's agent (`c/az.c`, included by `c/rl.c`) and,
  through it, dominion-v1's GA agent, genomes and bots. All engine pins must agree (`tests/test_core.py`).
- `c/rl.c` is the hot path: features (298), forward pass, sampling, recording, seats (GA / AZ / NET), clip trace.
  `rl/net.py` must match its layout (`test_forward_matches_c`).
- Training: `train.py` (CFG at the top, `--set k=v`, `--continue`, `--init_from`). Runs in `runs/<run>/` (gitignored
  except site.json `push_globs`). `opponents/az_r2_357.npy` = az-v1's best net (the agent to beat).
- Measured results in `results/` (goal_* include az-v1's goal games; `--vs_az` adds 1,000 games vs az-v1's best).
- Podcast: `tools/podcast_publish.sh NN` (uses az-v1's `.voice-venv`, Python 3.12 + Kokoro).

# dominion-rl-v1: one network plays all of Dominion, learned by reinforcement learning

**🌐 Live site: https://jca374.github.io/dominion-rl-v1/**

The third method of the Dominion project (base set, 2nd edition, Steam rules). After the evolved priority lists of
[dominion-v1](https://github.com/JCA374/dominion-v1) (genetic algorithm) and the AlphaZero-style buy search of
[dominion-az-v1](https://github.com/JCA374/dominion-az-v1) (70% against the GA genome), this one has **no search and no
hand-written rule**: a single network answers every decision (buys, actions, trashing, gains, discards). It starts by
imitating the GA genome (behaviour cloning) and improves by **PPO** against a league (the GA genome, the bots, its past
snapshots and itself).

Goal: better than az-v1's best agent: above 70% on az-v1's 1,000 goal games against the GA base genome, and above 50%
head to head against az-v1's best agent on 1,000 fresh games.

- Rules: [dominion-engine](https://github.com/JCA374/dominion-engine), submodule `engine/` (never copied here).
- az-v1, submodule `az-v1/`: its agent (the one to beat; weights in `opponents/az_r2_357.npy`), and through it
  dominion-v1's GA genomes and bots. All three engine pins must agree (a test checks it).
- Everything inside a game is C (`c/rl.c` + the engine + az-v1's `c/az.c` + dominion-v1's `c/ga_agent.c`, built into
  `rl.so`); training is PyTorch on the CPU.

```bash
git submodule update --init --recursive
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy pytest pillow matplotlib
.venv/bin/python -m pytest -q tests                      # builds rl.so, parity tests
.venv/bin/python -u train.py --run r1                    # imitation, then PPO -> runs/r1/
.venv/bin/python -u train.py --run r1 --continue         # resume
.venv/bin/python experiments/goal_check.py runs/r1/net_25.npy --name r1_iter25 --vs_az
.venv/bin/python publish.py [--push]                     # rebuild the site (docs/)
```

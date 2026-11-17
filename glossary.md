# Glossary

One term per line, shown on the Study tab:

**Dominion** — a deck-building card game: each turn you play action cards from your hand, then buy one card for your deck; most victory points when the game ends wins
**decision** — one question the engine asks a player: which action to play, what to buy, what to trash, discard or gain; about 22 kinds
**kingdom** — the 10 action cards (out of the base set's 26) dealt for one game
**GA genome** — dominion-v1's evolved priority lists: a fixed rule for every kind of decision, found by a genetic algorithm
**GA base** — the best portfolio-v2 genome whose required cards are in the kingdom; the reference opponent of every measure
**az-v1 / AZ best** — the AlphaZero-light project: the GA genome's action play with buys chosen by IS-MCTS (400 simulations) guided by a network; 70% vs the GA base
**reinforcement learning (RL)** — learning to act from the results of one's own actions (here: won or lost games), not from correct answers
**behaviour cloning (BC)** — supervised learning of an expert's choices; here the GA genome's, as the starting point
**policy** — the network's probabilities over the legal answers to a decision
**value / win estimate** — the network's guess of the final result from here, in [−1, +1]
**PPO** — proximal policy optimisation: a policy-gradient method that limits how far one update can change the policy (the clip)
**policy gradient** — make choices from games that went better than expected more likely, and the others less likely
**advantage** — how much better a decision turned out than the network expected; positive → reinforce, negative → discourage
**GAE** — generalised advantage estimation: advantages built from the chain of win estimates along a game, with a mixing factor λ
**entropy** — how spread out the policy's probabilities are; high = undecided / exploring, low = certain
**KL divergence** — how far one update moved the policy; PPO stops an update early when it gets too large
**league** — the mix of opponents in training: the GA base, the bots, past snapshots and the current network itself
**snapshot** — a saved copy of the network at some iteration, used later as an opponent
**argmax play** — taking the highest-probability answer; used in every evaluation (training games sample instead)
**goal check** — az-v1's 1,000 goal games against the GA base plus 1,000 head-to-head games against az-v1's best agent
**Wilson interval** — the 95% confidence interval of a win rate; about ±3 points at 1,000 games
**winner's curse** — the best of many noisy evaluations is flattered by luck

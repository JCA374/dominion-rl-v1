"""Project hooks for publish.py (ml-project-site): the Play tab's game clips, pictures of the net for
The model tab, and the results file for the custom pages.

Clips use az-v1's drawing (one frame per buy of either player: both decks, the score, what was
bought, and for the network the probabilities it gave the top buys). Drawing lives here only; no
policy imports this file.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os

import numpy as np

from rl import core

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = "r1"
EVERY = 25                     # the Play tab's newest net moves every EVERY iterations (clips are committed)
N = core.NUM_CARDS
NAME = core.NAME

_spec = importlib.util.spec_from_file_location("az_site_hooks", os.path.join(ROOT, "az-v1", "site_hooks.py"))
AZH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AZH)          # az-v1's frame(), encode_mp4(), GAMES and the model-tab figure helpers


def _nets(run=RUN):
    nets = glob.glob(os.path.join(ROOT, "runs", run, "net_*.npy"))
    return sorted((os.path.relpath(p, ROOT) for p in nets), key=lambda p: int(p.split("_")[-1][:-4]))


def models():
    out = [{"id": "ga-base", "label": "GA base genome", "path": "ga",
            "note": "dominion-v1's portfolio-v2 specialist for the kingdom: an evolved priority list that makes every "
                    "decision. The network started by imitating it."},
           {"id": "az-best", "label": "AZ-light best (search)", "path": "az",
            "note": "az-v1's best agent: the GA genome's action play, buys by IS-MCTS with 400 simulations guided by "
                    "r2's net 357. 70% against the GA base. The one to beat."}]
    nets = _nets()
    if nets:
        out.append({"id": "rl-bc", "label": "RL net, imitation only", "path": nets[0],
                    "note": "Iteration 0: the network after imitating the GA genome's decisions, before any RL."})
    newest = [p for p in nets if int(p.split("_")[-1][:-4]) % EVERY == 0 and not p.endswith("net_0.npy")]
    if newest:
        it = int(newest[-1].split("_")[-1][:-4])
        out.append({"id": f"rl-{it}", "label": f"RL net (iter {it})", "path": newest[-1],
                    "note": "The network after PPO: it makes every decision itself, with no search."})
    return out


def games():
    return AZH.games()


def _seat(path, kingdom):
    g = core.base_genome(kingdom)["ints"]
    if path == "ga":
        return core.Seat(core.GA, g), "GA base"
    if path == "az":
        return core.Seat(core.AZ_SEAT, g, np.load(os.path.join(ROOT, "opponents", "az_r2_357.npy")), sims=400), "AZ search"
    return core.Seat(core.NET, g, np.load(os.path.join(ROOT, path))), "RL net"


def render_play(policy_path, game, out_mp4):
    k = game["kingdom"]
    seat, label = _seat(policy_path, k)
    og = core.base_genome(k)["ints"] if game["opponent"] == "GA base" else core.bot(game["opponent"], k)
    seats = [seat, core.Seat(core.GA, og)]
    labels = [label, game["opponent"]]
    r = core.play(seats, k, game["seed"], rng_seed=game["seed"] + 1, trace=True)
    tr = r["trace"]
    frames, last = [], [None, None]
    for t in tr:
        me, coins, choice = int(t[0]), int(t[2]), int(t[3])
        what = "nothing" if choice == core.PASS else NAME[choice]
        line2 = ""
        pi = t[7 + 2 * N:]
        if me == 0 and label != "GA base" and pi.max() < 0.999:
            top = np.argsort(-pi)[:3]
            who = "search" if label == "AZ search" else "net"
            line2 = f"{who}: " + " · ".join(f"{'nothing' if a == core.PASS else NAME[a]} {pi[a] * 100:.0f}%"
                                            for a in top if pi[a] > 0.005)
            if label == "RL net":
                line2 += f"   win estimate {(float(t[6]) + 1) / 2 * 100:.0f}%"
        last[me] = (f"${coins}: buys {what}", line2)
        frames.append((AZH.frame(t, labels, k, last), 11))
    res = core.outcome(r, 0)
    word = "wins" if res > 0 else "loses" if res < 0 else "shares the win"
    sc = r["scores"]
    fin = f"{label} {word} {sc[0]}–{sc[1]} in {max(r['turns'])} turns"
    frames.append((AZH.frame(tr[-1], labels, k, last, final=fin), 75))
    AZH.encode_mp4(frames, out_mp4)
    return {"ok": res > 0, "result": "win" if res > 0 else "loss" if res < 0 else "shared", "detail": fin + "."}


# ── The model tab ───────────────────────────────────────────────────────────────────────────

KIND_SHOW = ["buy", "play action", "Militia discard", "Workshop gain", "Chapel trash", "Remodel gain"]


def model_viz(out_dir):
    """Architecture, weights, what it chooses per decision kind (vs the GA genome) and input
    sensitivity, for the newest net of RUN. Redrawn only when that net changes."""
    import torch
    from rl.net import load
    nets = _nets()
    if not nets:
        return []
    src = os.path.join(ROOT, nets[-1])
    pt = src.replace(".npy", ".pt")
    it = nets[-1].split("_")[-1][:-4]
    figs = [("arch", "Architecture", f"{RUN} net {it}: layer sizes read from the saved weights. One trunk, two heads: "
                                    "34 logits (a card, or pass) for whatever decision is asked, and a win estimate."),
            ("weights", "Weights per layer", "A wide, centred histogram means the layer is used; a spike at 0 means "
                                             "dead units."),
            ("agreement", "How often it does what the GA genome would do", "On positions from games against the GA "
             "base: the share of decisions, per kind, where the network's top choice equals the genome's. Imitation "
             "starts near 100% on easy kinds; RL moves away where it found something better (or worse)."),
            ("sensitivity", "What it looks at", "How much its choices move (total variation, averaged over all recorded "
                                                "decisions) when one input is nudged by one standard deviation. The 25 "
                                                "most influential of the 298 inputs.")]
    paths = {k: os.path.join(out_dir, f"{k}.png") for k, _, _ in figs}
    entries = [{"src": f"model/{k}.png", "title": t, "caption": c} for k, t, c in figs]
    if all(AZH.cached(paths[k], src) for k in paths):
        return entries
    net = load(pt)
    sd = net.state_dict()
    AZH.fig_arch([("input: what it knows at a decision", core.NF), ("dense ReLU", net.nh), ("dense ReLU", net.nh),
                  ("34 logits  |  value", core.NA)], paths["arch"], title=f"Net {it}")
    AZH.fig_weight_hist([("W1", sd["l1.weight"].numpy()), ("W2", sd["l2.weight"].numpy()),
                         ("policy", sd["pi.weight"].numpy()), ("value", sd["v.weight"].numpy())], paths["weights"])
    # positions: the GA base genome's decisions (recorded) in the Play games and more
    import random
    rng = random.Random("rl-v1 model viz")
    rows = []
    for i in range(60):
        k = core.random_kingdom(rng)
        g = core.base_genome(k)["ints"]
        r = core.play([core.Seat(core.GA, g, record=True), core.Seat(core.GA, g, record=True)], k, rng.getrandbits(32))
        rows.append(r["rec"])
    X = np.concatenate([r["X"] for r in rows])
    L = np.concatenate([r["LEGAL"] for r in rows])
    A = np.concatenate([r["A"] for r in rows])
    K = np.concatenate([r["KIND"] for r in rows])
    with torch.no_grad():
        lg, _ = net(torch.from_numpy(X), torch.from_numpy(L).bool())
        prob = torch.softmax(lg, -1).numpy()
    pick = prob.argmax(1)
    names, vals = [], []
    for kname in core.KINDS:
        m = K == core.KINDS.index(kname)
        if m.sum() >= 20:
            names.append(f"{kname} ({m.sum()})")
            vals.append(float((pick[m] == A[m]).mean()))
    AZH.fig_bars(names, vals, paths["agreement"], title="Top choice = the GA genome's choice", ylabel="share")
    std = X.std(0)
    eff = np.zeros(core.NF)
    with torch.no_grad():
        for j in range(core.NF):
            if std[j] == 0:
                continue
            X2 = X.copy()
            X2[:, j] += std[j]
            lg2, _ = net(torch.from_numpy(X2), torch.from_numpy(L).bool())
            eff[j] = 0.5 * np.abs(torch.softmax(lg2, -1).numpy() - prob).sum(1).mean()
    fn = feature_names()
    top = np.argsort(-eff)[:25]
    AZH.fig_bars([fn[j] for j in top], [eff[j] for j in top], paths["sensitivity"],
                 title="Change in its choices per +1 std of one input", ylabel="total variation")
    return entries


def feature_names():
    names = AZH.feature_names()
    names += [f"hand: {NAME[c]}" for c in range(N)] + [f"in play: {NAME[c]}" for c in range(N)]
    names += ["actions left", "my turn"] + [f"kind: {k}" for k in core.KINDS] + [f"about: {NAME[c]}" for c in range(N)]
    assert len(names) == core.NF
    return names


def extra(data_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        out[os.path.basename(p)[:-5]] = json.load(open(p))
    json.dump(out, open(os.path.join(data_dir, "results.json"), "w"), separators=(",", ":"))

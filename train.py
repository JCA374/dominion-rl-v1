"""rl-v1 training: behaviour cloning of the GA genome on every decision, then PPO against a league.

    python train.py --run r1                 # new run in runs/r1/
    python train.py --run r1 --continue      # resume from runs/r1/ckpt.pt
    python train.py --run r2 --init_from runs/r1/net_40.pt   # PPO from a given net

Per iteration: `games` games with the learner sampling from its policy against the league (the GA
base genome, the three bots, itself and its past snapshots), a PPO update on those decisions, then
the fixed evaluation (argmax play): 400 games against the GA base genome, 300 against the bots.
Writes runs/<run>/stats.csv, events.jsonl, net_<iter>.pt/.npy, ckpt.pt, config.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np

CFG = {
    "nh": 256,
    "bc_games": 20000,
    "bc_epochs": 6,
    "games": 3000,           # PPO games per iteration
    "vs_ga": 0.4,            # league shares: the GA base genome
    "vs_bots": 0.15,         #   the three bots
    "vs_past": 0.15,         #   past snapshots (the rest: the current net, both seats learning)
    "snapshot_every": 10,
    "snapshots": 8,
    "temp": 1.0,
    "lam": 0.95,             # GAE
    "clip": 0.2,
    "epochs": 3,
    "batch": 4096,
    "lr": 3e-4,
    "vf": 0.5,               # value loss weight
    "ent": 0.01,             # entropy bonus
    "max_kl": 0.03,          # stop the epoch loop early above this approximate KL
    "eval_kingdoms": 100,    # vs GA base: x 2 seeds x 2 seats = 400 games; bots: 25 x 2 x 2 x 3 = 300
    "workers": max(1, os.cpu_count() - 1),
    "iterations": 100000,
    "goal_every": 25,
    "vs_az_every": 100,      # goal checks with the 1,000 head-to-head games against az-v1's best (~3 min)
    "chunk": 50,
}
BOTS = ["Big Money", "Smithy-BM", "Witch-BM"]


def chunks(games, n):
    return [games[i:i + n] for i in range(0, len(games), n)]


def _ppo(job):
    from rl.jobs import ppo_chunk
    return ppo_chunk(job)


def _bc(job):
    from rl.jobs import bc_chunk
    return bc_chunk(job)


def _eval(job):
    from rl.jobs import eval_chunk
    return eval_chunk(job)


def eval_games(cfg):
    from rl.core import kingdom_for_bot, random_kingdom
    rng = random.Random("rl-v1 eval")
    games = []
    for _ in range(cfg["eval_kingdoms"]):
        k = random_kingdom(rng)
        for _ in range(2):
            s = rng.getrandbits(32)
            games += [{"kingdom": k, "seed": s, "seat": seat, "opp": {"kind": "ga", "tag": "GA base"}} for seat in (0, 1)]
    for b in BOTS:
        for _ in range(cfg["eval_kingdoms"] // 4):
            k = kingdom_for_bot(rng, b)
            for _ in range(2):
                s = rng.getrandbits(32)
                games += [{"kingdom": k, "seed": s, "seat": seat, "opp": {"kind": "bot", "name": b, "tag": b}}
                          for seat in (0, 1)]
    return games


def run_eval(pool, cfg, agent):
    jobs = [{"games": c, "agent": agent, "nh": cfg["nh"]} for c in chunks(eval_games(cfg), cfg["chunk"])]
    res = [r for part in pool.imap_unordered(_eval, jobs) for r in part]
    out = {}
    for tag in ["GA base"] + BOTS:
        pts = [r["points"] for r in res if r["opp"] == tag]
        out[tag] = float(np.mean(pts))
    out["bots"] = float(np.mean([out[b] for b in BOTS]))
    out["turns"] = float(np.mean([r["turns"] for r in res]))
    out["capped"] = float(np.mean([r["capped"] for r in res]))
    return out


def fmt(e):
    return f"GA {e['GA base']:.3f} BM {e['Big Money']:.3f} Smithy {e['Smithy-BM']:.3f} Witch {e['Witch-BM']:.3f} bots {e['bots']:.3f}"


class Log:
    def __init__(self, d):
        self.d = d

    def event(self, kind, **kw):
        with open(os.path.join(self.d, "events.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **kw}) + "\n")
        print(f"[{kind}]", json.dumps(kw)[:300], flush=True)

    def row(self, row):
        path = os.path.join(self.d, "stats.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


def ppo_update(net, opt, data, cfg, torch):
    """Clipped PPO on one iteration's decisions. Returns mean metrics."""
    n = len(data["A"])
    X = torch.from_numpy(data["X"])
    L = torch.from_numpy(data["LEGAL"]).bool()
    A = torch.from_numpy(data["A"]).long()
    OLD = torch.from_numpy(data["LOGP"])
    adv = data["ADV"]
    ADV = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))
    RET = torch.from_numpy(data["RET"])
    stats = []
    stop = False
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n)
        for i in range(0, n, cfg["batch"]):
            j = perm[i:i + cfg["batch"]]
            logits, v = net(X[j], L[j])
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(1, A[j][:, None]).squeeze(1)
            ratio = torch.exp(logp - OLD[j])
            s1 = ratio * ADV[j]
            s2 = torch.clamp(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * ADV[j]
            pg = -torch.min(s1, s2).mean()
            p = logp_all.exp()
            ent = -(p * torch.where(L[j], logp_all, torch.zeros_like(logp_all))).sum(-1).mean()
            vl = ((v - RET[j]) ** 2).mean()
            loss = pg + cfg["vf"] * vl - cfg["ent"] * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                kl = (OLD[j] - logp).mean().item()
                clipfrac = ((ratio - 1).abs() > cfg["clip"]).float().mean().item()
            stats.append({"pg_loss": pg.item(), "value_loss": vl.item(), "entropy": ent.item(), "kl": kl,
                          "clipfrac": clipfrac})
            if kl > cfg["max_kl"]:
                stop = True
                break
        if stop:
            break
    out = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    out["epochs_done"] = ep + 1
    return out


def bc_train(net, data, cfg, torch, log):
    """Imitation of the GA genome: cross-entropy on its choice + 0.5 x value error on the result.
    Batches index the numpy arrays (no copies: the data is ~1 GB)."""
    n = len(data["A"])
    idx = np.arange(n)
    hold = idx[idx % 10 == 0][:200000]
    train_idx = idx[idx % 10 != 0]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def batch(j):
        return (torch.from_numpy(data["X"][j]), torch.from_numpy(data["LEGAL"][j]).bool(),
                torch.from_numpy(data["A"][j]).long(), torch.from_numpy(data["Z"][j]))

    for ep in range(cfg["bc_epochs"]):
        perm = np.random.default_rng(ep).permutation(train_idx)
        for i in range(0, len(perm), cfg["batch"]):
            x, lg, a, z = batch(np.sort(perm[i:i + cfg["batch"]]))
            logits, v = net(x, lg)
            loss = torch.nn.functional.cross_entropy(logits, a) + 0.5 * ((v - z) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            x, lg, a, z = batch(hold)
            logits, v = net(x, lg)
            pick = logits.argmax(1)
            buy = torch.from_numpy(data["KIND"][hold] == 1)
            res = {"acc": (pick == a).float().mean().item(), "acc_buys": (pick == a)[buy].float().mean().item(),
                   "acc_other": (pick == a)[~buy].float().mean().item(),
                   "ce": torch.nn.functional.cross_entropy(logits, a).item(), "value_mse": ((v - z) ** 2).mean().item()}
        log.event("bc_epoch", epoch=ep, samples=n, holdout=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r1")
    ap.add_argument("--continue", dest="cont", action="store_true")
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from rl.core import kingdom_for_bot, random_kingdom
    from rl.net import Net, load, save

    d = os.path.join("runs", args.run)
    os.makedirs(d, exist_ok=True)
    log = Log(d)
    cfg = dict(CFG)
    if args.cont and os.path.exists(os.path.join(d, "config.json")):
        cfg.update(json.load(open(os.path.join(d, "config.json"))))
    for kv in args.set:
        k, v = kv.split("=")
        cfg[k] = type(CFG[k])(v)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    pool = mp.Pool(cfg["workers"])
    t0 = time.time()

    def snap(it):
        return os.path.join(d, f"net_{it}.npy")

    if args.cont:
        ck = torch.load(os.path.join(d, "ckpt.pt"), weights_only=False)
        net = Net(cfg["nh"])
        net.load_state_dict(ck["net"])
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        opt.load_state_dict(ck["opt"])
        it, snaps = ck["iter"] + 1, ck["snaps"]
        log.event("run_resume", iter=it)
    else:
        log.event("run_start", config=cfg, init_from=args.init_from)
        if args.init_from:
            net = load(args.init_from)
        else:
            net = Net(cfg["nh"])
            rng = random.Random("rl-v1 bc")
            games = []
            for i in range(cfg["bc_games"]):
                if i % 2 == 0:
                    games.append({"kingdom": random_kingdom(rng), "seed": rng.getrandbits(32)})
                else:
                    b = BOTS[i // 2 % 3]
                    games.append({"kingdom": kingdom_for_bot(rng, b), "seed": rng.getrandbits(32), "bot": b,
                                  "seat": i // 6 % 2})
            tb = time.time()
            parts = list(pool.imap_unordered(_bc, [{"games": c} for c in chunks(games, cfg["chunk"])]))
            data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            log.event("bc_data", games=len(games), decisions=int(len(data["A"])), secs=round(time.time() - tb))
            bc_train(net, data, cfg, torch, log)
            del data
        save(net, os.path.join(d, "net_0.pt"))
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        e = run_eval(pool, cfg, {"kind": "net", "path": snap(0)})
        log.event("init_eval", eval=fmt(e))
        log.row(make_row(0, t0, {}, {}, e))
        it, snaps = 1, [snap(0)]
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "iter": 0, "snaps": snaps},
                   os.path.join(d, "ckpt.pt"))

    best_eval = [-1.0]
    if os.path.exists(os.path.join(d, "stats.csv")):
        with open(os.path.join(d, "stats.csv")) as f:
            best_eval[0] = max(float(r["eval_GA_base"]) for r in csv.DictReader(f))
    while it <= cfg["iterations"]:
        ti = time.time()
        learner = snap(it - 1)
        rng = random.Random(f"rl-v1 ppo {args.run} {it}")
        games = []
        for i in range(cfg["games"]):
            u = rng.random()
            seat = i % 2
            if u < cfg["vs_ga"]:
                opp, k = {"kind": "ga", "tag": "GA base"}, random_kingdom(rng)
            elif u < cfg["vs_ga"] + cfg["vs_bots"]:
                b = BOTS[i % 3]
                opp, k = {"kind": "bot", "name": b, "tag": b}, kingdom_for_bot(rng, b)
            elif u < cfg["vs_ga"] + cfg["vs_bots"] + cfg["vs_past"] and len(snaps) > 1:
                opp, k = {"kind": "net", "path": rng.choice(snaps[:-1]), "tag": "past"}, random_kingdom(rng)
            else:
                opp, k = {"kind": "net", "path": learner, "tag": "self"}, random_kingdom(rng)
            games.append({"kingdom": k, "seed": rng.getrandbits(32), "seat": seat, "opp": opp})
        jobs = [{"games": c, "learner": learner, "nh": cfg["nh"], "temp": cfg["temp"], "lam": cfg["lam"]}
                for c in chunks(games, cfg["chunk"])]
        parts = list(pool.imap_unordered(_ppo, jobs))
        res = [r for p in parts for r in p["results"]]
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "results"}
        t_play = time.time() - ti
        sp = {"selfplay_vs_ga": float(np.mean([r["points"] for r in res if r["opp"] == "GA base"])),
              "selfplay_vs_bots": float(np.mean([r["points"] for r in res if r["opp"] in BOTS])),
              "selfplay_vs_past": float(np.mean([r["points"] for r in res if r["opp"] == "past"] or [float("nan")])),
              "decisions": int(len(data["A"])),
              "value_mse_fresh": float(((data["V"] - data["Z"]) ** 2).mean()),
              "t_play": t_play}
        tt = time.time()
        net.train()
        m = ppo_update(net, opt, data, cfg, torch)
        sp["t_train"] = time.time() - tt
        save(net, os.path.join(d, f"net_{it}.pt"))
        e = run_eval(pool, cfg, {"kind": "net", "path": snap(it)})
        sp["t_iter"] = time.time() - ti
        log.row(make_row(it, t0, m, sp, e))
        print(f"iter {it}: {fmt(e)} | train vs GA {sp['selfplay_vs_ga']:.3f} kl {m['kl']:.4f} ent {m['entropy']:.3f} "
              f"| {sp['t_iter']:.0f}s", flush=True)
        if e["GA base"] > best_eval[0]:
            best_eval[0] = e["GA base"]
            log.event("goal_best", iter=it, check={"score": round(e["GA base"], 4), "n": 4 * cfg["eval_kingdoms"]},
                      champion=snap(it), eval=fmt(e))
        if it % cfg["snapshot_every"] == 0:
            snaps = (snaps + [snap(it)])[-cfg["snapshots"]:]
        for old in range(max(0, it - 30), it - 2):          # keep snapshots, goal nets and the last two
            p = os.path.join(d, f"net_{old}.npy")
            if old % cfg["snapshot_every"] and old % cfg["goal_every"] and os.path.exists(p):
                os.remove(p)
                os.remove(p.replace(".npy", ".pt"))
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "iter": it, "snaps": snaps},
                   os.path.join(d, "ckpt.pt"))
        if it % cfg["goal_every"] == 0:
            import subprocess
            import sys
            subprocess.run([sys.executable, "experiments/goal_check.py", snap(it), "--name", f"{args.run}_iter{it}",
                            "--label", f"{args.run} iteration {it}", "--nh", str(cfg["nh"])]
                           + (["--vs_az"] if it % cfg["vs_az_every"] == 0 else []), check=False)
        it += 1


def make_row(it, t0, m, sp, e):
    row = {"iter": it, "hours": round((time.time() - t0) / 3600, 4)}
    for k in ("pg_loss", "value_loss", "entropy", "kl", "clipfrac", "epochs_done"):
        row[k] = round(m[k], 5) if k in m else ""
    for tag in ["GA base"] + BOTS + ["bots"]:
        row[f"eval_{tag.replace(' ', '_').replace('-', '_')}"] = round(e[tag], 4)
    row["eval_turns"] = round(e["turns"], 2)
    row["eval_capped"] = round(e["capped"], 4)
    for k in ("selfplay_vs_ga", "selfplay_vs_bots", "selfplay_vs_past", "decisions", "value_mse_fresh",
              "t_play", "t_train", "t_iter"):
        row[k] = (round(sp[k], 4) if isinstance(sp[k], float) and not math.isnan(sp[k]) else sp[k]) if k in sp else ""
    return row


if __name__ == "__main__":
    main()

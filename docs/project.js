// Project formatters: how this project's run events read in the log.
window.eventText = function (e) {
  const p = (x) => (x == null ? '–' : (100 * x).toFixed(1) + '%');
  switch (e.kind) {
    case 'run_start': return [`Run ${e.run} started`, e.init_from ? `PPO from ${e.init_from}.` :
      `Imitation of the GA genome (${e.config.bc_games.toLocaleString()} games, ${e.config.bc_epochs} epochs), then PPO: ${e.config.games.toLocaleString()} games per iteration, league GA ${p(e.config.vs_ga)}, bots ${p(e.config.vs_bots)}, past selves ${p(e.config.vs_past)}, itself the rest.`];
    case 'bc_data': return ['Imitation data collected', `${e.decisions.toLocaleString()} decisions of the GA genome from ${e.games.toLocaleString()} games (${e.secs} s).`];
    case 'bc_epoch': return [`Imitation, epoch ${e.epoch + 1}`, `Held-out decisions: makes the genome's choice ${p(e.holdout.acc)} of the time (buys ${p(e.holdout.acc_buys)}, other decisions ${p(e.holdout.acc_other)}); win-estimate error ${e.holdout.value_mse.toFixed(3)} (1.0 = no idea).`];
    case 'init_eval': return ['Iteration 0 evaluated (the imitating network, top choices)', e.eval];
    case 'goal_best': return [`New best against the GA base: iteration ${e.iter}`, `${p(e.check.score)} on ${e.check.n} evaluation games. ${e.eval}\nBest of many evaluations: flattered by luck (winner's curse). The goal checks are the real measure.`];
    case 'run_resume': return [`Run ${e.run} resumed at iteration ${e.iter}`, ''];
  }
  return null;
};

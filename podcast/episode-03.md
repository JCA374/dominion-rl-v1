# Episode 03: Confirmed, the third method is better

This is the episode I hoped to make. The new player, one network with no search, is better than the searching player. Here is how I checked, because a claim like that is easy to fool yourself about.

Where we left off, the new player was winning fifty five percent head to head against the searcher. But it still won only sixty eight percent against the evolved list, while the searcher won seventy. So it was better in a direct match and not yet better in general.

I let it keep practising, with the searcher still in its practice games. Over the next hundred rounds, the score against the evolved list crept up. Sixty eight, seventy, seventy one, seventy and a half. Against the three simple bots it rose to eighty percent, where the searcher wins about seventy six. And head to head it stayed at about fifty five.

Now the careful part. I had been looking at those checks every twenty five rounds. If you look at many noisy numbers and pick the best one, you are picking luck as well as skill. So I picked a version in advance: the one from round two hundred, a regular checkpoint, not the best looking one. Then I played it once on four thousand games that no part of the project had ever seen.

Two thousand games head to head against the searcher. The new player won fifty five percent. The range of likely values runs from about fifty three to fifty seven. So the lead is real.

And two thousand games against the evolved list, on new card sets. The new player won seventy point three percent. Then I gave the searcher exactly the same two thousand games. It won sixty nine. On identical games, the network without search is a little ahead, and in the direct match it is clearly ahead.

So what made the difference? I think it is three things the earlier methods could not do. It plays its own action cards, discards and trashes, where both older players still used the evolved list. It learns from its own results instead of from an imagined judge. And it is fast. A game takes three thousandths of a second, against half a second for the searcher. That let it practise on about one point two million games in three hours, against a set of opponents that included the searcher itself.

There are limits to say out loud. It was trained partly against the searcher. Some of its lead in the direct match may come from knowing that one opponent's habits. Its lead against the evolved list is small, about one point. And all of this is against computer players. Whether it would beat a strong human is a different question.

But the question I set out with has an answer. A completely different kind of learning, trial and error without any search, beat both earlier methods. The network is saved and the numbers are on the site. Training is still running in the background, so there may be a follow-up.

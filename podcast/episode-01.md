# Episode 01: A third way to learn Dominion

This is the first episode about a new project, the third attempt at teaching a computer to play the card game Dominion. I want to explain why it exists, how it works, and what happened in its first hour.

The first attempt was evolution. A genetic algorithm bred priority lists, simple rules like buy a Province if you can, else a Gold. After millions of games those lists became strong. I call that player the evolved list.

The second attempt borrowed the idea behind AlphaZero. Before every buy, it imagined the rest of the game four hundred times and picked the card that worked out best. It beat the evolved list seven games in ten. Then it got stuck there, whatever I changed. Its judgement of each imagined position leaned on the evolved list playing the game out. And it never touched how its cards were played. Every action, every discard, every card it trashed, was still decided by the evolved list.

So the third attempt throws out both the rules and the search. One neural network answers every question the game asks. What to buy. Which action to play first. What to throw away when the opponent plays a Militia. What to trash with a Chapel. There are about twenty kinds of question. For all of them, the network sees the same picture. What it owns, and what is in its hand. What is left on the table, and what the opponent has. And which question is being asked.

It learns in two steps. First it imitates. I recorded twenty thousand games of the evolved list, about a million decisions, and the network learned to predict each one. After six passes it made the same choice as the list ninety eight percent of the time. So it starts as a copy of the evolved list, not as a beginner.

The second step is reinforcement learning, with a method called PPO. Now the network plays for itself, and it does not always take its favourite answer. It rolls the dice in proportion to how much it likes each answer. Sometimes that means trying something new. When a game is won, every decision in it becomes a little more likely. When a game is lost, a little less likely. The network also guesses, at every moment, how likely it is to win. That guess lets it tell which decisions mattered: a decision after which its chances rose gets more credit than one after which they fell. And PPO keeps every step small, so one unlucky batch of games cannot wreck a good player.

It also needs good opponents. If it only played itself, it could learn tricks that only beat itself. If it only played the evolved list, it could learn tricks that only beat that list. So four games in ten are against the evolved list, a few against three simple bots, a few against its own older versions, and the rest against itself.

The surprise was speed. The network runs inside the game engine, and a whole game takes three thousandths of a second. The searching player from the second attempt needs half a second per game. So this one learns from three thousand games every ten seconds.

And it works. The copy, before any reinforcement learning, won about fifty two percent of its test games against the evolved list. After about ten minutes of practice, the fresh check said sixty four percent. After fifteen minutes, sixty five. The searching player's best was seventy, so the gap is closing fast.

There is one warning sign already. Against the three simple bots, it is getting slightly worse, from seventy five percent to seventy. It may be specialising in beating the evolved list, which it meets most often. The real test comes soon: a thousand games head to head against the searching player itself. I'll tell you how that goes.

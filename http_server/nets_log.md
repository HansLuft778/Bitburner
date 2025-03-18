mcts zero works:

no double pass
territory_bonus
no move count bonus
0.5 percent of board empty for pass penalty
0.3 alpha/epsilon in dirichlet noise
no temperature

checkpoint 252: 
~350 self play games played
2.0 ucb_c for mcts
0.3 dirichlet noise (both alpha and epsilon)
0.5 percent of board empty for pass penalty
no move count bonus
full territory bonus (stones + territory bonus)
net size:
    6 res blocks
    1 conv layer for policy head

no scheduler for learning rate
winrate against netburners: ~0.5
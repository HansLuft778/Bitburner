# Net config 1:

self.initialConvBlock = nn.Sequential(
nn.Conv2d(
in_channels=1 # disabled nodes + 1 # side (1...black, 0...white) + 2 # current black/white + num_past_steps \* 2, # past moves
out_channels=num_hiden,
kernel_size=3,
padding=1,
),
nn.BatchNorm2d(num_hiden),
nn.ReLU(),
)

self.feature*extractor = nn.ModuleList([ResBlock(num_hiden) for * in range(num_res_blocks)])

self.policyHead = nn.Sequential(
nn.Conv2d(num*hiden, 16, kernel_size=3, padding=1),
nn.BatchNorm2d(16),
nn.ReLU(),
nn.Conv2d(16, 8, kernel_size=3, padding=1),
nn.BatchNorm2d(8),
nn.ReLU(),
nn.Flatten(),
nn.Linear(8 * board*width * board_height, board_width \* board_height + 1),
)

self.valueHead = nn.Sequential(
nn.Conv2d(num*hiden, 3, kernel_size=3, padding=1),
nn.BatchNorm2d(3),
nn.ReLU(),
nn.Flatten(),
nn.Linear(3 * board*width * board_height, 1),
nn.Tanh(),
)

class ResBlock(nn.Module):
def **init**(self, num_hidden: int):
super().**init**() # pyright: ignore
self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
self.bn1 = nn.BatchNorm2d(num_hidden)
self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
self.bn2 = nn.BatchNorm2d(num_hidden)

## Training runs

~3 Sec for 1000 iterations
3 past steps
2 ResBlocks
32 channels

lr: 8e-5
batch size: 128
wheight decay: 2e-4
T_max: 2000
eta_min: 5e-6

collect 2 games min

~50k params

=> poor performance 0/17 for black

## Training runs 2

2 past steps
4 ResBlocks
32 channels

lr: 3e-4
batch size: 256
wheight decay: 2e-4
T_max: 2000
eta_min: 5e-6

88903 params

## Training runs 3

2 past steps
4 ResBlocks
64 channels

lr: 3e-4
batch size: 256
wheight decay: 2e-4
T_max: 2000
eta_min: 5e-6

collect 8 games min

318727 params

checkpoint_29_GOOD.pth
checkpoint_35_GOOD.pth
checkpoint_49_GOOD.pth <- the one before plateau
checkpoint_74_GOOD.pth

# 1814407 params

2 past steps
6 ResBlocks
128 channels

lr: 3e-4
batch size: 128

similar performance to 3 ResBlocks, 32 channels
result is ~60% winrate for black, but might be marginally better (overall loss is the same)

checkpoint_1_8M.pth

# katago model

66% netburner
50% slum snakes

=======================================================================================================================================

value head converges to 0.6 problem:

-   gradients not exploding
-   disabled aux heads -> no improvement
-   SGD instead of Adam -> no improvement
-   checked data augmeentation for correctness
-   value target smoothing (changing hard binary targets to softer 0.9/0.1 targets) -> no improvement
-   value head seperation (gets own 1x1 conv -> BN-> ReLu -> own global mean/max poling -> linear -> relu -> linear) -> no improvement (or maybe a slight improvement)
-   more inputs in game data vector (num own stones, num opp stones, num empty stones) -> no improvement, but i kept it in
-   lower learning rate from 3e-4 down to 1e-6 -> no improvement
-   white can also play first move -> no improvement (now it predicts first mover wins), kept it in
-   oversample second wins from buffer
    -   factor 2.0 -> no improvement (even tho the model predictions seemed a little better, loss was still 0.6)
    -   factor 3.0 -> no improvement
-   Training strategies:
    -   2 steps every 4 games -> no improvement (barely any learning in the first place)
    -   5 steps every 2 games -> no improvement
    -   15 steps every 4 games -> no improvement
    -   30 steps every 2 games -> no improvement
-   playout cap randomization -> no improvement, but i kept it in
-   disable score value target in MCTS -> no improvement
-   change value head to tanh with MSE loss (from cross entropy) -> converges to 0.5 (instead of 0.6) 

TODO:

-   larger batch size (256/512)
-   set buffer size to 3k

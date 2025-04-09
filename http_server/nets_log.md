# Net config 1:

self.initialConvBlock = nn.Sequential(
    nn.Conv2d(
        in_channels=1  # disabled nodes
        + 1  # side (1...black, 0...white)
        + 2  # current black/white
        + num_past_steps * 2,  # past moves
        out_channels=num_hiden,
        kernel_size=3,
        padding=1,
    ),
    nn.BatchNorm2d(num_hiden),
    nn.ReLU(),
)

self.feature_extractor = nn.ModuleList([ResBlock(num_hiden) for _ in range(num_res_blocks)])

self.policyHead = nn.Sequential(
    nn.Conv2d(num_hiden, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 8, kernel_size=3, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(8 * board_width * board_height, board_width * board_height + 1),
)

self.valueHead = nn.Sequential(
    nn.Conv2d(num_hiden, 3, kernel_size=3, padding=1),
    nn.BatchNorm2d(3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(3 * board_width * board_height, 1),
    nn.Tanh(),
)


class ResBlock(nn.Module):
    def __init__(self, num_hidden: int):
        super().__init__()  # pyright: ignore
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
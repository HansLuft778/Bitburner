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



largest model so far, not really good:
class ResNet(nn.Module):
    def __init__(
        self,
        board_width: int,
        board_height: int,
        num_res_blocks: int = 4,
        num_hiden: int = 64,
        num_past_steps: int = 2,
    ) -> None:
        super().__init__()  # pyright: ignore
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
            nn.Conv2d(num_hiden, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_width * board_height, board_width * board_height + 1),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hiden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * board_width * board_height, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.initialConvBlock(x)
        for block in self.feature_extractor:
            x = block(x)
        return (self.policyHead(x), self.valueHead(x))


class ResBlock(nn.Module):
    def __init__(self, num_hidden: int):
        super().__init__()  # pyright: ignore
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x: torch.Tensor):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
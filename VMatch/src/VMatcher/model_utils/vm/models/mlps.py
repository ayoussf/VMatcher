import torch.nn as nn

class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=nn.SiLU(),
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, u_0, u_1, **kwargs):
        feat_0 = self.fc1(u_0)
        feat_0, gate_0 = feat_0.chunk(2, dim=-1)
        feat_0 = feat_0 * self.activation(gate_0)
        feat_0 = self.fc2(feat_0)

        feat_1 = self.fc1(u_1)
        feat_1, gate_1 = feat_1.chunk(2, dim=-1)
        feat_1 = feat_1 * self.activation(gate_1)
        feat_1 = self.fc2(feat_1)

        return feat_0, feat_1

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=False, device="cuda", dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, **factory_kwargs)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, u_0, u_1, **kwargs):
        feat_0 = self.fc1(u_0)
        feat_0 = self.act(feat_0)
        feat_0 = self.fc2(feat_0)

        feat_1 = self.fc1(u_1)
        feat_1 = self.act(feat_1)
        feat_1 = self.fc2(feat_1)
        
        return feat_0, feat_1
import torch
loss = torch.nn.MultiLabelSoftMarginLoss()
loss_1 = torch.nn.BCEWithLogitsLoss()
output = torch.tensor([[0.3, 0.05, 0.8], [0.4, 0.25, 0.1]])  # Model predictions
output_2 = torch.tensor([[0.9, 0.05, 0.8], [0.01, 0.8, 0.05]])  # Model predictions
output_3 = torch.tensor([[1.0, 0.00, 1.0], [0.0, 1.0, 0.0]])  # Model predictions
target = torch.tensor([[1, 0, 1], [0, 1, 0]])  # Ground truth
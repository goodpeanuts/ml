import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from logic_regression.model import ImdbDataset, LogicRgeressionModel, asset_dir
import torch.nn as nn

    
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"working on {device}")

ds = ImdbDataset(256, True)

dl = DataLoader(dataset=ds, batch_size=128, shuffle=True, drop_last=True)

model = LogicRgeressionModel(32768, 8, 256).to(device)
model_save_path = os.path.join(asset_dir, "model.pth")

loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

start_epoch = 0
if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint["epoch"]
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("Starting training from scratch")

for epoch in range(start_epoch, num_epochs):
    for inputs, labels in dl:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_func(outputs.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(
        {"epoch": epoch + 1, "model_state_dict": model.state_dict()},
        model_save_path,
    )
    print(f"Model saved to {model_save_path}")

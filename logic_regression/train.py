import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from logic_regression.model import ImdbDataset, LogicRgeressionModel, asset_dir, model_save_path
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
device = torch.device("mps" if torch.mps.is_available() else "cpu")
logging.info(f"working on {device}")

model = LogicRgeressionModel(32768, 8, 256).to(device)

data_loader = DataLoader(dataset=ImdbDataset(256, True), batch_size=128, shuffle=True, drop_last=True)

loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train():
    num_epochs = 10
    start_epoch = 0 

    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        logging.info(f"Resuming training from epoch {start_epoch}")
    else:
        logging.info("Starting training from scratch")

    for epoch in range(start_epoch, num_epochs):
        for inputs, labels in data_loader:
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
        logging.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train()

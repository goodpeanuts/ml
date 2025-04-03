import logging
import torch
from torch.utils.data import DataLoader
from logic_regression.model import ImdbDataset, LogicRgeressionModel, model_save_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = torch.device("mps" if torch.mps.is_available() else "mps")
logging.info(f"working on {device}")

model = LogicRgeressionModel(32768, 8, 256).to(device)
checkpoint = torch.load(model_save_path)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

data_loader = DataLoader(dataset=ImdbDataset(256, False), batch_size=128)

correct = 0
total = 0

logging.info("Testing the model...")
with torch.no_grad():
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total

logging.info(f"Test Accuracy: {accuracy:.4f}")

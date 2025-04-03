import os
import csv
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from logic_regression.model import ImdbDataset, LogicRgeressionModel, asset_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


embedding_dim = 8

device = torch.device("mps" if torch.mps.is_available() else "cpu")
logging.info(f"working on {device}")

res = []
res.append(["epoch\\w"] + [epoch for epoch in range(4, 65, 4)])

for max_length in [16, 32, 64, 128, 256]:
    logging.info(f"loading train dataset, max_length={max_length}")
    train_ds = ImdbDataset(max_length, True)
    train_dl = DataLoader(
        dataset=train_ds, batch_size=128, shuffle=True, drop_last=True
    )

    logging.info(f"loading test dataset, max_length={max_length}")
    test_ds = ImdbDataset(
        max_length,
        False,
    )
    test_dl = DataLoader(dataset=test_ds, batch_size=128, shuffle=True)

    model = LogicRgeressionModel(32768, embedding_dim, max_length).to(device)

    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_res = [max_length]

    for epoch in range(1, 65):
        for train_inputs, train_labels in train_dl:
            train_inputs, train_labels = train_inputs.to(device), train_labels.to(
                device
            )
            outputs = model(train_inputs)
            loss = loss_func(outputs.squeeze(), train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 4 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for test_inputs, test_labels in test_dl:
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(
                        device
                    )
                    outputs = model(test_inputs)
                    predicted = (outputs.squeeze() > 0.5).float()
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
                accuracy = f"{(correct / total * 100):.2f}"
                logging.info(f"max_length={max_length}, epoch={epoch}, accuracy={accuracy}")
                epoch_res.append(accuracy)
            model.train()
    res.append(epoch_res)


csv_file = os.path.join(
    asset_dir,
    f"evaluation_embedding_dim={embedding_dim}.csv",
)

with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows([[row[i] for row in res] for i in range(len(res[0]))])

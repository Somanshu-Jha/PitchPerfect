import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.ml_models.dl_scoring_model import ScoringFFNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 [RLHF Training Daemon] Starting Incremental Retrain on {device}...")
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "scoring_dataset.csv")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "data", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    active_path = os.path.join(model_dir, "scoring_ffnn.pth")
    backup_path = os.path.join(model_dir, f"backup_scoring_ffnn_{datetime.now().strftime('%Y%m%d%H%M')}.pth")
    
    # 1. Back up Model for Safety
    if os.path.exists(active_path):
        print(f"🔒 [RLHF] Checkpoint detected. Backing up active weights -> {backup_path}")
        shutil.copy2(active_path, backup_path)
    
    # 2. Load Extended Dataset
    X_data = []
    Y_data = []
    
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if len(row) < 8:
                continue
            try:
                # Anti-poison checks: Must be 0-1 for F[0]-F[6]
                features = [max(0.0, min(1.0, float(x))) for x in row[:7]]
                target = max(1.0, min(10.0, float(row[7])))  # Anti-poison checks 1-10
                
                X_data.append(features)
                Y_data.append([target])
            except ValueError:
                # Corrupt row data silently skipped during DB ingest errors
                continue
            
    if len(X_data) < 10:
        print("❌ Dataset corrupted. Rollback enforced.")
        return
        
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y_data, dtype=torch.float32).to(device)
    
    # 3. Incremental Weight Injection
    model = ScoringFFNN().to(device)
    if os.path.exists(active_path):
        print("🔄 [RLHF] Resuming checkpoint. Performing incremental training...")
        model.load_state_dict(torch.load(active_path, map_location=device))
        
    criterion = nn.MSELoss()
    # Learning rate deliberately lowered to .001 to prevent catastrophic forgetting
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    # 4. Train
    print("⏳ [RLHF] Optimizing PyTorch Graph...")
    epochs = 300
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, Y_tensor)
        loss.backward()
        optimizer.step()
        
    print(f"📉 [RLHF] Final PyTorch MSE Loss: {loss.item():.4f}")
        
    # 5. Overwrite Active Weights Safely
    if loss.item() < 3.0: # Arbitrary explosion bound
        torch.save(model.state_dict(), active_path)
        print("✅ [RLHF] Continuous model iteration successfully written to disk!")
    else:
        print("⚠️ [RLHF] Network explosion detected! Weights invalidated, backup restored.")
        shutil.copy2(backup_path, active_path)

if __name__ == "__main__":
    train()

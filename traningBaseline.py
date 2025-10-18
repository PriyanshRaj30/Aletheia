import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import load_npz
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load prepared data
print("Loading data...")
combined = pd.read_csv('cleaned_books_data.csv')
matrix = load_npz('user_item_matrix_sparse.npz')
with open('user_mapping.pkl', 'rb') as f:
    user_mapping = pickle.load(f)
with open('book_mapping.pkl', 'rb') as f:
    book_mapping = pickle.load(f)

n_users, n_books = matrix.shape
print(f"Loaded: {n_users} users, {n_books} books")

# print(combined.index)
# Train-val split (time-agnostic; use ratings timestamp if added later)
train_idx, val_idx = train_test_split(combined.index, test_size=0.2, random_state=42)
train_ratings = combined.loc[train_idx, ['user_code', 'book_code', 'Book-Rating']].values
val_ratings = combined.loc[val_idx, ['user_code', 'book_code', 'Book-Rating']].values
# print(train_ratings[:5])

# print(train_ratings)
# Convert to tensors (user, item, rating)

train_users = torch.tensor(train_ratings[:, 0], dtype=torch.long) # Bitch this is the user colume data

train_items = torch.tensor(train_ratings[:, 1], dtype=torch.long) # Bitch this is the book titel colume data

train_ratings_tensor = torch.tensor(train_ratings[:, 2], dtype=torch.float32) # Bitch this is the rating colume data

val_users = torch.tensor(val_ratings[:, 0], dtype=torch.long)
val_items = torch.tensor(val_ratings[:, 1], dtype=torch.long)
val_true = val_ratings[:, 2] 
# print("THOS OS OT")
# print(val_true)



# Model: Simple MF (embeddings + dot product)
class MF(nn.Module):
    def __init__(self, n_users, n_books, emb_dim=50):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.book_emb = nn.Embedding(n_books, emb_dim)
        self.fc = nn.Linear(emb_dim, 1)  # Optional bias layer
    
    def forward(self, u, i):
        u_e = self.user_emb(u) # fetches the 50-dimensional embeddings for the user
        i_e = self.book_emb(i) # fetches the 50-dimensional embeddings for the book
        return torch.sigmoid(self.fc((u_e * i_e))).squeeze() * 10  # Scale to 1-10

model = MF(n_users, n_books)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()



# Training loop (10 epochs; monitor loss)
epochs = 30
batch_size = 2048
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i in range(0, len(train_users), batch_size):
        batch_u = train_users[i:i+batch_size]
        batch_i = train_items[i:i+batch_size]
        batch_r = train_ratings_tensor[i:i+batch_size]
        
        pred = model(batch_u, batch_i)
        loss = criterion(pred, batch_r)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        
    print(f"Epoch {epoch+1}: Loss = {total_loss / (len(train_users) // batch_size):.4f}")

# Eval on val
model.eval()

with torch.no_grad():
    val_pred = model(val_users, val_items).numpy()
rmse = np.sqrt(mean_squared_error(val_true, val_pred))
print(f"Val RMSE: {rmse:.4f}")  # Aim <0.90

# Save model
torch.save(model.state_dict(), 'mf_baseline.pth')
print("Baseline trained & saved!")

# Quick rec example: For user 0, top-5 books (excluding known)
def get_recs(model, user_id, n_recs=5):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_id] * n_books)
        scores = model(user_tensor, torch.arange(n_books)).numpy()
    # Mask known (from matrix row)
    known = matrix[user_id].toarray().flatten() > 0
    scores[known] = -np.inf
    top_idx = np.argsort(scores)[-n_recs:][::-1]
    return [(book_mapping[idx], scores[idx]) for idx in top_idx]

user_id = 0  
recs = get_recs(model, user_id)
print(f"Top recs for user {user_mapping[user_id]}: {recs}")
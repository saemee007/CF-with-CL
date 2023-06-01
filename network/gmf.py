import torch
from torch import nn

class GMF(torch.nn.Module):
    def __init__(self, config, num_users, num_items):
        super(GMF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = config.latent_dim

        self.embedding_user = nn.Embedding(num_embeddings = self.num_users, embedding_dim = self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings = self.num_items, embedding_dim = self.latent_dim)

        self.linear = nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        user_item_product = torch.mul(user_embedding, item_embedding)
        
        logits = self.linear(user_item_product)
        rating = self.logistic(logits)
        
        return rating

    def init_weight(self):
        pass



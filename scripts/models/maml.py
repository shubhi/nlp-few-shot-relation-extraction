# scripts/models/maml.py
import torch
import torch.nn as nn
import torch.optim as optim

class MAML(nn.Module):
    def __init__(self, encoder, inner_lr=0.01, num_inner_steps=1):
        super(MAML, self).__init__()
        self.encoder = encoder
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, support_set, query_set, N, K, Q):
        support_inputs, support_targets = support_set
        query_inputs, query_targets = query_set

        # Initialize the model weights
        fast_weights = list(self.encoder.parameters())

        for _ in range(self.num_inner_steps):
            # Forward pass on the support set
            support_embeddings = self.encoder(support_inputs)
            support_loss = self.criterion(support_embeddings, support_targets)
            
            # Compute gradients and update the fast weights
            grads = torch.autograd.grad(support_loss, fast_weights, create_graph=True)
            fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]
        
        # Forward pass on the query set with the updated weights
        query_embeddings = self.encoder(query_inputs, params=fast_weights)
        query_loss = self.criterion(query_embeddings, query_targets)
        query_preds = torch.argmax(query_embeddings, dim=1)
        
        accuracy = (query_preds == query_targets).float().mean().item()
        
        return query_loss, accuracy

    def meta_update(self, outer_optimizer, query_loss):
        outer_optimizer.zero_grad()
        query_loss.backward()
        outer_optimizer.step()


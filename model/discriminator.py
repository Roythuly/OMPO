import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import autograd
from torch.optim.lr_scheduler import MultiStepLR

class Discriminator_SAS(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super(Discriminator_SAS, self).__init__()

        self.device = torch.device("cuda:{}".format(str(args.device)) if args.cuda else "cpu")

        self.hidden_size = args.hidden_size

        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim, self.hidden_size), nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(),
            nn.Linear(self.hidden_size, 1)).to(self.device)

        self.trunk.train()

        self.optimizer = Adam(self.trunk.parameters(), lr = args.lr)
        self.scedular = MultiStepLR(self.optimizer, milestones=[20000, 40000], gamma=0.3)
        

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         expert_next_state,
                         policy_state,
                         policy_action,
                         policy_next_state,
                         lambda_=10):
        # alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action, expert_next_state], dim=1)
        policy_data = torch.cat([policy_state, policy_action, policy_next_state], dim=1)

        alpha = torch.rand(expert_data.size(0), 1).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
    
    def compute_grad_pen_back(self,
                         expert_state,
                         expert_action,
                         expert_next_state,
                         policy_state,
                         policy_action,
                         policy_next_state,
                         lambda_=10):
        # alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([torch.unsqueeze(expert_state, dim=0), torch.unsqueeze(expert_action, dim=0), torch.unsqueeze(expert_next_state, dim=0)], dim=1)
        policy_data = torch.cat([torch.unsqueeze(policy_state, dim=0), torch.unsqueeze(policy_action, dim=0), torch.unsqueeze(policy_next_state, dim=0)], dim=1)

        alpha = torch.rand(expert_data.size(0), 1).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


    def update(self, expert_buffer, replay_buffer, batch_size):
        self.train()

        expert_state_batch, expert_action_batch, _, expert_next_state_batch, _ = expert_buffer.sample(batch_size=batch_size)
        expert_state_batch = torch.FloatTensor(expert_state_batch).to(self.device)
        expert_next_state_batch = torch.FloatTensor(expert_next_state_batch).to(self.device)
        expert_action_batch = torch.FloatTensor(expert_action_batch).to(self.device)

        state_batch, action_batch, _, next_state_batch, _ = replay_buffer.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        
        policy_d = self.trunk(torch.cat([state_batch, action_batch, next_state_batch], dim=1))
        expert_d = self.trunk(torch.cat([expert_state_batch, expert_action_batch, expert_next_state_batch], dim=1))
        
        policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

        expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
        
        gail_loss = expert_loss + policy_loss
        grad_pen = self.compute_grad_pen(expert_state_batch, expert_action_batch, expert_next_state_batch,
                                        state_batch, action_batch,next_state_batch)
        
        self.optimizer.zero_grad()
        (gail_loss + grad_pen).backward()
        self.optimizer.step()
        
        # self.scedular.step()
        
        loss = (gail_loss + grad_pen).item()
        
        return loss
    
    def update_back(self, expert_buffer, replay_buffer, batch_size):
        self.train()

        expert_state_batch, expert_action_batch, _, expert_next_state_batch, _ = expert_buffer.sample(batch_size=batch_size)
        expert_state_batch = torch.FloatTensor(expert_state_batch).to(self.device)
        expert_next_state_batch = torch.FloatTensor(expert_next_state_batch).to(self.device)
        expert_action_batch = torch.FloatTensor(expert_action_batch).to(self.device)

        state_batch, action_batch, _, next_state_batch, _ = replay_buffer.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)

        loss = 0
        n = 0

        for i in range(batch_size):
            policy_d = self.trunk(torch.cat([torch.unsqueeze(state_batch[i], dim=0), torch.unsqueeze(action_batch[i], dim=0), torch.unsqueeze(next_state_batch[i], dim=0)], dim = 1))
            expert_d = self.trunk(torch.cat([torch.unsqueeze(expert_state_batch[i],dim=0), torch.unsqueeze(expert_action_batch[i], dim=0), torch.unsqueeze(expert_next_state_batch[i], dim=0)], dim = 1))

            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            
            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state_batch[i], expert_action_batch[i], expert_next_state_batch[i],
                                             state_batch[i], action_batch[i],next_state_batch[i])

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        
        return loss / n

    def predict_reward(self, state, action, next_state):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action, next_state], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            return reward
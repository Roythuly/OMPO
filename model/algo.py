import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import copy
from .utils import soft_update, hard_update, orthogonal_regularization
from .model import GaussianPolicy, QNetwork, DeterministicPolicy

epsilon = 1e-6

class OMPO(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.exponent = args.exponent
        
        self.actor_loss = 0
        self.alpha_loss = 0
        self.alpha_tlogs = 0

        if self.exponent <= 1:
            raise ValueError('Exponent must be greather than 1, but received %f.' %
                            self.exponent)

        self.tomac_alpha = args.tomac_alpha

        self.f = lambda resid: torch.pow(torch.abs(resid), self.exponent) / self.exponent
        clip_resid = lambda resid: torch.clamp(resid, 0.0, 1e6)
        self.fgrad = lambda resid: torch.pow(clip_resid(resid), self.exponent - 1)

        self.device = torch.device("cuda:{}".format(str(args.device)) if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_optim_scheduler = MultiStepLR(self.critic_optim, milestones=[200000, 400000], gamma=0.2)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device) # initial alpha = 1.0
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
                self.alpha_optim_scheduler = MultiStepLR(self.alpha_optim, milestones=[100000, 200000], gamma=0.2)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
            self.policy_optim_scheduler = MultiStepLR(self.policy_optim, milestones=[100000, 200000], gamma=0.2)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def critic_mix(self, s, a):
        target_q1, target_q2 = self.critic_target(s, a)
        target_q = torch.min(target_q1, target_q2)
        q1, q2 = self.critic(s, a)
        return q1 * 0.05 + target_q * 0.95, q2 * 0.05 + target_q * 0.95
    
    def update_critic(self, discriminator, states, actions, next_states, rewards, masks, init_states, updates, writer):
        init_actions, _, _ = self.policy.sample(init_states)
        next_actions, next_log_probs, _ = self.policy.sample(next_states)
        
        # rewards = torch.clamp(rewards, 0, torch.inf)
        rewards = torch.clamp(rewards, epsilon, torch.inf)

        d_sas_rewards = discriminator.predict_reward(states, actions, next_states)
        
        writer.add_scalar('para/d_sas_reward', torch.mean(d_sas_rewards).item(), updates)

        # compute the reward
        # rewards = torch.log(rewards + epsilon * torch.ones(rewards.shape[0]).to(self.device)) - self.tomac_alpha * d_sas_rewards
        # rewards = torch.log(rewards) - self.tomac_alpha * d_sas_rewards
        rewards = rewards - self.tomac_alpha * d_sas_rewards
        
        # rewards -= self.tomac_alpha * d_sas_rewards

        with torch.no_grad():
            target_q1, target_q2 = self.critic_mix(next_states, next_actions)

            target_q1 = target_q1 - self.alpha * next_log_probs
            target_q2 = target_q2 - self.alpha * next_log_probs

            target_q1 = rewards + self.gamma * masks * target_q1
            target_q2 = rewards + self.gamma * masks * target_q2

        q1, q2 = self.critic(states, actions)
        init_q1, init_q2 = self.critic(init_states, init_actions)

        critic_loss1 = torch.mean(self.f(target_q1 - q1) + (1 - self.gamma) * init_q1 * self.tomac_alpha)
        critic_loss2 = torch.mean(self.f(target_q2 - q2) + (1 - self.gamma) * init_q2 * self.tomac_alpha)

        critic_loss = (critic_loss1 + critic_loss2)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return critic_loss.item()
    
    def update_actor(self, discriminator, states, actions, next_states, rewards, masks, init_states):
        init_actions, _, _ = self.policy.sample(init_states)
        next_actions, next_log_probs, _ = self.policy.sample(next_states)
        
        # rewards = torch.clamp(rewards, 0, torch.inf)
        rewards = torch.clamp(rewards, epsilon, torch.inf)

        d_sas_rewards = discriminator.predict_reward(states, actions, next_states)

        # compute the reward
        # rewards = torch.log(rewards + epsilon * torch.ones(rewards.shape[0]).to(self.device)) - self.tomac_alpha * d_sas_rewards
        # rewards = torch.log(rewards) - self.tomac_alpha * d_sas_rewards
        rewards = rewards - self.tomac_alpha * d_sas_rewards
        
        # rewards -= self.tomac_alpha * d_sas_rewards

        target_q1, target_q2 = self.critic_mix(next_states, next_actions)

        target_q1 = target_q1 - self.alpha * next_log_probs
        target_q2 = target_q2 - self.alpha * next_log_probs

        target_q1 = rewards + self.gamma * masks * target_q1
        target_q2 = rewards + self.gamma * masks * target_q2

        q1, q2 = self.critic(states, actions)
        init_q1, init_q2 = self.critic(init_states, init_actions)

        actor_loss1 = -torch.mean(self.fgrad(target_q1 - q1).detach() * (target_q1 - q1) + (1 - self.gamma) * init_q1 * self.tomac_alpha)
        actor_loss2 = -torch.mean(self.fgrad(target_q2 - q2).detach() * (target_q2 - q2) + (1 - self.gamma) * init_q2 * self.tomac_alpha)

        actor_loss = (actor_loss1 + actor_loss2) / 2.0
        # actor_loss += orthogonal_regularization(self.policy)

        self.policy_optim.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()
        
        _, log_probs, _ = self.policy.sample(states)

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            # self.alpha_optim_scheduler.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs
        
        return actor_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def update_parameters(self, initial_state_memory, memory, discriminator, batch_size, updates, writer):
        # Sample a batch from initial_state_memory
        initial_state_batch, _, _, _, _ = initial_state_memory.sample(batch_size=batch_size)
        initial_state_batch = torch.FloatTensor(initial_state_batch).to(self.device)
        
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        critic_loss = self.update_critic(discriminator, state_batch, action_batch, next_state_batch, reward_batch, mask_batch, initial_state_batch, updates, writer)
        # self.critic_optim_scheduler.step()

        if updates % self.target_update_interval == 0:
            self.actor_loss, self.alpha_loss, self.alpha_tlogs = self.update_actor(discriminator, state_batch, action_batch, next_state_batch, reward_batch, mask_batch, initial_state_batch)
            soft_update(self.critic_target, self.critic, self.tau)
            # self.policy_optim_scheduler.step()
        
        return critic_loss, self.actor_loss, self.alpha_loss, self.alpha_tlogs
    
    # Save model parameters
    def save_checkpoint(self, path, i_episode):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, path, i_episode, evaluate=False):
        ckpt_path = path + '/' + '{}.torch'.format(i_episode)
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()


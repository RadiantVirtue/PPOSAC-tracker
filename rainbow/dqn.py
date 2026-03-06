import torch
import numpy as np
import copy
from rainbow.network import Dueling_CNN_Net, CNN_Net


class DQN(object):
    def __init__(self, args):
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_soft_update = args.use_soft_update
        self.target_update_freq = args.target_update_freq
        self.update_count = 0

        self.grad_clip = args.grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_double = args.use_double
        self.use_dueling = args.use_dueling
        self.use_per = args.use_per
        self.use_n_steps = args.use_n_steps
        if self.use_n_steps:
            self.gamma = self.gamma ** args.n_steps

        if self.use_dueling:
            self.net = Dueling_CNN_Net(args)
        else:
            self.net = CNN_Net(args)

        self.target_net = copy.deepcopy(self.net)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def choose_action(self, state, epsilon):
        with torch.no_grad():
            state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
            q = self.net(state)
            if np.random.uniform() > epsilon:
                action = q.argmax(dim=-1).item()
            else:
                action = np.random.randint(0, self.action_dim)
            return action

    def learn(self, replay_buffer, total_steps):
        batch, batch_index, IS_weight = replay_buffer.sample(total_steps)

        with torch.no_grad():
            if self.use_double:
                a_argmax = self.net(batch['next_state']).argmax(dim=-1, keepdim=True)
                q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(batch['next_state']).gather(-1, a_argmax).squeeze(-1)
            else:
                q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(batch['next_state']).max(dim=-1)[0]

        q_current = self.net(batch['state']).gather(-1, batch['action']).squeeze(-1)
        td_errors = q_current - q_target

        if self.use_per:
            loss = (IS_weight * (td_errors ** 2)).mean()
            replay_buffer.update_batch_priorities(batch_index, td_errors.detach().cpu().numpy())
        else:
            loss = (td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.use_soft_update:
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

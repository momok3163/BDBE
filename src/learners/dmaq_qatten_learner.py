import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop
from utils.torch_utils import to_cuda
import numpy as np

import torch
import torch.optim as optim
from torch.nn import functional as F
import torch.nn as nn

class DMAQ_qattenLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args)
            elif args.mixer == 'dmaq_qatten':
                self.mixer = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.save_buffer_cnt = 0

        self.n_actions = self.args.n_actions


        self.start_anneal_time = 5e6
        self.init_anneal_time = False
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies

        self.eval_model_env = Predict_Network(args,
            args.state_shape + args.n_actions * args.n_agents, 128, args.enemy_shape * args.n_enemies)
        self.target_model_env = Predict_Network(args,
            args.state_shape + args.n_actions * args.n_agents, 128, args.enemy_shape * args.n_enemies)

        self.target_model_env.load_state_dict(self.eval_model_env.state_dict())
        self.Target_update = False
        if args.use_cuda:
            self.eval_model_env.cuda()
            self.target_model_env.cuda()


    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,
                  show_demo=False, save_data=None, show_v=False, save_buffer=False):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        actions_onehot = batch["actions_onehot"][:, :-1]
        visible = batch['visible_matrix'][:, :-1]

        b, t, a, _ = batch["obs"][:, :-1].shape

        model_s = th.cat((state, actions_onehot.reshape(b, t, -1)), dim=-1)
        model_opp_s = batch['extrinsic_state'][:, 1:]
        intrinsic_mask = mask.clone()

        # Calculate estimated Q-Values
        mac.init_hidden(batch.batch_size)
        mac_out = mac.forward(batch, batch.max_seq_length, batch_inf=True)

        if save_buffer:
            # curiosity_r=intrinsic_rewards.clone().detach().cpu().numpy()
            # rnd_r = rnd_intrinsic_rewards.clone().detach().cpu().numpy()
            # extrinsic_mac_out_save=extrinsic_mac_out.clone().detach().cpu().numpy()
            mac_out_save = mac_out.clone().detach().cpu().numpy()
            actions_save=actions.clone().detach().cpu().numpy()
            terminated_save=terminated.clone().detach().cpu().numpy()
            state_save=batch["state"][:, :-1].clone().detach().cpu().numpy()
            data_dic={# 'curiosity_r':curiosity_r,# 'extrinsic_Q':extrinsic_mac_out_save,
                        'control_Q':mac_out_save,'actions':actions_save,'terminated':terminated_save,
                        'state':state_save}

            self.save_buffer_cnt += self.args.save_buffer_cycle
            import os
            if not os.path.exists(self.args.save_buffer_path):
                os.makedirs(self.args.save_buffer_path)
            np.save(self.args.save_buffer_path + 'data_{}'.format(self.save_buffer_cnt), data_dic)
            print('save buffer ({}) at time{}'.format(batch.batch_size, self.save_buffer_cnt))
            return

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()
            # self.logger.log_stat('agent_1_%d_q_1' % save_data[0], np.squeeze(q_data)[0], t_env)
            # self.logger.log_stat('agent_2_%d_q_2' % save_data[1], np.squeeze(q_data)[1], t_env)

        # Calculate the Q-Values necessary for the target
        self.target_mac.init_hidden(batch.batch_size)
        target_mac_out = self.target_mac.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = to_cuda(th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)), self.args.device)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if mixer is not None:
            if self.args.mixer == "dmaq_qatten":
                ans_chosen, q_attend_regs, head_entropies = \
                    mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv, _, _ = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                      max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            else:
                ans_chosen = mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                if self.args.mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

        # Calculate 1-step Q-Learning targets

            agent_visible = visible[..., :self.n_agents]
            enemies_visible = visible[..., self.n_agents:]
            agent_alive = (agent_visible * (torch.eye(self.n_agents).to(agent_visible.device))).sum(dim=-1)
            agent_alive_mask = torch.bmm(agent_alive.reshape(-1, self.n_agents, 1),
                                         agent_alive.reshape(-1, 1, self.n_agents)).reshape(b, t, self.n_agents,
                                                                                            self.n_agents)
            enemies_visible = enemies_visible.unsqueeze(-1).repeat(1, 1, 1, 1, self.args.enemy_shape)
            enemies_visible = enemies_visible.reshape(b, t, self.n_agents, -1)
            mask_env = mask.clone().roll(dims=-2, shifts=-1)
            mask_env[:, -1, :] = 0
            # Opp_mse_Exp = self.target_model_env.get_log_pi(model_s, model_opp_s) * mask_env
            ac = avail_actions[:, :-1]
            ac = (1 - actions_onehot) * avail_actions[:, :-1]
            lazy_avoid_intrinsic,team_intrinsic, enemy_ate = self.target_model_env.get_opp_intrinsic(model_s.clone(), state.clone(),
                                                                                        actions_onehot,
                                                                                        enemies_visible, ac)

            lazy_avoid_intrinsic = lazy_avoid_intrinsic.clamp(max=self.args.i_one_clip)
            mean_rewards = rewards.sum() / mask.sum()
            lazy_avoid_intrinsic = lazy_avoid_intrinsic * agent_alive
            if not self.args.cuda_save:
                CDI =team_intrinsic.clamp(max=0.1)/100
            else:
                old_extrin_s = batch['extrinsic_state'][:, :-1]
                new_extrin_s = batch['extrinsic_state'][:, 1:]
                s_transition = (old_extrin_s - new_extrin_s) ** 2
                CDI = s_transition.sum(dim=-1).clamp(max=0.15).unsqueeze(-1) / 100

            IDI = (lazy_avoid_intrinsic.sum(dim=-1).unsqueeze(-1))
            CDI, IDI = CDI * intrinsic_mask, IDI * intrinsic_mask  # ()
            CDI=CDI.clamp(max=0.06)
            intrinsic = self.args.beta2 * CDI + self.args.beta1 * IDI
            intrinsic = intrinsic.clamp(max=self.args.itrin_two_clip)
            mean_alive = (agent_alive * terminated).sum(dim=-1).sum(dim=-1).mean()
            enemy_alive = (((batch['extrinsic_state'][:, 1:]) * terminated).sum(-2).reshape(b, self.n_enemies, 3)[
                               ..., 0] > 0).float().sum(-1).mean()
            if not self.init_anneal_time and mean_rewards > 0.00:
                self.init_anneal_time = True
                self.start_anneal_time = t_env
            if t_env > self.start_anneal_time and self.args.env_args['reward_sparse'] and self.args.anneal_intrin:
                intrinsic = max(1 - (
                        t_env - self.start_anneal_time) / self.args.anneal_speed, 0) * intrinsic

            rewards_new = rewards + intrinsic  # +intrinsic
            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards_new, terminated, mask, target_max_qvals, qvals,
                                                 self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards_new, terminated, mask, target_max_qvals,
                                                  self.args.n_agents, self.args.gamma, self.args.td_lambda)

        #targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        if show_v:
            mask_elems = mask.sum().item()

            actual_v = rewards.clone().detach()
            for t in reversed(range(rewards.shape[1] - 1)):
                actual_v[:, t] += self.args.gamma * actual_v[:, t + 1]
            self.logger.log_stat("test_actual_return", (actual_v * mask).sum().item() / mask_elems, t_env)

            self.logger.log_stat("test_q_taken_mean", (chosen_action_qvals * mask).sum().item() / mask_elems, t_env)
            return

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        if self.args.mixer == "dmaq_qatten":
            loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

        if self.args.is_prioritized_buffer:
            return masked_td_error ** 2, mask

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None, show_v=False):
        if self.args.is_prioritized_buffer:
            masked_td_error, mask = self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                           show_demo=show_demo, save_data=save_data, show_v=show_v)
        else:
            self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                           show_demo=show_demo, save_data=save_data, show_v=show_v)

        if hasattr(self.args, 'save_buffer') and self.args.save_buffer:
            if self.buffer.episodes_in_buffer - self.save_buffer_cnt >= self.args.save_buffer_cycle:
                if self.buffer.can_sample(self.args.save_buffer_cycle):
                    self.sub_train(self.buffer.sample(self.args.save_buffer_cycle, newest=True), t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                        show_demo=show_demo, save_data=save_data, show_v=show_v, save_buffer=True)
                else:
                    print('**' * 20, self.buffer.episodes_in_buffer, self.save_buffer_cnt)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        if self.args.is_prioritized_buffer:
            res = th.sum(masked_td_error, dim=(1, 2)) / th.sum(mask, dim=(1, 2))
            res = res.cpu().detach().numpy()
            return res

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        to_cuda(self.mac, self.args.device)
        to_cuda(self.target_mac, self.args.device)
        if self.mixer is not None:
            to_cuda(self.mixer, self.args.device)
            to_cuda(self.target_mixer, self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


class Predict_Network(nn.Module):

    def __init__(self,args, num_inputs, hidden_dim, num_outputs, lr=3e-4):
        super(Predict_Network, self).__init__()

        def weights_init_(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                torch.nn.init.constant_(m.bias, 0)

        self.hideen_dim = hidden_dim
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            num_layers=1,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.last_fc = nn.Linear(hidden_dim, num_outputs)
        self.args=args
        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        b, t, _ = input.shape
        hidden = torch.zeros((1, b, self.hideen_dim)).to(input.device)
        h1 = F.relu(self.linear1(input))
        hrnn, _ = self.rnn(h1, hidden)
        x = self.last_fc(hrnn)
        return x, hrnn

    def counterfactual(self, input, h):
        b, t, n_a, _ = input.shape
        input = input.reshape(b * t * n_a, 1, -1)
        h = h.reshape(1, b * t * n_a, -1)
        h1 = F.relu(self.linear1(input))
        hrnn, _ = self.rnn(h1, h)
        x = self.last_fc(hrnn)
        return x.reshape(b, t, n_a, -1)

    def get_log_pi(self, own_variable, other_variable):
        predict_variable, _ = self.forward(own_variable)
        log_prob = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def get_opp_intrinsic(self, s_a, s, a, enemies_visible, avail_u=None):
        b, t, n_agents, n_actions = a.shape

        p_s_a, h = self.forward(s_a)

        h_new = torch.zeros_like(h).to(h.device)
        h_new[:, 1:] = h[:, :-1]
        full_actions = torch.ones((b, t, n_agents, n_actions, n_actions)) * torch.eye(n_actions)
        full_actions = full_actions.type_as(s).to(a.device)
        full_s = s.unsqueeze(-2).repeat(1, 1, n_actions, 1)
        full_a = a.unsqueeze(-2).repeat(1, 1, 1, n_actions, 1)
        full_h = h_new.unsqueeze(-2).repeat(1, 1, n_actions, 1)
        intrinsic_1 = torch.zeros((b, t, n_agents)).to(a.device)
        Enemy = torch.zeros((b, t, n_agents, p_s_a.shape[-1])).to(a.device)
        if not self.args.cuda_save:
            sample_size=self.args.sample_size
            random_ = torch.rand(b, t, sample_size, n_agents, n_actions).type_as(s)*(avail_u.unsqueeze(-3))
            sample_a=torch.zeros_like(random_)
            values, indices = random_.topk(1, dim=-1, largest=True, sorted=True)
            random_=(random_==values).type_as(s)*(avail_u.unsqueeze(-3))
            random_full_s = s.unsqueeze(-2).repeat(1, 1, sample_size, 1)
            random_s_a=torch.cat((random_full_s,sample_a.reshape(b, t, sample_size, -1)),dim=-1)
            random_full_h = h_new.unsqueeze(-2).repeat(1, 1, sample_size, 1)
            s_enemy_visible=enemies_visible.sum(dim=-2).clamp(min=0,max=1)
            p_s_random = self.counterfactual(random_s_a, random_full_h).mean(dim=-2)
            ATE_enemy_joint=s_enemy_visible * F.mse_loss(p_s_random, p_s_a, reduction='none')
            intrinsic_2=ATE_enemy_joint.sum(dim=-1).unsqueeze(-1)
        else:
            intrinsic_2 =torch.zeros((b, t,1))
        if avail_u == None:
            avail_u = torch.ones_like(a).type_as(a)
        for i in range(n_agents):
            ATE_a = (full_a.clone())
            ATE_a[..., i, :, :] = full_actions[..., i, :, :]
            ATE_a = ATE_a.transpose(-2, -3).reshape(b, t, n_actions, -1)
            s_a_noi = torch.cat((full_s, ATE_a), dim=-1)
            p_s_a_noi = self.counterfactual(s_a_noi, full_h)
            p_s_a_noi = p_s_a_noi * (avail_u[..., i, :].unsqueeze(-1))
            p_s_a_mean_noi = p_s_a_noi.sum(dim=-2) / (avail_u[..., i, :].sum(dim=-1).unsqueeze(-1) + 1e-6)
            ATE_enemy_i = enemies_visible[..., i, :] * F.mse_loss(p_s_a_mean_noi, p_s_a, reduction='none')
            # ATE_enemy_i=enemies_visible[...,i,:]*torch.abs(p_s_a_mean_noi-p_s_a)
            ATE_i = ATE_enemy_i.sum(dim=-1)
            intrinsic_1[..., i] = ATE_i
            Enemy[..., i, :] = ATE_enemy_i
        return intrinsic_1,intrinsic_2, Enemy

    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable, _ = self.forward(own_variable)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

            return loss.to('cpu').detach().item()

        return None

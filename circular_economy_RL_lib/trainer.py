import torch
import os
import numpy as np
from simulator import Manufacturing_Simulator
from agent import PPOAgent, OptimalFollowerValueEstimator
from config import config, LEADER, BUYER, TRANSFORM, stages
from torch.utils.tensorboard import SummaryWriter

class RunningMeanStd:
    """
    Tracks running mean and variance of observations using Welford's algorithm
    to normalize state features dynamically into a stable [-1.0, 1.0] range.
    """
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
            batch_mean = x[0]
            batch_var = np.zeros_like(batch_mean)
            batch_count = 1
            
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
        
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

class BilevelTrainer:
    def __init__(self):
        for key, value in config.items():
            setattr(self, key, value)

        self.env = Manufacturing_Simulator()
        self.result_folder = './result/bilevel_ppo'
        os.makedirs(self.result_folder + '/log', exist_ok=True)
        os.makedirs(self.result_folder + '/debug', exist_ok=True)
        self.writer = SummaryWriter(self.result_folder + '/log')

        # Dimensional setup and validation (14, 1824, 1908, 2088)
        self.seller_obs_dim = self.num_commodities * self.history_length * (6 + self.num_agents * 8) + 2 * self.num_commodities # 1824
        self.buyer_obs_dim = self.seller_obs_dim + self.num_commodities + 2 * self.num_commodities * self.num_agents # 1908
        self.trans_obs_dim = self.buyer_obs_dim + self.num_commodities * (4 * self.num_agents + 3) # 2088
        
        self.buyer_act_dim = 2 * self.num_commodities * (self.num_agents - 1) + self.num_commodities # 60
        self.trans_act_dim = 2 * self.num_commodities # 24
        self.leader_act_dim = 3  
        self.leader_obs_dim = self.num_commodities + 2 # 14

        # Initialize Leader Agent with continuous bounds [0.1, 10.0] matching the active physical envelope
        self.leader_agent = PPOAgent(
            self.leader_obs_dim, self.leader_act_dim, f"{self.result_folder}/chpkt/leader", 
            lr=self.lr_leader, min_val=0.1, max_val=10.0
        )
        
        # Instantiate separate estimator networks per agent to prevent catastrophic interference
        self.best_response_estimators = [
            OptimalFollowerValueEstimator(self.leader_act_dim + self.buyer_obs_dim)
            for _ in range(self.num_agents)
        ]
        
        # Initialize running mean-std normalizers for each stage
        self.leader_rms = RunningMeanStd(shape=(self.leader_obs_dim,))
        self.buyer_rms = RunningMeanStd(shape=(self.num_agents, self.buyer_obs_dim))
        self.trans_rms = RunningMeanStd(shape=(self.num_agents, self.trans_obs_dim))
        
        # Follower agents maintain standard [0.01, 100.0] action bounds
        self.buyer_agents = [
            PPOAgent(
                self.buyer_obs_dim, self.buyer_act_dim, f"{self.result_folder}/chpkt/buyer_{ag}", 
                lr=self.lr_follower, min_val=0.01, max_val=100.0
            ) for ag in range(self.num_agents)
        ]
        self.trans_agents = [
            PPOAgent(
                self.trans_obs_dim, self.trans_act_dim, f"{self.result_folder}/chpkt/trans_{ag}", 
                lr=self.lr_follower, min_val=0.01, max_val=100.0
            ) for ag in range(self.num_agents)
        ]

    def rollout(self):
        batch_obs = [[] for _ in stages]
        batch_acts = [[] for _ in stages]
        batch_log_probs = [[] for _ in stages]
        batch_rews = [[] for _ in stages]
        batch_active_phi = []  # Tracks active smoothed parameters for estimator fitting
        batch_lens = []

        t = 0
        while t < self.num_steps:
            ep_rews = [[] for _ in stages]
            s_leader, s_follower = self.env.reset()
            done = False
            
            for ep_t in range(self.episode_length):
                t += 1
                
                # --- Step 1: Upper-Level Leader decides rules (phi) ---
                batch_obs[LEADER].append(s_leader)
                
                # Normalize leader observation dynamically
                self.leader_rms.update(s_leader)
                s_leader_norm = self.leader_rms.normalize(s_leader)
                
                phi, log_p_leader = self.leader_agent.get_action(s_leader_norm)
                batch_acts[LEADER].append(phi)
                batch_log_probs[LEADER].append(log_p_leader)
                
                s_follower = self.env.step_sell(phi)
                
                # Track the active smoothed market parameters (phi_active)
                batch_active_phi.append(self.env.active_phi)
                
                s_buyer = self.env.get_buyer_state(s_follower)

                # --- Step 2: Followers decide trading quantities ---
                batch_obs[BUYER].append(s_buyer)
                
                self.buyer_rms.update(s_buyer)
                s_buyer_norm = self.buyer_rms.normalize(s_buyer)
                
                buyer_actions = []
                log_p_buyers = []
                for ag in range(self.num_agents):
                    act_b, log_p_b = self.buyer_agents[ag].get_action(s_buyer_norm[ag])
                    buyer_actions.append(act_b)
                    log_p_buyers.append(log_p_b)
                
                buyer_actions = np.array(buyer_actions)
                batch_acts[BUYER].append(buyer_actions)
                batch_log_probs[BUYER].append(log_p_buyers)

                rew_b = self.env.step_buy(buyer_actions)
                s_trans = self.env.get_trans_state(s_buyer)

                # --- Step 3: Followers decide transformation/utility ---
                batch_obs[TRANSFORM].append(s_trans)
                
                self.trans_rms.update(s_trans)
                s_trans_norm = self.trans_rms.normalize(s_trans)
                
                trans_actions = []
                log_p_trans = []
                for ag in range(self.num_agents):
                    act_t, log_p_t = self.trans_agents[ag].get_action(s_trans_norm[ag])
                    trans_actions.append(act_t)
                    log_p_trans.append(log_p_t)
                
                trans_actions = np.array(trans_actions)
                batch_acts[TRANSFORM].append(trans_actions)
                batch_log_probs[TRANSFORM].append(log_p_trans)

                s_leader_next, s_follower_next, rew_t, rew_l, done = self.env.step_trans(trans_actions)

                ep_rews[LEADER].append(rew_l)
                ep_rews[BUYER].append(rew_b)
                ep_rews[TRANSFORM].append(rew_t)

                s_leader = s_leader_next
                s_follower = s_follower_next
                s_buyer = self.env.get_buyer_state(s_follower)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            for stage in stages:
                batch_rews[stage].append(ep_rews[stage])

        tensor_obs = [torch.tensor(np.array(obs), dtype=torch.float) for obs in batch_obs]
        tensor_acts = [torch.tensor(np.array(act), dtype=torch.float) for act in batch_acts]
        tensor_log_probs = [torch.tensor(np.array(lp), dtype=torch.float) for lp in batch_log_probs]

        batch_rtgs, batch_rets = self.compute_rtgs(batch_obs, batch_rews, batch_lens)
        return tensor_obs, tensor_acts, tensor_log_probs, batch_rtgs, batch_rets, batch_lens, batch_active_phi

    def compute_rtgs(self, batch_obs, batch_rews, batch_lens):
        batch_rtgs = [[] for _ in stages]
        batch_rets = [[] for _ in stages]
        for stage in stages:
            ep_ret_list = []
            flat_rtg_list = []
            
            # Retrieve raw observations and normalize using running statistics
            raw_obs = np.array(batch_obs[stage])
            if stage == LEADER:
                norm_obs = self.leader_rms.normalize(raw_obs)
            elif stage == BUYER:
                norm_obs = self.buyer_rms.normalize(raw_obs)
            else:
                norm_obs = self.trans_rms.normalize(raw_obs)
                
            obs_tensor = torch.tensor(norm_obs, dtype=torch.float32)
            num_episodes = len(batch_rews[stage])
            
            with torch.no_grad():
                if stage == LEADER:
                    V = self.leader_agent.critic(obs_tensor).squeeze(-1)
                elif stage == BUYER:
                    V = torch.stack([self.buyer_agents[ag].critic(obs_tensor[:, ag, :]).squeeze(-1) for ag in range(self.num_agents)], dim=-1)
                else:
                    V = torch.stack([self.trans_agents[ag].critic(obs_tensor[:, ag, :]).squeeze(-1) for ag in range(self.num_agents)], dim=-1)
            
            V = V.cpu().numpy()
            
            V_episodes = []
            curr_idx = 0
            for ep_len_val in batch_lens:
                V_episodes.append(V[curr_idx:curr_idx + ep_len_val])
                curr_idx += ep_len_val
            
            for ep in range(num_episodes):
                rews = np.array(batch_rews[stage][ep])
                vals = V_episodes[ep]
                
                advantages = np.zeros_like(rews)
                gae = np.zeros_like(rews[0])
                
                for t in reversed(range(len(rews))):
                    next_val = vals[t+1] if t + 1 < len(rews) else np.zeros_like(rews[0])
                    delta = rews[t] + self.gamma * next_val - vals[t]
                    gae = delta + self.gamma * 0.95 * gae
                    advantages[t] = gae
                
                rtg = advantages + vals
                flat_rtg_list.extend(rtg)
                ep_ret_list.append(rtg[0])
                
            batch_rtgs[stage] = torch.tensor(np.array(flat_rtg_list), dtype=torch.float)
            batch_rets[stage] = np.mean(ep_ret_list, axis=0)
            
        return batch_rtgs, batch_rets

    def learn(self):
        t_so_far = 0
        i_so_far = 0

        while i_so_far < self.num_epochs:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_rets, batch_lens, batch_active_phi = self.rollout()
            t_so_far += 1000  # Budgeted step count
            i_so_far += 1

            # =================================================================
            # INTEGRATED BRL FEATURE ALIGNMENT & RELATIVE ERROR DIAGNOSTICS
            # =================================================================
            print("\n" + "="*70)
            print("          BRL ESTIMATOR FEATURE ALIGNMENT AUDIT (EPOCH {})".format(i_so_far))
            print("="*70)
            with torch.no_grad():
                raw_phi = batch_acts[LEADER].cpu().numpy()
                active_phi_history = np.array([self.env.active_phi for _ in range(len(raw_phi))]) if self.env.active_phi is not None else np.zeros_like(raw_phi)
                
                print("Leader Action Parameter Mismatch Analysis:")
                print(f"  Raw Policy phi_0 (Mean/Std)   : {np.mean(raw_phi[:, 0]):.4f} / {np.std(raw_phi[:, 0]):.4f}")
                print(f"  Active Smoothed phi_0 (Mean/Std): {np.mean(active_phi_history[:, 0]):.4f} / {np.std(active_phi_history[:, 0]):.4f}")
                print(f"  Action Variance Discrepancy Ratio (Raw / Active): {np.std(raw_phi[:, 0]) / (np.std(active_phi_history[:, 0]) + 1e-10):.2f}")

                # Safe on-the-fly normalization for diagnostic relative error check
                raw_obs_buyer_diag = batch_obs[BUYER].cpu().numpy()
                norm_obs_buyer_diag = self.buyer_rms.normalize(raw_obs_buyer_diag)
                norm_obs_buyer_tensor = torch.tensor(norm_obs_buyer_diag, dtype=torch.float32).to(batch_obs[BUYER].device)

                flat_active_phi_diag = torch.tensor(np.array(batch_active_phi), dtype=torch.float32).reshape(-1, self.leader_act_dim).to(batch_obs[BUYER].device)

                print("\nEstimator Relative Error Analysis (Active phi):")
                for ag in range(self.num_agents):
                    flat_state_norm = norm_obs_buyer_tensor[:, ag]
                    estimator_input = torch.cat([flat_active_phi_diag, flat_state_norm], dim=-1)
                    target_returns = batch_rtgs[BUYER][:, ag]
                    
                    v_star = self.best_response_estimators[ag](estimator_input).squeeze()
                    
                    abs_errors = torch.abs(v_star - target_returns)
                    mean_target_magnitude = torch.mean(torch.abs(target_returns)) + 1e-10
                    relative_error = (torch.mean(abs_errors) / mean_target_magnitude).item() * 100.0
                    
                    print(f"  Estimator {ag} - V* Mean: {v_star.mean().item():.4f} | Target Mean: {target_returns.mean().item():.4f} | Relative Error: {relative_error:.2f}%")
            print("="*70 + "\n")
            # =================================================================

            # --- Update Lower-Level Follower Policies ---
            raw_obs_buyer = batch_obs[BUYER].cpu().numpy()
            self.buyer_rms.update(raw_obs_buyer)
            norm_obs_buyer = torch.tensor(self.buyer_rms.normalize(raw_obs_buyer), dtype=torch.float32).to(batch_obs[BUYER].device)

            raw_obs_trans = batch_obs[TRANSFORM].cpu().numpy()
            self.trans_rms.update(raw_obs_trans)
            norm_obs_trans = torch.tensor(self.trans_rms.normalize(raw_obs_trans), dtype=torch.float32).to(batch_obs[TRANSFORM].device)

            for ag in range(self.num_agents):
                # Update Buyer Agents with normalized states
                a_loss_b, c_loss_b = self.buyer_agents[ag].learn(
                    norm_obs_buyer[:, ag], batch_acts[BUYER][:, ag], 
                    batch_log_probs[BUYER][:, ag], batch_rtgs[BUYER][:, ag], 10
                )
                self.writer.add_scalar(f'buyer_actor_loss_agent_{ag}', a_loss_b, t_so_far)
                self.writer.add_scalar(f'buyer_critic_loss_agent_{ag}', c_loss_b, t_so_far)

                # Update Transformer Agents with normalized states
                a_loss_t, c_loss_t = self.trans_agents[ag].learn(
                    norm_obs_trans[:, ag], batch_acts[TRANSFORM][:, ag], 
                    batch_log_probs[TRANSFORM][:, ag], batch_rtgs[TRANSFORM][:, ag], 10
                )
                self.writer.add_scalar(f'trans_actor_loss_agent_{ag}', a_loss_t, t_so_far)
                self.writer.add_scalar(f'trans_critic_loss_agent_{ag}', c_loss_t, t_so_far)

            # --- Update Best-Response Value Estimators ---
            flat_active_phi = torch.tensor(np.array(batch_active_phi), dtype=torch.float32).reshape(-1, self.leader_act_dim).to(batch_obs[BUYER].device)
            for ag in range(self.num_agents):
                flat_state_norm = norm_obs_buyer[:, ag]
                estimator_input = torch.cat([flat_active_phi, flat_state_norm], dim=-1)
                target_returns = batch_rtgs[BUYER][:, ag]
                
                loss_est = self.best_response_estimators[ag].update(estimator_input, target_returns)
                self.writer.add_scalar(f'best_response_est_loss_agent_{ag}', loss_est, t_so_far)

            # --- Update Upper-Level Leader Policy ---
            if i_so_far % self.leader_update_frequency == 0:
                penalties = []
                flat_active_phi = torch.tensor(np.array(batch_active_phi), dtype=torch.float32).reshape(-1, self.leader_act_dim).to(batch_obs[BUYER].device)
                for ag in range(self.num_agents):
                    flat_state_norm = norm_obs_buyer[:, ag]
                    estimator_input = torch.cat([flat_active_phi, flat_state_norm], dim=-1)
                    
                    v_star = self.best_response_estimators[ag](estimator_input).squeeze().detach()
                    v_actual = batch_rtgs[BUYER][:, ag]
                    penalties.append(v_star - v_actual)
                
                total_penalty = torch.stack(penalties, dim=0).mean(dim=0).unsqueeze(-1).detach()
                
                # Soft penalty clipping
                clamped_penalty = torch.clamp(total_penalty, min=-10.0, max=10.0)
                penalized_rtgs = batch_rtgs[LEADER] - self.lambda_penalty * clamped_penalty

                # Update Leader Agent with normalized states (retaining raw batch_acts for standard PPO math)
                raw_obs_leader = batch_obs[LEADER].cpu().numpy()
                self.leader_rms.update(raw_obs_leader)
                norm_obs_leader = torch.tensor(self.leader_rms.normalize(raw_obs_leader), dtype=torch.float32).to(batch_obs[LEADER].device)

                a_loss_l, c_loss_l = self.leader_agent.learn(
                    norm_obs_leader, batch_acts[LEADER], 
                    batch_log_probs[LEADER], penalized_rtgs, 10
                )
                self.writer.add_scalar('leader_actor_loss', a_loss_l, t_so_far)
                self.writer.add_scalar('leader_critic_loss', c_loss_l, t_so_far)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.writer.add_scalar('leader_avg_return', np.mean(batch_rets[LEADER]), t_so_far)
            self.writer.add_scalar('buyer_avg_return', np.mean(batch_rets[BUYER]), t_so_far)
            self.writer.add_scalar('trans_avg_return', np.mean(batch_rets[TRANSFORM]), t_so_far)
            
            print(f"Epoch {i_so_far}/{self.num_epochs} Done. Leader Return: {np.mean(batch_rets[LEADER]):.5f}")

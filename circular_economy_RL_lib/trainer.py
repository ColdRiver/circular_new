import torch
import os
import numpy as np
from simulator import Manufacturing_Simulator
from agent import PPOAgent, OptimalFollowerValueEstimator
from config import config, LEADER, BUYER, TRANSFORM, stages
from torch.utils.tensorboard import SummaryWriter

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

        self.leader_agent = PPOAgent(self.leader_obs_dim, self.leader_act_dim, f"{self.result_folder}/chpkt/leader", lr=self.lr_leader)
        self.buyer_agents = [PPOAgent(self.buyer_obs_dim, self.buyer_act_dim, f"{self.result_folder}/chpkt/buyer_{ag}", lr=self.lr_follower) for ag in range(self.num_agents)]
        self.trans_agents = [PPOAgent(self.trans_obs_dim, self.trans_act_dim, f"{self.result_folder}/chpkt/trans_{ag}", lr=self.lr_follower) for ag in range(self.num_agents)]
        
        self.best_response_estimator = OptimalFollowerValueEstimator(self.leader_act_dim + self.buyer_obs_dim)

    def rollout(self):
        batch_obs = [[] for _ in stages]
        batch_acts = [[] for _ in stages]
        batch_log_probs = [[] for _ in stages]
        batch_rews = [[] for _ in stages]
        batch_lens = []

        t = 0
        while t < self.num_steps:
            ep_rews = [[] for _ in stages]
            s_leader, s_follower = self.env.reset()
            done = False
            
            for ep_t in range(self.episode_length):
                t += 1
                
                # --- Upper-Level Leader decides rules (phi) ---
                # Step-by-step updates for mathematically valid importance ratios
                batch_obs[LEADER].append(s_leader)
                phi, log_p_leader = self.leader_agent.get_action(s_leader / 100.0)
                batch_acts[LEADER].append(phi)
                batch_log_probs[LEADER].append(log_p_leader)
                
                # Update follower environment with regulatory state
                s_follower = self.env.step_sell(phi)

                # Reconstruct step 2 buyer states (1908 dimensions)
                s_buyer = self.env.get_buyer_state(s_follower)

                # --- Step 2: Followers decide trading quantities ---
                batch_obs[BUYER].append(s_buyer)
                buyer_actions = []
                log_p_buyers = []
                for ag in range(self.num_agents):
                    act_b, log_p_b = self.buyer_agents[ag].get_action(s_buyer[ag] / 100.0)
                    buyer_actions.append(act_b)
                    log_p_buyers.append(log_p_b)
                
                buyer_actions = np.array(buyer_actions)
                batch_acts[BUYER].append(buyer_actions)
                batch_log_probs[BUYER].append(log_p_buyers)

                # Step the buying environment
                rew_b = self.env.step_buy(buyer_actions)

                # Reconstruct step 3 transformer states (2088 dimensions)
                s_trans = self.env.get_trans_state(s_buyer)

                # --- Step 3: Followers decide transformation/utility ---
                batch_obs[TRANSFORM].append(s_trans)
                trans_actions = []
                log_p_trans = []
                for ag in range(self.num_agents):
                    act_t, log_p_t = self.trans_agents[ag].get_action(s_trans[ag] / 100.0)
                    trans_actions.append(act_t)
                    log_p_trans.append(log_p_t)
                
                trans_actions = np.array(trans_actions)
                batch_acts[TRANSFORM].append(trans_actions)
                batch_log_probs[TRANSFORM].append(log_p_trans)

                # Step the transformation environment
                s_leader_next, s_follower_next, rew_t, rew_l, done = self.env.step_trans(trans_actions)

                ep_rews[LEADER].append(rew_l)
                ep_rews[BUYER].append(rew_b)
                ep_rews[TRANSFORM].append(rew_t)

                s_leader = s_leader_next
                s_follower = s_follower_next

                if done:
                    break

            batch_lens.append(ep_t + 1)
            for stage in stages:
                batch_rews[stage].append(ep_rews[stage])

        tensor_obs = [torch.tensor(np.array(obs), dtype=torch.float) for obs in batch_obs]
        tensor_acts = [torch.tensor(np.array(act), dtype=torch.float) for act in batch_acts]
        tensor_log_probs = [torch.tensor(np.array(lp), dtype=torch.float) for lp in batch_log_probs]

        # Evaluate GAE-Lambda returns
        batch_rtgs, batch_rets = self.compute_rtgs(batch_obs, batch_rews)
        return tensor_obs, tensor_acts, tensor_log_probs, batch_rtgs, batch_rets, batch_lens

    def compute_rtgs(self, batch_obs, batch_rews):
        """
        Calculates GAE-Lambda advantages and value targets
        """
        batch_rtgs = [[] for _ in stages]
        batch_rets = [[] for _ in stages]
        for stage in stages:
            ep_rtg_list = []
            ep_ret_list = []
            
            obs_tensor = torch.tensor(np.array(batch_obs[stage]), dtype=torch.float32)
            num_episodes, ep_len = len(batch_rews[stage]), len(batch_rews[stage][0])
            
            with torch.no_grad():
                if stage == LEADER:
                    V = self.leader_agent.critic(obs_tensor.view(-1, self.leader_obs_dim)).view(num_episodes, ep_len)
                elif stage == BUYER:
                    V = torch.stack([self.buyer_agents[ag].critic(obs_tensor[:, :, ag] / 100.0).squeeze(-1) for ag in range(self.num_agents)], dim=-1)
                else:
                    V = torch.stack([self.trans_agents[ag].critic(obs_tensor[:, :, ag] / 100.0).squeeze(-1) for ag in range(self.num_agents)], dim=-1)
            
            V = V.cpu().numpy()
            
            for ep in range(num_episodes):
                rews = np.array(batch_rews[stage][ep])
                vals = V[ep]
                
                advantages = np.zeros_like(rews)
                gae = np.zeros_like(rews[0])
                
                for t in reversed(range(ep_len)):
                    next_val = vals[t+1] if t + 1 < ep_len else np.zeros_like(rews[0])
                    delta = rews[t] + self.gamma * next_val - vals[t]
                    gae = delta + self.gamma * 0.95 * gae
                    advantages[t] = gae
                
                rtg = advantages + vals
                ep_rtg_list.append(rtg)
                ep_ret_list.append(rtg[0])
            
            flat_rtg = torch.tensor(np.array(ep_rtg_list), dtype=torch.float).reshape(-1, self.num_agents if stage != LEADER else 1)
            batch_rtgs[stage] = flat_rtg
            batch_rets[stage] = np.mean(ep_ret_list, axis=0)
        return batch_rtgs, batch_rets

    def learn(self):
        total_timesteps = self.num_steps * self.num_epochs
        t_so_far = 0
        i_so_far = 0

        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_rets, batch_lens = self.rollout()
            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            # --- Update Lower-Level Follower Policies ---
            for ag in range(self.num_agents):
                # Update Buyer Agents
                a_loss_b, c_loss_b = self.buyer_agents[ag].learn(
                    batch_obs[BUYER][:, ag] / 100.0, batch_acts[BUYER][:, ag], 
                    batch_log_probs[BUYER][:, ag], batch_rtgs[BUYER][:, ag], 10
                )
                self.writer.add_scalar(f'buyer_actor_loss_agent_{ag}', a_loss_b, t_so_far)
                self.writer.add_scalar(f'buyer_critic_loss_agent_{ag}', c_loss_b, t_so_far)

                # Update Transformer Agents
                a_loss_t, c_loss_t = self.trans_agents[ag].learn(
                    batch_obs[TRANSFORM][:, ag] / 100.0, batch_acts[TRANSFORM][:, ag], 
                    batch_log_probs[TRANSFORM][:, ag], batch_rtgs[TRANSFORM][:, ag], 10
                )
                self.writer.add_scalar(f'trans_actor_loss_agent_{ag}', a_loss_t, t_so_far)
                self.writer.add_scalar(f'trans_critic_loss_agent_{ag}', c_loss_t, t_so_far)

            # --- Update Best-Response Value Estimator ---
            for ag in range(self.num_agents):
                flat_phi = batch_acts[LEADER].reshape(-1, self.leader_act_dim)
                flat_state = batch_obs[BUYER][:, ag] / 100.0
                estimator_input = torch.cat([flat_phi, flat_state], dim=-1)
                target_returns = batch_rtgs[BUYER][:, ag]
                
                loss_est = self.best_response_estimator.update(estimator_input, target_returns)
                self.writer.add_scalar(f'best_response_est_loss_agent_{ag}', loss_est, t_so_far)

            # --- Update Upper-Level Leader Policy ---
            if i_so_far % self.leader_update_frequency == 0:
                penalties = []
                for ag in range(self.num_agents):
                    flat_phi = batch_acts[LEADER].reshape(-1, self.leader_act_dim)
                    flat_state = batch_obs[BUYER][:, ag] / 100.0
                    estimator_input = torch.cat([flat_phi, flat_state], dim=-1)
                    
                    v_star = self.best_response_estimator(estimator_input).squeeze().detach()
                    v_actual = batch_rtgs[BUYER][:, ag]
                    penalties.append(v_star - v_actual)
                
                total_penalty = torch.stack(penalties, dim=0).mean(dim=0).unsqueeze(-1).detach()
                
                # Soft penalty clipping
                clamped_penalty = torch.clamp(total_penalty, min=-2.0, max=2.0)
                penalized_rtgs = batch_rtgs[LEADER] - self.lambda_penalty * clamped_penalty

                a_loss_l, c_loss_l = self.leader_agent.learn(
                    batch_obs[LEADER] / 100.0, batch_acts[LEADER], 
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

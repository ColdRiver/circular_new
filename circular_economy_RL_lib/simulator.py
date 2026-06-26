import numpy as np
from config import config, init_historical_data
from surrogate_models import SurrogateModel

class Manufacturing_Simulator:
    """
    Bilevel Environment Class with Temporal Price Inertia
    -- reset: returns the global leader state and decentralized follower states
    -- step_sell: receives raw phi, filters it via a low-pass filter, and establishes prices
    -- get_buyer_state: constructs the 1908-dimension state vector for step 2 buyers
    -- step_buy: computes P2P transactions and returns follower rewards
    -- get_trans_state: constructs the 2088-dimension state vector for step 3 transformers
    -- step_trans: evaluates non-linear surrogate networks and transitions inventories
    """
    def __init__(self):
        for key, value in config.items():
            setattr(self, key, value)
        self.surrogate_model = SurrogateModel()
        
    def reset(self):
        self.t = self.history_length
        self.active_phi = None  # Reset the low-pass filter for the new episode
        
        data_length = self.history_length + self.episode_length + 1
        general_shape = (self.num_commodities, data_length)
        individual_shape = (self.num_agents, self.num_commodities, data_length)
        pair_shape = (self.num_agents, self.num_agents, self.num_commodities, data_length)

        self.spot_price = np.zeros(shape=general_shape)
        self.uc_p = np.zeros(shape=general_shape)
        self.tx_p = np.zeros(shape=general_shape)

        self.price = np.zeros(shape=individual_shape)
        self.waste_price = np.zeros(shape=individual_shape)

        self.q = np.zeros(shape=pair_shape)
        self.waste_q = np.zeros(shape=pair_shape)
        self.spot_q = np.zeros(shape=individual_shape)

        self.actual_d = np.zeros(shape=pair_shape)
        self.waste_actual_d = np.zeros(shape=pair_shape)

        self.inv = np.zeros(shape=individual_shape) 
        self.waste_inv = np.zeros(shape=individual_shape) 
        self.inv_buy = np.zeros(shape=individual_shape) 
        self.waste_inv_buy = np.zeros(shape=individual_shape) 

        self.eco_u = np.zeros(shape=individual_shape)
        self.tx_u = np.zeros(shape=individual_shape)
        self.wastewater = np.zeros(shape=(1, 1, data_length))

        historical_data = init_historical_data()
        self.spot_price = np.repeat(historical_data['spot_price'], repeats=data_length, axis=1)
        self.uc_p = self.UC * self.spot_price
        self.tx_p = self.TX_P * self.spot_price
        self.inv[:, :, self.t] = self.inv[:, :, self.t] + self.INIT_INV
        self.waste_inv = self.inv.copy()
        self.init_inv = self.inv[:, :, self.t].copy()

        return self.get_leader_state(), self.get_follower_state()
    
    def get_leader_state(self):
        avg_spot = np.mean(self.spot_price[:, self.t-self.history_length:self.t], axis=1)
        total_waste_landfilled = np.sum(self.spot_q[:, :, self.t-1]) if self.t > self.history_length else 0.
        total_freshwater = np.sum(self.inv[:, 0, self.t])
        return np.concatenate((avg_spot, [total_waste_landfilled, total_freshwater]))

    def get_follower_state(self):
        follower_states = []
        start_time = self.t - self.history_length
        for n in range(self.num_agents):
            p = self.spot_price[..., start_time:self.t].flatten()
            e = self.price[..., start_time:self.t].flatten()
            ew = self.waste_price[..., start_time:self.t].flatten()

            q = self.q[:, n, :, start_time:self.t].flatten()
            qw = self.waste_q[:, n, :, start_time:self.t].flatten()

            q_ = self.q[n, :, :, start_time:self.t].flatten()
            qw_ = self.waste_q[n, :, :, start_time:self.t].flatten()

            qs = self.spot_q[n, :, start_time:self.t].flatten()

            d = self.actual_d[n, :, :, start_time:self.t].flatten()
            dw = self.waste_actual_d[n, :, :, start_time:self.t].flatten()

            I = self.inv[n, :, start_time:self.t+1].flatten()
            Iw = self.waste_inv[n, :, start_time:self.t+1].flatten()

            u_eco = self.eco_u[n, :, start_time:self.t].flatten()
            u_tx = self.tx_u[n, :, start_time:self.t].flatten()   
            
            state_flat = np.concatenate((p, e, ew, q, qw, q_, qw_, qs, d, dw, I, Iw, u_eco, u_tx))
            follower_states.append(state_flat)

        return np.array(follower_states)

    def get_buyer_state(self, s_follower):
        buyer_states = []
        for n in range(self.num_agents):
            state_flat = s_follower[n]
            state_flat = np.concatenate((
                state_flat, 
                self.spot_price[:, self.t], 
                self.price[:, :, self.t].flatten(), 
                self.waste_price[:, :, self.t].flatten()
            ))
            buyer_states.append(state_flat)
        return np.array(buyer_states)

    def get_trans_state(self, s_buyer):
        trans_states = []
        for n in range(self.num_agents):
            state_flat = s_buyer[n]
            
            q_flat = self.q[n, :, :, self.t].flatten()               
            waste_q_flat = self.waste_q[n, :, :, self.t].flatten()   
            spot_q_flat = self.spot_q[n, :, self.t].flatten()     
            state_flat = np.concatenate([state_flat, q_flat, waste_q_flat, spot_q_flat])
            
            actual_d_flat = self.actual_d[n, :, :, self.t].flatten()             
            waste_actual_d_flat = self.waste_actual_d[n, :, :, self.t].flatten() 
            inv_buy_flat = self.inv_buy[n, :, self.t].flatten()               
            waste_inv_buy_flat = self.waste_inv_buy[n, :, self.t].flatten()   
            state_flat = np.concatenate([state_flat, actual_d_flat, waste_actual_d_flat, inv_buy_flat, waste_inv_buy_flat])
            
            trans_states.append(state_flat)
        return np.array(trans_states)

    def step_sell(self, orig_leader_action):
        """
        Phase 1: Upper-Level Leader sets baseline pricing constraints (phi)
        Filters raw actions via a low-pass filter (alpha = 0.05) to maintain follower stability
        """
        raw_phi = np.clip(orig_leader_action, 0.1, 10.0)
        
        # Low-pass filter (Exponential Moving Average)
        alpha_smooth = 0.05
        if self.active_phi is None:
            self.active_phi = raw_phi
        else:
            self.active_phi = (1.0 - alpha_smooth) * self.active_phi + alpha_smooth * raw_phi
        
        for n in range(self.num_agents):
            self.price[n, :, self.t] = self.active_phi[0] * self.spot_price[:, self.t]
            self.waste_price[n, :, self.t] = 0.5 * self.price[n, :, self.t]
            
        return self.get_follower_state()

    def step_buy(self, orig_buyer_actions):
        keys = ['q', 'waste_q', 'spot_q']
        nc = (self.num_agents - 1) * self.num_commodities
        lengths = [nc, nc, self.num_commodities]
        key_len_dict = {k: v for k, v in zip(keys, lengths)}
        
        buyer_actions = {k: np.zeros((self.num_agents, length)) for k, length in key_len_dict.items()}
        for i in range(self.num_agents):
            start = 0
            for key, length in key_len_dict.items():
                buyer_actions[key][i] = orig_buyer_actions[i, start:start + length]
                start += length

        for k, arr in buyer_actions.items():
            if k == 'spot_q': continue
            new_actions = np.zeros((self.num_agents, self.num_agents, self.num_commodities))
            arr = arr.reshape(self.num_agents, self.num_agents - 1, self.num_commodities)
            for i in range(self.num_agents):
                i_list = list(range(self.num_agents))
                i_list.remove(i)
                new_actions[i, i_list] = arr[i]
            buyer_actions[k] = new_actions

        for key, value in buyer_actions.items():
            getattr(self, key)[..., self.t] = value

        actual_d = self.calc_actual_sold(self.q[:, :, :, self.t], self.inv[:, :, self.t])
        actual_dw = self.calc_actual_sold(self.waste_q[:, :, :, self.t], self.waste_inv[:, :, self.t])
        
        inv_buy = self.inv[:, :, self.t] + np.sum(actual_d, axis=1) - np.sum(actual_d, axis=0)
        waste_inv_buy = self.waste_inv[:, :, self.t] + np.sum(actual_dw, axis=1) - np.sum(actual_dw, axis=0)
        inv_buy = inv_buy + self.spot_q[:, :, self.t]

        self.actual_d[..., self.t] = actual_d
        self.waste_actual_d[..., self.t] = actual_dw
        self.inv_buy[..., self.t] = inv_buy
        self.waste_inv_buy[..., self.t] = waste_inv_buy

        buyer_rewards = []
        for n in range(self.num_agents):
            e_reshape = self.price[:, :, self.t]
            ew_reshape = self.waste_price[:, :, self.t]
            p_reshape = self.spot_price[:, self.t]
            
            reward = -np.sum(self.actual_d[n, :, :, self.t] * e_reshape)
            reward -= np.sum(self.waste_actual_d[n, :, :, self.t] * ew_reshape)
            reward -= np.sum(self.spot_q[n, :, self.t] * p_reshape)
            
            reward += self.active_phi[1] * np.sum(self.waste_actual_d[n, :, :, self.t])
            reward -= self.active_phi[2] * np.sum(self.spot_q[n, :, self.t])
            
            buyer_rewards.append(reward * self.RWD_SCALE)

        return np.array(buyer_rewards)

    def step_trans(self, orig_trans_actions):
        keys = ['tx_u', 'eco_u']
        key_len_dict = {k: self.num_commodities for k in keys}
        
        trans_actions = {k: np.zeros((self.num_agents, self.num_commodities)) for k in keys}
        for i in range(self.num_agents):
            start = 0
            for key, length in key_len_dict.items():
                trans_actions[key][i] = orig_trans_actions[i, start:start + length]
                start += length

        trans_actions['tx_u'] = np.minimum(trans_actions['tx_u'], 0.5 * self.inv_buy[..., self.t])
        trans_actions['eco_u'] = np.minimum(trans_actions['eco_u'], 0.5 * self.inv_buy[..., self.t])

        for key, value in trans_actions.items():
            getattr(self, key)[..., self.t] = value

        u_bot, w_bot = self.apply_agent_surrogate(trans_actions['tx_u'])
        
        self.inv[:, :, self.t + 1] = np.maximum(
            self.inv_buy[:, :, self.t] - trans_actions['tx_u'] - trans_actions['eco_u'] + u_bot, 0.0
        )
        self.waste_inv[:, :, self.t + 1] = (1 - self.delta) * (self.waste_inv_buy[:, :, self.t] + w_bot)

        uc_p = self.uc_p[:, self.t]
        tx_p = self.tx_p[:, self.t]
        trans_rewards = []
        for n in range(self.num_agents):
            r = np.sum(self.eco_u[n, :, self.t] * uc_p) - np.sum(self.tx_u[n, :, self.t] * tx_p)
            trans_rewards.append(r * self.RWD_SCALE)

        recycled_volume = np.sum(self.waste_actual_d[..., self.t])
        landfilled_volume = np.sum(self.spot_q[..., self.t])
        freshwater_consumption = np.sum(self.inv[:, 0, self.t])
        
        leader_reward = recycled_volume - self.active_phi[2] * landfilled_volume - 0.1 * freshwater_consumption
        leader_reward_scaled = leader_reward * self.RWD_SCALE

        self.t += 1
        done = (self.t == self.episode_length)
        
        return self.get_leader_state(), self.get_follower_state(), np.array(trans_rewards), leader_reward_scaled, done

    def calc_actual_sold(self, q, I):
        d = np.zeros_like(q)
        for c in range(self.num_commodities):
            for n in range(self.num_agents):
                buys = np.array([q[m, n, c] for m in range(self.num_agents)])
                sorted_indices = np.argsort(-buys)
                cum_sum = 0
                for i in range(self.num_agents):
                    agent_i = sorted_indices[i]
                    if i == 0:
                        d[agent_i, n, c] = min(I[n, c], buys[agent_i])
                    else:
                        available_I = I[n, c] - cum_sum
                        if available_I <= 0: break
                        d[agent_i, n, c] = min(available_I, buys[agent_i])
                    cum_sum += d[agent_i, n, c]
        return d

    def apply_agent_surrogate(self, tx_u):
        agent0 = tx_u[0]
        agent1 = tx_u[1]
        agent2 = tx_u[2]

        agents_final_output_vec = np.zeros_like(tx_u)
        agent0_surrogate_input_vec = np.array([agent0[5], agent0[7], agent0[2], agent0[0], agent0[1]])
        agent0_surrogate_output_vec = self.surrogate_model.get_apap_model_outputs(agent0_surrogate_input_vec.reshape(1,-1))
        agents_final_output_vec[0, [2, 6, 0]] = np.array(agent0_surrogate_output_vec)[0, [0, 1, 3]]
        self.wastewater[:, :, self.t] = np.array(agent0_surrogate_output_vec)[0, [3]]

        agent1_surrogate_input_vec = np.array([agent1[3], agent1[4], agent1[10], agent1[9]])
        agent1_surrogate_output_vec = self.surrogate_model.get_pap_model_outputs(agent1_surrogate_input_vec.reshape(1,-1))
        agents_final_output_vec[1, [11, 5]] = np.array(agent1_surrogate_output_vec)[0, [0, 1]]

        water_conv = agent2[0] if agent2[0] >= 19.*agent2[2] else 20.*agent2[0]/19.
        agent2_surrogate_input_vec = np.array([water_conv])
        agent2_surrogate_output_vec = self.surrogate_model.get_hyd_model_outputs(agent2_surrogate_input_vec.reshape(1,-1))
        agents_final_output_vec[2, [2, 8, 3, 0]] = np.array(agent2_surrogate_output_vec)[0, [0, 1, 3, 4]]

        return agents_final_output_vec, np.zeros_like(tx_u)

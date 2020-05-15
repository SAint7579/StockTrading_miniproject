import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler

def get_data():
    df = pd.read_csv("aapl_msi_sbux.csv")
    return df.values

def get_scaler(env):
    states = []
    for _ in range(env.n_steps):
        action=np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break
    #Object to sacle the data
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class LinearModel:
    def __init__(self,input_dim, n_action):
        self.W = np.random.randn(input_dim,n_action)/ np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        # momentum terms
        self.vW = 0
        self.vb = 0

        self.losses = []

    def predict(self,X):
        #It needs to be 2D since ML models in SKLEARN needs 2D
        assert(len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self,X,Y,learning_rate=0.01, momentum=0.9):
        assert(len(X.shape) == 2)

        num_values = np.prod(Y.shape)
        Yhat = self.predict(X)
        #Gradient
        gW = 2* X.T.dot(Yhat - Y)/ num_values
        gb = 2* (Yhat - Y).sum(axis=0)/num_values

        #Getting the momentum
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        self.W += self.vW
        self.b += self.vb

        mse = np.mean((Yhat-Y)**2)
        self.losses.append(mse)

    def load_weights(self,filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self,filepath):
        np.savez(filepath,W = self.W, b=self.b)

        
        
class MultiStockEnv:

    def __init__(self,data,initial_investment=20000):
        self.stock_price_history = data
        self.n_steps, self.n_stock = self.stock_price_history.shape

        #instance attribute
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        self.action_space = np.arange(3**self.n_stock)

        #For our action set,
        #0 = sell
        #1 = hold
        #2 = buy
        self.action_list = list(map(list,itertools.product([0,1,2],repeat=self.n_stock)))

        #calculate size of state
        self.state_dim = self.n_stock * 2 * 1

        self.reset()

    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def step(self,action):
        #Action exist in the action space
        assert action in self.action_space

        #Current protfolio value
        prev_val = self._get_val()

        #Update price to the next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]

        # perform the trade
        self._trade(action)

        #get the new values after taking the action
        cur_val = self._get_val()

        #reward in the increase in the protfolio value
        reward = cur_val - prev_val

        #done if we have run out of data
        done = self.cur_step == self.n_steps -1

        #store the current value of the portfolio here
        info = {'cur_val':cur_val}

        return self._get_obs(),reward, done, info

    #Utility fuction
    def _get_obs(self):
        #In this case, we can use observation and state interchangably
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2*self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def _trade(self,action):
        
        action_vec = self.action_list[action]

        #determine which stocks to buy or sell

        sell_index = []
        buy_index = []
        for i,a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        if sell_index:
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False


        
        
        
class DQNAgent(object):
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)

    def act(self,state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_value = self.model.predict(state)
        return np.argmax(act_value[0])

    def train(self,state,action,reward,next_state,done):
        #Doing a Q learning training with linear model for state estimation
        
        if done :
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state),axis = 1)

        target_full = self.model.predict(state)
        target_full[0,action] = target

        self.model.sgd(state,target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self,name):
        self.model.load_weights(name)
    def save(self,name):
        self.model.save_weights(name)

def play_one_episode(agent,env,is_train):
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent.train(state,action,reward, next_state,done)
        state = next_state

    return info['cur_val']

if __name__ == '__main__':
    models_folder = 'linear_rl_trader_model'
    rewards_folder = 'linear_rl_trader_rewards'

    num_episodes = 2000
    batch_size = 32
    initial_investment = 20000

    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode',type=str,required = True, help = 'either "train" or "test"')
    args = parser.parse_args()

    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    data = get_data()
    n_timesteps,n_stocks = data.shape
    n_train = n_timesteps//2

    train_data = data[:n_train]
    test_data = data[n_train:]


    env = MultiStockEnv(train_data,initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size,action_size)
    scaler = get_scaler(env)

    portfolio_value = []
    if args.mode == 'test':
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        env = MultiStockEnv(test_data,initial_investment)

        #To stop high exploration
        agent.epsilon = 0.01

        agent.load(f'{models_folder}/linear.npz')

    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent,env,args.mode)
        dt = datetime.now() - t0
        print(f"episode: {e+1}/{num_episodes}, episode and value: {val:.2f}, duration: {dt}")
        portfolio_value.append(val)

    if args.mode == "train":
        agent.save(f'{models_folder}/linear.npz')

        with open(f'{models_folder}/scaler.pkl','wb') as f:
            pickle.dump(scaler,f)
        plt.plot(agent.model.losses)
        plt.show()

    np.save(f'{rewards_folder}/{args.mode}.npy',portfolio_value)

import numpy as np
import sys
from gym import utils
from gym.envs.toy_text import discrete
import time


LEFT = 0
RIGHT = 1

class Chain(discrete.DiscreteEnv):
    def __init__(self, prob, nS):
        self.nS = nS #= 8  #6 main + 2 terminal
        
        if self.nS < 5:
            raise ValueError('A very specific bad thing happened basically nS should not be less than 5.')
            
        self.nA = nA = 2
           
        self.left_states = int(nS / 2 - 1) if nS % 2 == 0 else int((nS - 1) / 2 - 1)
        self.right_states = self.left_states - 1 if nS % 2 == 0 else self.left_states
        
        self.desc = desc = np.asarray(['L'] + self.left_states*['F'] + ['S'] + self.right_states*['F'] + ['R'], dtype='c')
        self.reward_range = (1, 10)

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def inc(s, a):
            if a == LEFT:
                s = max(s - 1, 0)
            if a == RIGHT:
                s = min(s + 1, nS - 1)

            return s

        def update_probability_matrix(s, a):
            new_s = inc(s, a)
            new_letter = desc[new_s]
            done = bytes(new_letter) in b'LR'
            reward = 10 if done else 1
            return new_s, reward, done

        for s in range(nS):
            for a in range(nA):
                li = P[s][a]
                letter = desc[s]
                if letter in b'LR':
                    li.append((1.0, s, 0, True))
                else:
                    li.append((
                        1 - prob/2.,
                        *update_probability_matrix(s, a)
                    ))

                    li.append((
                        prob/2.,
                        *update_probability_matrix(s, int(not a))
                    ))

        super(Chain, self).__init__(nS, nA, P, isd)

        
    def get_rewards(self):
        # (s, a, s')
        rewards = np.zeros((self.nS, self.nA, self.nS))

        for state in range(1, self.nS - 1):
            rewards[state, :, state - 1] = 1
            rewards[state, :, state + 1] = 1
            
        rewards[1, :, 0] = 10
        rewards[self.nS - 2, :, self.nS - 1] = 10
        
        return rewards
    
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        terminal_state_1 = self.nS - 1
        terminal_state_2 = 0

        desc = ['T'] + (self.nS - 2)*['S'] + ['T']

        desc[terminal_state_1] = utils.colorize(desc[terminal_state_1], "magenta", highlight=False)
        desc[terminal_state_2] = utils.colorize(desc[terminal_state_2], "magenta", highlight=False)
        desc[self.s] = utils.colorize(desc[self.s], "red", highlight=True)

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Right"][self.lastaction]))
        else:
            outfile.write("\n")

        outfile.write(desc[0] + "<-" + desc[1] + '<=>' + '<=>'.join(desc[2:-2]) + '<=>'  + desc[-2] + "->" + desc[-1] +"\n")
    
    
if __name__ == "__main__":
    p = 0.1
    env = Chain(p, 8)
    env.render()
    opt_actions = (env.left_states + 1)*[1] + (env.right_states + 2)*[0]

    n_episodes = 5
    max_steps = 25
    
    for episode in range(n_episodes):
        s = env.reset()
        total_reward = 0
        print('---------EPISODE {}---------'.format(episode))
        env.render()
        
        for i in range(max_steps):
            a = opt_actions[s]
            s, r, done, info = env.step(a)
            total_reward += r
            env.render()
            time.sleep(0.1)
            
        if done: 
            print(total_reward)
        else:
            print("Not enough steps.")
            
    reward_matrix = env.get_rewards()
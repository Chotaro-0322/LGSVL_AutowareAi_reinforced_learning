from .brain import Brain

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self, episode):
        self.brain.replay(episode)

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()
    
    def memorize_td_error(self, td_error):
        self.brain.td_error_memory.push(td_error)

    def update_td_error_memory(self):
        self.brain.update_td_error_memory()

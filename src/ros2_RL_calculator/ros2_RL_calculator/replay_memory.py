
class ReplayMemory():
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

            self.memory[self.index] = [state, action, state_next, reward]
            self.index = (self.index + 1) % self.capacity # 保存しているindexを1つずらす　→　1001 % self.capacity = 1となる

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

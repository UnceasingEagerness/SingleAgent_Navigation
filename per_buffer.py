import numpy as np
import torch

class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children.
    Used for O(log N) proportional sampling in Prioritized Experience Replay.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0
        self.size = 0

    def add(self, priority):
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
            
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate the change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we reach bottom, end the search
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            # Downward search, always search for a higher priority node
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
                
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) Buffer using a SumTree.
    Assigns sampling priorities proportional to TD errors.
    """
    def __init__(self, obs_dim, action_dim, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000, eps=1e-5, device="cpu"):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.eps = eps
        self.device = device
        
        self.tree = SumTree(capacity)
        
        # Data storage arrays
        self.observations = np.zeros((capacity, *obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.max_priority = 1.0

    def add(self, obs, action, reward, next_obs, done):
        """Store a new transition with maximum priority to ensure it is sampled at least once."""
        idx = self.tree.data_pointer
        
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_observations[idx] = next_obs
        self.dones[idx] = done
        
        # New transitions get max priority to guarantee they are sampled
        self.tree.add(self.max_priority)

    def sample(self, batch_size):
        """Sample a batch of transitions proportionally based on their priority."""
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_next_obs = []
        batch_dones = []
        
        idxs = []
        priorities = []
        
        segment = self.tree.total_priority / batch_size
        
        # Increment Beta towards 1.0
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            
            tree_idx, priority, data_idx = self.tree.get_leaf(v)
            
            priorities.append(priority)
            idxs.append(tree_idx)
            
            batch_obs.append(self.observations[data_idx])
            batch_actions.append(self.actions[data_idx])
            batch_rewards.append(self.rewards[data_idx])
            batch_next_obs.append(self.next_observations[data_idx])
            batch_dones.append(self.dones[data_idx])
            
        # Compute Importance Sampling Weights: w_t = (N * P_t) ^ -beta
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        weights = np.power(self.tree.size * sampling_probabilities, -self.beta)
        
        # Normalize weights by max weight for stability
        weights /= weights.max()
        
        # Convert to PyTorch tensors
        data = (
            torch.tensor(np.array(batch_obs), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(batch_actions), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(batch_rewards), dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(np.array(batch_next_obs), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(batch_dones), dtype=torch.float32, device=self.device).unsqueeze(1)
        )
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        return data, idxs, weights

    def update_priorities(self, idxs, td_errors):
        """Update the priorities of sampled transitions based on absolute TD error."""
        for idx, td_error in zip(idxs, td_errors):
            # p_t = |delta_t| + epsilon
            priority = (abs(td_error) + self.eps) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return self.tree.size

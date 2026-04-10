# Deep Q-Learning: From Basics to Brain Mapping

## A Comprehensive Guide to Understanding the Cross et al. (2021) Paper and DQN Implementation

---

## Table of Contents

1. [Reinforcement Learning Foundations](#1-reinforcement-learning-foundations)
2. [From Q-Learning to Deep Q-Networks](#2-from-q-learning-to-deep-q-networks)
3. [The DQN Architecture](#3-the-dqn-architecture)
4. [The Cross et al. Study: Connecting DQN to the Brain](#4-the-cross-et-al-study-connecting-dqn-to-the-brain)
5. [Code Implementation Walkthrough](#5-code-implementation-walkthrough)
6. [Key Experiments and Analyses](#6-key-experiments-and-analyses)
7. [Practical Exercises](#7-practical-exercises)

---

## 1. Reinforcement Learning Foundations

### 1.1 The RL Framework Refresher

You mentioned knowing the basics of states, actions, and Q-values. Let's solidify these concepts with concrete examples from the Atari games in the paper.

**Core Components:**

| Component | Symbol | Atari Example (Pong) |
|-----------|--------|---------------------|
| State | s | The current frame (what you see on screen) |
| Action | a | Move paddle up, down, or stay still |
| Reward | r | +1 when you score, -1 when opponent scores |
| Policy | π | The strategy for choosing actions |
| Value | V(s) | How good is this game situation? |
| Q-Value | Q(s,a) | How good is taking this action in this situation? |

### 1.2 The Decision Problem

The fundamental question in RL: **Given what I see (state), what should I do (action)?**

In Pong, this translates to:
- I see the ball coming toward me at a certain angle
- I see my paddle is below where the ball will arrive
- What action maximizes my chance of returning the ball?

The **Q-function** Q(s, a) answers: "If I'm in state s and take action a, what's the expected total future reward?"

### 1.3 The Bellman Equation

The magic of Q-learning comes from the Bellman equation:

```
Q(s, a) = r + γ × max[Q(s', a')]
```

In plain English:
- The value of taking action `a` in state `s` equals:
  - The immediate reward `r` I get, PLUS
  - The discounted (γ) value of the best action I can take in the next state

**γ (gamma)** is the discount factor (typically 0.99). It makes future rewards worth slightly less than immediate rewards.

---

## 2. From Q-Learning to Deep Q-Networks

### 2.1 The Problem with Traditional Q-Learning

In simple environments, we can store Q-values in a table:

```
           | Up    | Down  | Stay  |
State_1    | 0.5   | 0.2   | 0.1   |
State_2    | 0.1   | 0.8   | 0.3   |
...
```

But in Atari games, the state is an image (210×160 pixels × 3 color channels). That's **100,800 dimensions**! You can't have a table entry for every possible image.

### 2.2 The Deep Learning Solution

Instead of storing Q-values, we **approximate** them with a neural network:

```
Input: Game frame (84×84×4 grayscale)
       ↓
   [Neural Network]
       ↓
Output: Q-values for each possible action
        Q(s, up), Q(s, down), Q(s, left), Q(s, right), Q(s, fire)
```

This is the key insight of DQN: The neural network learns to **generalize** across similar states.

### 2.3 Why This Matters for the Cross et al. Paper

The paper's central hypothesis: **The brain might solve the same problem the same way.**

When you play Pong:
1. Your visual cortex processes the raw pixels
2. Higher brain regions extract relevant features (ball position, paddle position, velocity)
3. Motor areas select and execute actions

DQN does something analogous:
1. Early layers (Conv1-2) extract visual features
2. Later layers (Conv3, FC) encode game-relevant state representations
3. Output layer maps to action values

---

## 3. The DQN Architecture

### 3.1 Network Structure

The DQN used in the Cross et al. study follows the original Mnih et al. (2015) architecture:

```
Input Layer:
├── 84×84×4 grayscale images (4 stacked frames for motion)

Convolutional Layers (Feature Extraction):
├── Conv1: 32 filters, 8×8 kernel, stride 4 → ReLU
├── Conv2: 64 filters, 4×4 kernel, stride 2 → ReLU  
├── Conv3: 64 filters, 3×3 kernel, stride 1 → ReLU

Fully Connected Layers:
├── FC (Layer 4): 512 units → ReLU

Output Layer:
└── Q-values: One output per possible action
```

### 3.2 What Each Layer Does

**Conv Layer 1 (32 filters):**
- Detects basic visual features: edges, gradients, simple shapes
- Large 8×8 kernel with stride 4 reduces spatial dimensions quickly
- Similar to V1 in the brain (primary visual cortex)

**Conv Layer 2 (64 filters):**
- Combines basic features into more complex patterns
- Detects corners, textures, simple object parts

**Conv Layer 3 (64 filters):**
- Extracts game-relevant features: the ball, paddles, cars, invaders
- This is where the Cross et al. paper focuses much analysis
- The 64 "filters" can be individually analyzed

**Fully Connected Layer (512 units):**
- Integrates spatial features into a holistic state representation
- Encodes the "meaning" of the current game state

**Output Layer:**
- One Q-value per possible action
- Agent takes the action with highest Q-value

### 3.3 Why 4 Stacked Frames?

A single frame doesn't contain velocity information. By stacking 4 consecutive frames:
- The network can "see" motion
- It can determine: Is the ball moving up or down? How fast?
- This is crucial for predicting where objects will be

---

## 4. The Cross et al. Study: Connecting DQN to the Brain

### 4.1 The Core Research Questions

1. **Do DQN's state representations resemble the brain's?**
   - Can DQN hidden layer activations predict fMRI brain activity?

2. **Which brain regions encode these representations?**
   - Does a sensorimotor network emerge?

3. **What computational principles do brain and DQN share?**
   - Do they both encode high-level features?
   - Are they both invariant to irrelevant visual changes?

### 4.2 The Experimental Design

**Participants:** 6 people, 4.5 hours of gameplay each (1.5 hours per game)

**Games:**
- **Pong:** Simple, 2D, can manually label all relevant features
- **Enduro:** Driving game, weather/time changes (nuisance variables)
- **Space Invaders:** Shooting game, varying number of enemies

**Analysis Pipeline:**
```
1. Human plays game in fMRI scanner
2. Record brain activity + game frames
3. Run game frames through trained DQN
4. Extract hidden layer activations
5. Build encoding model: DQN activations → Brain activity
6. Test: Can DQN features predict voxel responses?
```

### 4.3 The Encoding Model

This is the core analysis method. Here's the intuition:

```
For each voxel in the brain:
    
    DQN_features = [Conv1_PCA, Conv2_PCA, Conv3_PCA, FC_PCA]  # 400 features
    
    Model: voxel_response = β₀ + β₁×feature_1 + β₂×feature_2 + ... + βₙ×feature_n
    
    Train this linear model
    Test: Can it predict held-out brain responses?
```

**Key findings:**
- DQN features significantly predict activity in dorsal visual stream
- This extends into posterior parietal cortex (PPC)
- Also predicts motor and premotor cortex activity

### 4.4 Why DQN Outperforms Control Models

The paper tests several control models:

| Model | What it captures | Performance |
|-------|------------------|-------------|
| Motor regressors | Just button presses | Low |
| PCA on pixels | Linear visual features | Medium |
| VAE | Nonlinear visual features (no reward signal) | Medium |
| DQN (other game) | Wrong game features | Medium |
| **DQN (correct game)** | **Task-relevant features + reward** | **Highest** |

The conclusion: The brain encodes **reward-relevant** features, not just visual features.

### 4.5 The State-Space Representation

In Pong, the paper manually labeled 6 features:
1. Ball X position
2. Ball Y position  
3. Ball X velocity
4. Ball Y velocity
5. Left paddle position
6. Right paddle position

**Key finding:** DQN layers 3 and 4 become highly correlated with these hand-drawn features, while pixel space has low correlation (ρ = 0.058).

This means DQN learns to **disentangle** relevant features from raw pixels—and the brain appears to do the same thing.

### 4.6 Nuisance Invariance

In Enduro, the sky color changes (day → night → snow → fog). This is a "nuisance variable"—it doesn't affect what actions you should take.

**Finding:** Posterior parietal cortex (PPC) representations are more invariant to these nuisances than early visual cortex.

This suggests PPC encodes an **abstract state space** stripped of irrelevant visual details.

---

## 5. Code Implementation Walkthrough

### 5.1 The Minimal DQN (keon/deep-q-learning)

Let me break down the essential components with detailed explanations.

#### 5.1.1 The DQN Agent Class

```python
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size      # Dimension of state (e.g., 4 for CartPole)
        self.action_size = action_size    # Number of possible actions
        
        # HYPERPARAMETERS
        self.memory = deque(maxlen=2000)  # Replay buffer (stores experiences)
        self.gamma = 0.95                  # Discount factor for future rewards
        self.epsilon = 1.0                 # Exploration rate (start fully random)
        self.epsilon_min = 0.01            # Minimum exploration
        self.epsilon_decay = 0.995         # How fast to reduce exploration
        self.learning_rate = 0.001         # Neural network learning rate
        
        # The neural network that approximates Q(s,a)
        self.model = self._build_model()
```

**Understanding the Parameters:**

- **`memory` (deque):** Stores (state, action, reward, next_state, done) tuples. The deque automatically removes old experiences when it exceeds 2000 entries. This is the "experience replay" buffer.

- **`gamma = 0.95`:** The discount factor. A reward 10 steps in the future is worth 0.95^10 ≈ 0.60 of its face value. This makes the agent prefer sooner rewards.

- **`epsilon`:** The exploration-exploitation tradeoff. With probability ε, take a random action (explore). Otherwise, take the best known action (exploit).

- **`epsilon_decay`:** After each episode, multiply ε by this. The agent gradually shifts from exploration to exploitation.

#### 5.1.2 Building the Neural Network

```python
def _build_model(self):
    """Build the neural network that approximates Q(s,a)"""
    model = Sequential()
    
    # Hidden layer 1: Takes state as input
    model.add(Dense(24, input_dim=self.state_size, activation='relu'))
    
    # Hidden layer 2: Further feature processing
    model.add(Dense(24, activation='relu'))
    
    # Output layer: One Q-value per action (no activation = linear output)
    model.add(Dense(self.action_size, activation='linear'))
    
    # Compile with MSE loss (we're doing regression on Q-values)
    model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    
    return model
```

**Why this architecture works:**
- Input: The current state (e.g., [cart_position, cart_velocity, pole_angle, pole_velocity])
- Hidden layers: Learn nonlinear combinations of state features
- Output: Q-value for each action. If 2 actions, output is [Q(s, left), Q(s, right)]

#### 5.1.3 Storing Experiences

```python
def memorize(self, state, action, reward, next_state, done):
    """Store an experience in the replay buffer"""
    self.memory.append((state, action, reward, next_state, done))
```

**Experience tuple:** (s, a, r, s', done)
- s: State before action
- a: Action taken
- r: Reward received
- s': State after action
- done: Did the episode end?

#### 5.1.4 Choosing Actions (ε-greedy policy)

```python
def act(self, state):
    """Choose an action using epsilon-greedy policy"""
    if np.random.rand() <= self.epsilon:
        # EXPLORE: Random action
        return random.randrange(self.action_size)
    
    # EXPLOIT: Use neural network to predict Q-values
    act_values = self.model.predict(state)
    
    # Return action with highest Q-value
    return np.argmax(act_values[0])
```

**The exploration-exploitation balance:**
- Early in training (ε ≈ 1.0): Almost always random
- Late in training (ε ≈ 0.01): Almost always use learned policy
- This ensures the agent explores enough to find good strategies

#### 5.1.5 Learning from Experience (The Core of DQN)

```python
def replay(self, batch_size):
    """Train on a batch of experiences from memory"""
    
    # Sample random batch from memory
    minibatch = random.sample(self.memory, batch_size)
    
    for state, action, reward, next_state, done in minibatch:
        
        # CALCULATE TARGET Q-VALUE
        if done:
            # If episode ended, there's no future reward
            target = reward
        else:
            # Bellman equation: Q(s,a) = r + γ × max[Q(s',a')]
            target = reward + self.gamma * np.amax(
                self.model.predict(next_state)[0]
            )
        
        # Get current Q-value predictions for all actions
        target_f = self.model.predict(state)
        
        # Update only the Q-value for the action we took
        target_f[0][action] = target
        
        # Train the network: make Q(s,a) closer to target
        self.model.fit(state, target_f, epochs=1, verbose=0)
    
    # Decay exploration rate
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
```

**This is the heart of DQN. Let's trace through an example:**

Suppose:
- state = [ball at position 5, paddle at position 3]
- action = 1 (move up)
- reward = 0
- next_state = [ball at position 4, paddle at position 4]
- done = False

Step 1: Calculate target
```python
# What's the best Q-value we can get from next_state?
best_next_q = max(Q(next_state, up), Q(next_state, down), Q(next_state, stay))

# Target for Q(state, up):
target = 0 + 0.95 * best_next_q
```

Step 2: Update the network
```python
# Current predictions: [Q(state, up)=0.2, Q(state, down)=0.5, Q(state, stay)=0.3]
# We want Q(state, up) to become closer to `target`
# Leave Q(state, down) and Q(state, stay) unchanged

target_f = [target, 0.5, 0.3]
model.fit(state, target_f)  # One gradient descent step
```

### 5.2 The Atari DQN (Cross et al. style)

For Atari games, the architecture is more complex because the state is an image.

#### 5.2.1 Preprocessing

```python
def preprocess_frame(frame):
    """Convert raw game frame to DQN input"""
    
    # 1. Convert to grayscale (210x160x3 → 210x160)
    gray = np.mean(frame, axis=2)
    
    # 2. Downscale to 84x84
    resized = cv2.resize(gray, (84, 84))
    
    # 3. Normalize to [0, 1]
    normalized = resized / 255.0
    
    return normalized

def stack_frames(frames):
    """Stack 4 consecutive frames for motion information"""
    # Shape: (84, 84, 4) - 4 channels, one per frame
    return np.stack(frames, axis=-1)
```

#### 5.2.2 Convolutional Architecture

```python
from keras.layers import Conv2D, Flatten

def build_atari_dqn(action_size):
    model = Sequential()
    
    # Conv Layer 1: 32 filters, 8x8 kernel, stride 4
    # Input: (84, 84, 4)
    # Output: (20, 20, 32) - features about edges, gradients
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu',
                     input_shape=(84, 84, 4)))
    
    # Conv Layer 2: 64 filters, 4x4 kernel, stride 2
    # Output: (9, 9, 64) - features about shapes, patterns
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    
    # Conv Layer 3: 64 filters, 3x3 kernel, stride 1
    # Output: (7, 7, 64) - features about game objects
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    
    # Flatten: (7, 7, 64) → 3136 units
    model.add(Flatten())
    
    # Fully Connected: 3136 → 512
    # This layer encodes the "state representation"
    model.add(Dense(512, activation='relu'))
    
    # Output: Q-values for each action
    model.add(Dense(action_size, activation='linear'))
    
    return model
```

### 5.3 Understanding the Code in Context of Cross et al.

The Cross et al. paper uses this DQN architecture but focuses on **extracting the hidden layer activations** rather than just using the output Q-values.

#### Extracting Hidden Representations

```python
from keras.models import Model

# Load trained DQN
dqn_model = load_model('trained_dqn_pong.h5')

# Create a new model that outputs hidden layers
layer_outputs = [layer.output for layer in dqn_model.layers[:-1]]  # All but output
feature_extractor = Model(inputs=dqn_model.input, outputs=layer_outputs)

# For each frame from human gameplay:
def extract_features(frame):
    preprocessed = preprocess_frame(frame)
    stacked = stack_frames([preprocessed] * 4)  # Simplified
    
    # Get activations from all layers
    layer_activations = feature_extractor.predict(stacked[np.newaxis, ...])
    
    # layer_activations[0] = Conv1 output (20, 20, 32)
    # layer_activations[1] = Conv2 output (9, 9, 64)
    # layer_activations[2] = Conv3 output (7, 7, 64)
    # layer_activations[3] = FC output (512,)
    
    return layer_activations
```

#### Dimensionality Reduction for Brain Mapping

```python
from sklearn.decomposition import PCA

def prepare_features_for_encoding_model(all_frames):
    """Prepare DQN features for the encoding model analysis"""
    
    all_features = []
    for frame in all_frames:
        activations = extract_features(frame)
        
        # Flatten each layer's activations
        flat_features = [act.flatten() for act in activations]
        
        all_features.append(flat_features)
    
    # Apply PCA to each layer (reduce to 100 dimensions each)
    # This matches the Cross et al. methodology
    pca_models = []
    reduced_features = []
    
    for layer_idx in range(4):
        layer_features = np.array([f[layer_idx] for f in all_features])
        
        pca = PCA(n_components=100)
        reduced = pca.fit_transform(layer_features)
        
        pca_models.append(pca)
        reduced_features.append(reduced)
    
    # Concatenate: 4 layers × 100 PCs = 400 features
    final_features = np.concatenate(reduced_features, axis=1)
    
    return final_features, pca_models
```

---

## 6. Key Experiments and Analyses

### 6.1 Behavioral Validation

Before using DQN as a brain model, the paper validates that DQN and humans have similar policies.

**Analysis:** When humans take a "move left" action, is DQN's Q(left) > Q(right)?

```python
def validate_behavioral_similarity(human_actions, dqn_q_values):
    """
    Check: When human moves left, does DQN also prefer left?
    """
    left_frames = [i for i, a in enumerate(human_actions) if a == 'left']
    right_frames = [i for i, a in enumerate(human_actions) if a == 'right']
    
    # Q-values when human chose left
    q_left_when_human_left = [dqn_q_values[i]['left'] for i in left_frames]
    q_right_when_human_left = [dqn_q_values[i]['right'] for i in left_frames]
    
    # Is Q(left) > Q(right) when human chose left?
    agreement = np.mean(np.array(q_left_when_human_left) > 
                        np.array(q_right_when_human_left))
    
    return agreement  # Should be significantly > 0.5
```

**Result:** DQN action values are significantly higher for the action humans actually chose.

### 6.2 The Encoding Model

This is the core fMRI analysis:

```python
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

def run_encoding_model(dqn_features, fmri_data, n_runs=11):
    """
    Cross-validated encoding model: Can DQN features predict voxel responses?
    
    dqn_features: (n_timepoints, 400) - 100 PCs from each of 4 layers
    fmri_data: (n_timepoints, n_voxels) - BOLD signal from brain
    """
    
    n_timepoints, n_voxels = fmri_data.shape
    timepoints_per_run = n_timepoints // n_runs
    
    predictions = np.zeros_like(fmri_data)
    
    for test_run in range(n_runs):
        # Split data
        test_start = test_run * timepoints_per_run
        test_end = (test_run + 1) * timepoints_per_run
        
        train_idx = list(range(0, test_start)) + list(range(test_end, n_timepoints))
        test_idx = list(range(test_start, test_end))
        
        X_train = dqn_features[train_idx]
        X_test = dqn_features[test_idx]
        
        # For each voxel, train a ridge regression model
        for voxel in range(n_voxels):
            y_train = fmri_data[train_idx, voxel]
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            
            predictions[test_idx, voxel] = model.predict(X_test)
    
    # Calculate prediction accuracy for each voxel
    accuracies = []
    for voxel in range(n_voxels):
        r, _ = pearsonr(fmri_data[:, voxel], predictions[:, voxel])
        accuracies.append(r)
    
    return np.array(accuracies)
```

### 6.3 Representational Similarity Analysis (RSA)

RSA compares the "shape" of representations across different systems:

```python
def compute_dsm(features):
    """
    Compute Dissimilarity Matrix (DSM)
    
    For each pair of timepoints, how different are the representations?
    """
    from scipy.spatial.distance import pdist, squareform
    
    # Pairwise correlation distances
    distances = pdist(features, metric='correlation')
    
    # Convert to square matrix
    dsm = squareform(distances)
    
    return dsm

def compare_representations(dqn_features, hand_drawn_features, brain_features):
    """
    Compare representational geometry across systems
    """
    from scipy.stats import spearmanr
    
    # Compute DSMs
    dqn_dsm = compute_dsm(dqn_features)
    hdf_dsm = compute_dsm(hand_drawn_features)
    brain_dsm = compute_dsm(brain_features)
    
    # Extract upper triangle (avoid diagonal and symmetry)
    def get_upper_tri(dsm):
        return dsm[np.triu_indices_from(dsm, k=1)]
    
    # Compare DSMs
    dqn_vs_hdf, _ = spearmanr(get_upper_tri(dqn_dsm), get_upper_tri(hdf_dsm))
    brain_vs_hdf, _ = spearmanr(get_upper_tri(brain_dsm), get_upper_tri(hdf_dsm))
    brain_vs_dqn, _ = spearmanr(get_upper_tri(brain_dsm), get_upper_tri(dqn_dsm))
    
    return {
        'DQN_vs_HandDrawn': dqn_vs_hdf,
        'Brain_vs_HandDrawn': brain_vs_hdf,
        'Brain_vs_DQN': brain_vs_dqn
    }
```

### 6.4 Filter Analysis

The paper analyzes which of the 64 Conv3 filters are most "brain-like":

```python
def analyze_filters(dqn_model, fmri_data, frames):
    """
    Which filters in Conv3 best predict brain activity?
    """
    
    # Get Conv3 layer
    conv3_layer = dqn_model.layers[2]  # Index depends on architecture
    
    # Create feature extractor for just Conv3
    conv3_extractor = Model(inputs=dqn_model.input, 
                            outputs=conv3_layer.output)
    
    # Conv3 output shape: (7, 7, 64) - 64 filters
    n_filters = 64
    filter_predictivity = np.zeros((n_filters, fmri_data.shape[1]))
    
    for filter_idx in range(n_filters):
        # Extract activations for just this filter
        filter_activations = []
        for frame in frames:
            act = conv3_extractor.predict(frame[np.newaxis, ...])
            # act shape: (1, 7, 7, 64)
            # Get this filter and flatten: (7, 7) → 49 features
            filter_act = act[0, :, :, filter_idx].flatten()
            filter_activations.append(filter_act)
        
        filter_features = np.array(filter_activations)
        
        # Run encoding model with just this filter's features
        accuracies = run_encoding_model(filter_features, fmri_data)
        filter_predictivity[filter_idx] = accuracies
    
    # Neural Predictivity: average accuracy across voxels for each filter
    neural_predictivity = np.mean(filter_predictivity, axis=1)
    
    return neural_predictivity
```

**Key Finding:** Filters that predict brain activity well also predict human behavior well.

---

## 7. Practical Exercises

### Exercise 1: Understand Q-Value Updates

```python
"""
Manual Q-value update exercise.

Given:
- Current state s: Ball at (5, 3), Paddle at (5)
- Action taken: Move up
- Reward: 0
- Next state s': Ball at (5, 2), Paddle at (6)
- gamma = 0.95

Current Q-values:
- Q(s, up) = 0.2
- Q(s, down) = 0.5
- Q(s, stay) = 0.3

Next state Q-values:
- Q(s', up) = 0.4
- Q(s', down) = 0.1
- Q(s', stay) = 0.2

Question: What is the target value for Q(s, up)?
"""

# Your answer:
gamma = 0.95
reward = 0
max_next_q = max(0.4, 0.1, 0.2)  # = 0.4

target = reward + gamma * max_next_q
print(f"Target Q(s, up) = {target}")  # Should be 0.38
```

### Exercise 2: Implement Experience Replay

```python
"""
Why is experience replay important?

Without replay: Learning from consecutive experiences causes:
1. High correlation between samples
2. Catastrophic forgetting (new experiences overwrite old)

With replay: Learning from random samples:
1. Breaks correlations
2. Revisits old experiences

Implement a simple replay buffer:
"""

from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store an experience"""
        # YOUR CODE HERE
        pass
    
    def sample(self, batch_size):
        """Sample a random batch of experiences"""
        # YOUR CODE HERE
        pass
    
    def __len__(self):
        return len(self.buffer)

# Test your implementation
buffer = ReplayBuffer(capacity=100)
for i in range(50):
    buffer.push(
        state=np.random.randn(4),
        action=np.random.randint(2),
        reward=np.random.randn(),
        next_state=np.random.randn(4),
        done=np.random.rand() > 0.9
    )

batch = buffer.sample(10)
print(f"Sampled {len(batch)} experiences")
```

### Exercise 3: Analyze Your Own Model

```python
"""
After training a DQN, analyze what the hidden layers encode:

1. Run gameplay frames through the network
2. Extract hidden layer activations
3. Correlate with hand-coded features (if available)
4. Visualize using t-SNE or PCA
"""

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_state_space(hidden_features, labels=None):
    """
    Visualize the learned state space using t-SNE
    
    hidden_features: (n_samples, n_features) from a hidden layer
    labels: (n_samples,) optional labels for coloring (e.g., reward, action)
    """
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedded = tsne.fit_transform(hidden_features)
    
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(embedded[:, 0], embedded[:, 1], 
                            c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter)
    else:
        plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5)
    
    plt.title('State Space Visualization (t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

# Example usage:
# hidden_features = extract_conv3_features(all_frames)
# rewards = [get_reward(frame) for frame in all_frames]
# visualize_state_space(hidden_features, labels=rewards)
```

---

## Summary: Key Takeaways

### Conceptual Understanding

1. **DQN solves the "state representation problem"** by learning to map high-dimensional inputs (pixels) to a compact representation useful for decision-making.

2. **The brain appears to use similar computational principles:**
   - Nonlinear feature extraction
   - Reward-relevant state encoding
   - Nuisance invariance in higher regions

3. **The hidden layers are the interesting part** — they encode the learned state-space representation, not the output Q-values.

### Implementation Understanding

1. **Experience replay** breaks correlations in training data and enables learning from rare events.

2. **The target Q-value** comes from the Bellman equation: r + γ × max Q(s', a')

3. **Epsilon-greedy exploration** balances exploration (random actions) and exploitation (learned policy).

4. **Convolutional layers** extract hierarchically more abstract features, similar to the visual cortex.

### Connection to Neuroscience

1. **Encoding models** test whether computational features can predict brain activity.

2. **RSA** compares the "shape" of representations across different systems.

3. **Filter analysis** identifies which learned features are most brain-like.

4. **The dorsal visual stream and PPC** appear to encode task-relevant, nuisance-invariant state representations.

---

## Further Reading

1. **Original DQN Paper:** Mnih et al. (2015). "Human-level control through deep reinforcement learning." Nature.

2. **This Paper:** Cross et al. (2021). "Using deep reinforcement learning to reveal how the brain encodes abstract state-space representations in high-dimensional environments." Neuron.

3. **DeepMind Blog:** https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning

4. **Code Repository:** https://github.com/locross93/Atari-Project

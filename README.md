# Policy Gradient Mouse Agent 

## **Quick note** ##

Please just assess the discrete implementation because it works and therefore I tested/commented that one thoroughly and not the other. I'm including it cause it's what I worked on for longer. Thanks!

## Problem 

This policy gradient implementation addresses a very serious problem: Helping a mouse find a cookie in a grid world, while avoiding salad and eating cheese.

Specifically, I implement the REINFORCE algorithm to learn a policy to map the grid world's states to the mouse's actions. The policy is encoded in a pytorch neural network.

## Discrete Case

In the discrete case, the agent's state is 2-dimensional, where the first dimension corresponds to the row of state and the second corresponds to the column of the state in the grid world. 

$$ s = \begin{bmatrix} row, & col \end{bmatrix}$$

The action space is 4-dimensional, and consists of the directions the agent can move in to get to other states in the grid. 

$$ actions = \begin{bmatrix} up, & down, & left, & right \end{bmatrix}$$

If the action the policy selects were to put the agent state out of bounds, the action is not applied and the agent remains where it is. 

The grid world is a 4x4 matrix, where each element in the matrix encodes a state type. The state types are:

* Cheese: The reward for being in a cheese state is 100
* Salad: The reward/penalty for being in a salad state is -20,000
* Cookie: The reward for being in a cookie state is 10,000
* Empty: Empty states have a reward of 0

The current world set up is hard coded and looks like this: 

$$ world = \begin{bmatrix}
  1 & 0 & 0 & 2 \\
  2 & 0 & 1 & 0 \\
 3  & 0 & 0 & 0 \\
  0 & 1 & 0 & 0 
\end{bmatrix}$$

The encoding is:

* 0 = "Empty"
* 1 = "Cheese"
* 2 = "Salad"
* 3 = "Cookie"

In the program, the mapping is a dictionary of encoding integers to strings representing the state type.

The policy is encoded in a neural network, but is rendered as a matrix where the placement in the matrix corresponds to a position in the grid world and each element in the matrix describes the
action the policy suggests for the agent to take, given that the agent is in that state. 

## Continuous Case

In the continuous case, the mouse's state is represented as a vector where the first two components encode position and the second two encode velocity. 

$$ s = \begin{bmatrix} x, & y, & \dot{x}, & \dot{y} \end{bmatrix}$$

The actions are continuous and are sampled from a continuous multivariate distribution (which the neural network produces). 

The actions are two dimensional, as opposed to 1D (discrete case), representing the horizonal component of acceleration and the vertical component of acceleration.

$$ a = \begin{bmatrix} \ddot{x}, \ddot{y} \end{bmatrix}$$

The state types are not encoded directly, rather the agent determines what state it's in by figuring out its' proximity to every other "important" state in the environment. If the agent
is within a thresholded euclidean distance of an important state, than it shares the state type of that important state. 

## Theory

The REINCFORCE algorithm involves the following steps: 
### Collect an episode 
An episode is collected by:
* Resetting the agent state to a known start position
* Running a forward pass through the policy network to get a discrete probability distribution over actions, out.
* Sampling an action from the distribution
* Updating the state by applying the selected action
* Retreiving the reward associated with the agent's new state
* Logging the state, action, reward, distribution and done (which indicates whether the step was a terminating step or not) as a tuple corresponding to that step in the episode
All of the step tuples are appended to a list called episode, which is returned by the collect_episode method of the Agent class.

Once an episode is collected..
### For each step in the episode
* Compute the log probability of selecting the selected action from the learned probability distribution
* Compute the discounted sum of rewards for that step in the episode (the sum of rewards from that step in the episode to the end, in a Monte-Carlo like fashion.
* Summing the log probabilities over the episode, multiplied by the discounted sums of rewards, and negating it to make it a loss.
* Compute the gradient of the log probabilities * the discounted sums, which will point the policy in the direction of convergence. 

In mathematical terms, the heart of the policy gradient algorithm is this: 

Initialize the policies parameters randomly (done implicitly when you initialize a neural network in pytorch)

Collect an episode: 
$$ep = \{(s_1, a_1, r_1), (s_2, a_2, r_2), \ldots, (s_T, a_T, r_T)\}$$

Calculate the returns (discounted sum of rewards): 

$$ G_t = \sum_{i=t}^T \gamma^{i-t} r_i$$

Calculate the log probabilities: 
$$\nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

Repeat until the convergence. 

My implementation's loss function negates and sums the log probabilities and returns to compute a loss: 

$$ Loss = \sum-\nabla_\theta \log \pi_\theta(a_t \mid s_t)G_t$$

## Architecture

### Discrete Case

My neural network architecture consists of a single layer that maps the input (2D) to a probability distribution over actions (4D)

It uses a softmax activation function to produce the distribution. 

### Continuous case

Rather than outputting a discrete probability distribution over actions, because the actions are continuous, I used a Tontinuous multivariate distribution and sampled a 2D action from that.

To output the distribution from a neural network, I had 3 linear layers that were shared for the first part (to relate the mean to the covariance matrix defining the distribution). Then, the data gets passed through two separate final layers, one that outputs a 2D mean of the distribution and 1 that outputs a 4x4 covariance matrix. 

## Results 

### Discrete

The policy converged on something meaningful and actually encoded actions that were desirable. 

Here is an illustration of the results: 

<img src="https://github.com/jes-bro/policy_gradient_mouse/blob/main/works.png"
     alt="Working policy"
     style="float: left; margin-right: 10px;" />

### Continuous

That wasn't the case for the continuous policy :(. Not sure if it's a bug or a symptom of training or of the ode integration, there were a lot of potential points of failure. I'm including it here because I did a lot of work on it and I still learned a lot. 

I didn't comment or doc string the continuous case because it didn't work and so I consider it as more of a relic and less of a final deliverable. So I consider the disrcete case to be my success and therefore the thing to be assessed for quality etc. 

## Reflection

I learned a lot from doing this project, especially about the differences between learning continuous actions and discrete actions. It's helped me understand what's going on in certain robot learning repos I've been looking into for research as well. I'm a little sad that I didn't get the continuous one working, but it's partially a symptom of my procrastination. I'm happy I got a working version of a Monte Carlo policy gradient algorithm implemented though. I feel like I've come a long way from the first time I tried to learn this stuff in my RL independent study last semester. 

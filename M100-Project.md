# The Dining Problem

_Will Kaufman, Ivy Yan_

May 28, 2021

# The problem

Here's the simplest case: each night a student decides where to eat dinner. She has a choice of $n$ dining venues and is indifferent between all of them. However, she is busy and wants to minimize the time she waits for food. If the waiting time $t_W^{(i)}$ at venue $i$ follows a Poisson distribution with unknown parameter $\lambda_i$
$$
t_W^{(i)} \sim \text{Pois}(\lambda_i)
$$
what strategy would minimize the student's waiting time for meals?

This formulation is exactly a multi-armed bandit problem.

# Multi-armed bandits

@thompson1933likelihood, @thompson1934, @robbins1952some

Describe multi-armed bandits, cite stuff

# Greedy policy

# The Dining Problem

_Will Kaufman, Ivy Yan_

May 28, 2021

# The problem

Here's the simplest case: each night a student decides where to eat dinner. She has a choice of $n$ dining venues and is indifferent between all of them. However, she is busy and wants to minimize the time she waits for food. If the waiting time $t_W^{(i)}$ at venue $i$ follows a Poisson distribution with unknown parameter $\lambda_i$
$$
t_W^{(i)} \sim \text{Pois}(\lambda_i)
$$
what strategy would minimize the student's waiting time for meals?

This formulation is exactly a multi-armed bandit problem.

# Multi-armed bandits

Describe multi-armed bandits, cite stuff

# Greedy policy


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
lam = np.array([4, 5, 6])
```


```python
waiting_times = np.random.poisson(lam=lam, size=(70*3, lam.size))
```


```python
plt.plot(waiting_times[:, 0], label='Foco')
plt.plot(waiting_times[:, 1], label='Hop')
plt.plot(waiting_times[:, 2], label='Collis')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f7eec1256d0>




    
![png](output_11_1.png)
    



```python
random_picks = np.random.randint(3, size=(14,))
```


```python
random_picks = np.eye(3, dtype=int)[random_picks]
```


```python
random_picks
```




    array([[0, 1, 0],
           [1, 0, 0],
           [0, 0, 1],
           [0, 1, 0],
           [0, 0, 1],
           [1, 0, 0],
           [1, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [0, 1, 0],
           [0, 0, 1],
           [1, 0, 0],
           [0, 1, 0],
           [0, 1, 0]])




```python
waiting_times[random_picks].shape
```




    (14, 3, 3)




```python
random_picks
```




    array([[0, 1, 0],
           [1, 0, 0],
           [0, 0, 1],
           [0, 1, 0],
           [0, 0, 1],
           [1, 0, 0],
           [1, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [0, 1, 0],
           [0, 0, 1],
           [1, 0, 0],
           [0, 1, 0],
           [0, 1, 0]])




```python
waiting_times[:14, :]
```




    array([[ 4,  2,  9],
           [ 4, 13,  5],
           [ 3,  6, 10],
           [ 9,  5,  2],
           [ 3,  7,  6],
           [ 1,  4,  8],
           [ 5,  9,  3],
           [ 3,  7,  2],
           [ 0,  6,  8],
           [ 5,  7,  5],
           [ 6,  1,  9],
           [ 4,  4,  5],
           [ 6,  3,  4],
           [ 3,  3,  5]])




```python
(waiting_times[:14, :] * random_picks).mean(axis=0)
```




    array([1.21428571, 1.85714286, 1.78571429])



# $\epsilon$-greedy policy

# Upper confidence bound (UCB) policy

# What these look like for the dining problem

- Descriptions, feasibility
- Simulations

# Relaxing assumptions

- Non-stationary distributions (different terms -> different people on campus -> different wait times)
- 


```python

```

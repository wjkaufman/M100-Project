---
title: The Dining Problem

bibliography: references.bib

theme: white
transition: none
slideNumber: true
progress: false
width: 1500
height: 1000

---

## The Dining Problem

<style>
.reveal h1, .reveal h2, .reveal h3, .reveal h4, .reveal h5 {
  text-transform: none;
}
</style>

Math 100, Spring 2021

_Will Kaufman, Ivy Yan_

May 28, 2021


<!-- ```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
plt.rcParams['figure.dpi'] = 150  # 72
``` -->

## The problem

(Assuming normal non-COVID times)

- A student decides where to eat dinner
- Choice of 3 dining venues (Foco, Collis, Hop), indifferent between all of them
- Wants to minimize the time $T_W^{(i)}$ she waits for food

$$
T_W^{(i)} \sim \text{Pois}(\lambda_i)
$$

What strategy would minimize the student's wait time for meals?

This formulation is exactly a multi-armed bandit problem.

## Multi-armed bandits

![](graphics/slot-machine.jpg){width=30%}

- Introduced by @thompson1933likelihood, @thompson1934, @robbins1952some as "sequential design of experiments."
- Must allocate limited resources to maximize expected gain
- Balance exploration, exploitation

Common examples of multi-armed bandits

- Research funding for clinical trials
- College course selection
- Dining problem!

<!-- # Multi-armed bandit formulation

TODO fill this in -->

## Q functions

The expected reward given an action $a$ is selected

$$
q_*(a) = \mathbb{E}[R_t | A_t = a]
$$

The expected values aren't known, so we come up with an estimate $Q_t(a)$.

For the dining problem, $R_t = -T_W^{(i)}$ (want to maximize rewards, minimize wait time). So want to maximize $Q$ to minimize expected wait time.

## Problem setup


```python
lambdas = np.array([4, 5, 6])
wait_times = np.random.poisson(lam=lambdas, size=(1000, 70*3, lambdas.size))
```


<!-- ```python
plt.plot(wait_times[0, :, 0], marker='o', linestyle='none', alpha=.5, label='Foco')
plt.plot(wait_times[0, :, 1], marker='o', linestyle='none', alpha=.5, label='Hop')
plt.plot(wait_times[0, :, 2], marker='o', linestyle='none', alpha=.5, label='Collis')
plt.legend()
plt.title('Wait times (simulation 0)')
plt.xlabel('Day')
plt.ylabel('Wait time (minutes)')

plt.show()
``` -->


    
![](output_10_0.png)
    


## 1: Same old same old...

What if the student always goes to the venue with the lowest average wait time so far?

Also called "greedy policy" [@pearl1984heuristics]

1. Picks action $a$ with maximized $Q(a)$
    - Which venue the student has experienced the shortest average wait time so far
    $$
    A \leftarrow {\arg \max}_a Q(a)
    $$
2. Increment visit count for chosen $a$ (venue), update estimated $Q(a)$
    $$
    N(A) \leftarrow N(A) + 1, Q(A) \leftarrow Q(A)  + \frac{1}{N(A)} [R - Q(A)]
    $$



<!-- ```python
chosen_wait_times = np.zeros((wait_times.shape[0], 70*3))
extra_time = np.zeros((wait_times.shape[0], 70*3))
```


```python
for _ in range(wait_times.shape[0]):
    N = np.zeros((3,))
    Q_estimate = np.zeros((3,))
    actions = np.zeros((wait_times.shape[1],), dtype=int)
    for i in range(wait_times.shape[1]):
        a = Q_estimate.argmin()
        actions[i] = a
        w = wait_times[_, i, a]
        chosen_wait_times[_, i] = w
        N[a] += 1
        Q_estimate[a] = Q_estimate[a] + 1/N[a]*(w - Q_estimate[a])
    extra_time[_, :] = chosen_wait_times[_, :] - wait_times[_, :, lambdas.argmin()]
```


```python
plt.plot(extra_time.cumsum(axis=1).T, color='k', alpha=.1)
plt.xlabel('Day')
plt.ylabel('Cumulative extra time waited vs. optimal policy')
plt.show()
``` -->


## Greedy policy compared to optimal policy

    
![](output_14_0.png)
    

## Greedy policy compared to optimal policy

<!-- ```python
plt.plot(extra_time.cumsum(axis=1).mean(axis=0), color='k')
plt.xlabel('Day')
plt.ylabel('Cumulative extra time waited vs. optimal policy')
plt.title('Mean cumulative time waited for 100 simulations')
plt.show()
``` -->


    
![](output_15_0.png)
    




<!-- ```python
extra_time.sum(axis=1).mean()
``` -->


53.632





## 2: O-week exploration, then same old...

What if our student tries out different venues for the first two weeks, then settles on whichever had the lowest empirical wait time?

More commonly called "$\epsilon$-first strategy"

1. For first $\epsilon n$ decisions, pick random actions
    - Increment visit count for chosen $a$ (venue), update estimated $Q(a)$
    $$
    N(A) \leftarrow N(A) + 1, Q(A) \leftarrow Q(A)  + \frac{1}{N(A)} [R - Q(A)]
    $$
2. For last $(1-\epsilon)n$ decisions, pick action $a$ with maximized $Q(a)$
    $$
    A \leftarrow {\arg \max}_a Q(a)
    $$



<!-- ```python
chosen_wait_times = np.zeros((wait_times.shape[0], 70*3))
extra_time = np.zeros((wait_times.shape[0], 70*3))
```


```python
for _ in range(wait_times.shape[0]):
    random_picks = np.random.randint(3, size=(14,))
    random_picks = np.eye(3, dtype=int)[random_picks]
    chosen_wait_times_0 = wait_times[_, :14, :] * random_picks
    Q_estimate = chosen_wait_times_0.sum(axis=0) / random_picks.sum(axis=0)
    remaining_wait_times = wait_times[_, 14:, Q_estimate.argmin()]
    chosen_wait_times[_, :] = np.concatenate(
        [chosen_wait_times_0.sum(axis=1),
         remaining_wait_times],
        axis=0
    )
    extra_time[_, :] = chosen_wait_times[_, :] - wait_times[_, :, lambdas.argmin()]
```

    <ipython-input-35-0ee9f1893331>:5: RuntimeWarning: invalid value encountered in true_divide
      Q_estimate = chosen_wait_times_0.sum(axis=0) / random_picks.sum(axis=0)


-->

## Epsilon-first policy compared to optimal policy


<!--
```python
plt.plot(extra_time.cumsum(axis=1).T, color='k', alpha=.1)
plt.xlabel('Day')
plt.ylabel('Cumulative extra time waited vs. optimal policy')
plt.show()
``` -->


    
![](output_20_0.png)
    


## Epsilon-first policy compared to optimal policy



<!-- ```python
y = extra_time.cumsum(axis=1).mean(axis=0)
plt.plot(y, color='k')
plt.vlines(x=14, ymin=0, ymax=y.max(), color='k', linestyle='dashed')
plt.xlabel('Day')
plt.ylabel('Cumulative extra time waited vs. optimal policy')
plt.title('Mean cumulative time waited for 100 simulations')
plt.show()
``` -->


    
![](output_21_0.png)
    


80.757


## 3: Occasionally spontaneous

What if the student acts greedy _most_ of the time, but occasionally goes crazy and does something different?

Called "$\epsilon$-greedy policy"

1. With probability $1-\epsilon$, picks action $a$ with maximized $Q(a)$
    $$
    A \leftarrow {\arg \max}_a Q(a)
    $$
    
    With probability $\epsilon$, picks random action $a$
    
2. Increment visit count for chosen $a$ (venue), update estimated $Q(a)$
    $$
    N(A) \leftarrow N(A) + 1, Q(A) \leftarrow Q(A)  + \frac{1}{N(A)} [R - Q(A)]
    $$

Spontaneity is a good thing!!!


<!-- ```python
epsilon_array = np.array([.01, .05, .2, .5])
```


```python
chosen_wait_times = np.zeros((epsilon_array.shape[0], wait_times.shape[0], 70*3))
extra_time = np.zeros((epsilon_array.shape[0], wait_times.shape[0], 70*3))
```


```python
for e_ind, epsilon in enumerate(epsilon_array):
    for _ in range(wait_times.shape[0]):
        N = np.zeros((3,))
        Q_estimate = np.zeros((3,))
        actions = np.zeros((wait_times.shape[1],), dtype=int)
        for i in range(wait_times.shape[1]):
            if np.random.rand() > epsilon:
                a = Q_estimate.argmin()
            else:
                a = np.random.choice(Q_estimate.size)
            actions[i] = a
            w = wait_times[_, i, a]
            chosen_wait_times[e_ind, _, i] = w
            N[a] += 1
            Q_estimate[a] = Q_estimate[a] + 1/N[a]*(w - Q_estimate[a])
        extra_time[e_ind, _, :] = chosen_wait_times[e_ind, _, :] - wait_times[_, :, lambdas.argmin()]
``` -->


## $\epsilon$-greedy policy compared to optimal policy


<!-- ```python
plt.plot(extra_time[0, ...].cumsum(axis=1).T, color='k', alpha=.1)
plt.xlabel('Day')
plt.ylabel('Cumulative extra time waited vs. optimal policy')
plt.title(fr'$\epsilon$={epsilon_array[0]}')
plt.show()
``` -->


    
![](output_27_0.png)
    


## $\epsilon$-greedy policy compared to optimal policy


<!-- ```python
for e_ind, epsilon in enumerate(epsilon_array):
    plt.plot(extra_time[e_ind, ...].cumsum(axis=1).mean(axis=0), label=fr'$\epsilon$={epsilon}')
plt.xlabel('Day')
plt.ylabel('Cumulative extra time waited vs. optimal policy')
plt.title('Mean cumulative time waited for 100 simulations')
plt.legend()
plt.show()
``` -->


    
![](output_28_0.png)
    

50.367,  44.834,  62.245, 113.129

<!-- ```python
extra_time.sum(axis=2).mean(axis=1)
``` -->





## 4: The time-crunched math major

The student took decision theory, so she understands she needs to balance maximizing expected value while reducing uncertainty in those estimates. One way she can do this is by maximizing the "upper confidence bound (UCB) score" [@auer2002finite]

$$
A_t \dot{=} {\arg \max}_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]
$$

- $Q_t(a)$ is estimate for expected value
- $\sqrt{\frac{\ln t}{N_t(a)}}$ quantifies uncertainty of estimate for action $a$
    - Choosing action $a$ will increase $N(a)$, decrease uncertainty
    - $\ln t$ de-prioritizes exploration over time
- $c$ controls the amount of exploration


<!-- ```python
c_array = np.array([.5, 1, 2, 4, 8])
```


```python
chosen_wait_times = np.zeros((c_array.shape[0], wait_times.shape[0], 70*3))
extra_time = np.zeros((c_array.shape[0], wait_times.shape[0], 70*3))
```


```python
for c_ind, c in enumerate(c_array):
    for _ in range(wait_times.shape[0]):
        N = np.zeros((3,))
        Q_estimate = np.zeros((3,))
        actions = np.zeros((wait_times.shape[1],), dtype=int)
        for i in range(wait_times.shape[1]):
            ucb_score = -Q_estimate + c * np.sqrt(np.log(i + 1) / (N + 1e-100))
            a = ucb_score.argmax()
            actions[i] = a
            w = wait_times[_, i, a]
            chosen_wait_times[c_ind, _, i] = w
            N[a] += 1
            Q_estimate[a] = Q_estimate[a] + 1/N[a]*(w - Q_estimate[a])
        extra_time[c_ind, _, :] = chosen_wait_times[c_ind, _, :] - wait_times[_, :, lambdas.argmin()]
``` -->


## UCB policy compared to optimal policy


<!-- ```python
plt.plot(extra_time[0, ...].cumsum(axis=1).T, color='k', alpha=.1)
plt.xlabel('Day')
plt.ylabel('Cumulative extra time waited vs. optimal policy')
plt.title(f'c={c_array[0]}')
plt.show()
``` -->


    
![](output_34_0.png)
    


## UCB policy compared to optimal policy



<!-- ```python
for c_ind, c in enumerate(c_array):
    plt.plot(extra_time[c_ind, ...].cumsum(axis=1).mean(axis=0), label=f'c={c}')
plt.xlabel('Day')
plt.ylabel('Cumulative extra time waited vs. optimal policy')
plt.title('Mean cumulative time waited for 100 simulations')
plt.legend()
plt.show()
``` -->


    
![](output_35_0.png)
    


46.164,  36.506,  30.16 ,  56.849, 106.442

<!-- ```python
extra_time.sum(axis=2).mean(axis=1)
``` -->




## Comparison of different policies

- UCB does best by balancing exploration and exploitation
    - More effort than most people would put in to deciding where to eat
- Occasional spontaneity ($\epsilon$-greedy) does pretty well, and it's simple!
    - Go where you normally go most of the time, but be open to trying something new once in a while

## Other considerations

- Non-stationary distributions: wait times might be different each term
- Contextual bandits: weekdays vs. weekends
- Utility function: food preferences, social time, distance, price, ...
    - Opportunity cost of all the time spent optimizing dining venue!

## References

Slot machine picture from @slot_machine

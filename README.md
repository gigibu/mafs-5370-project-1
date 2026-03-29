# MAFS 5370 Project 1

### Student name: Angeline Candice

### SID: 20634390

Consider the discrete-time asset allocation example in Section 8.4 of Rao and Jelvis.

Let the price vector \( X \) have dimension \( n \), where \( n > 2 \), and:

\[
X(0) = 1
\]

The one-period return of asset \( k \) follows a normal distribution:

\[
R_k \sim \mathcal{N}(a(k), s(k))
\]

where:

- \( a(k) \) is the mean return of asset \( k \)
- \( s(k) \) is the variance of asset \( k \)

You are given an initial portfolio allocation:

- Asset \( k \) has portion \( p(k) \)
- \( p(0) \) is the portion allocated to cash

Portfolio constraint:

\[
p(0) + p(1) + \dots + p(n) = 1
\]

Cash earns a fixed interest rate \( r \).

At each period:

- You may adjust **at most 10%** of the total portfolio allocation.

The investor has an **absolute risk-averse utility function**.

## Task

Use **Reinforcement Learning (RL)** to compute the optimal strategy at each period for any reasonable choice of:

- \( r \)
- \( a(k) \)
- \( s(k) \)
- \( p(k) \)

## Requirements

Your program must:

- Work for any time horizon \( T < 10 \)
- Work for any number of assets \( n < 5 \)
- Compute the optimal strategy at each period
- Respect the 10% reallocation constraint
- Incorporate absolute risk aversion in the reward/utility formulation

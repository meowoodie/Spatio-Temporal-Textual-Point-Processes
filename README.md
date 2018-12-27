Event Correlation Detection
===

Introduction
---
Consider events represented by *spatial-temporal text*, a data tuple consists of time, location, and text. And we model the sequence of *spatial-temporal text* events using a multivariate Hawkes point process, called s*patial-temporal text* point process. *Spatial-temporal text* point process is essentially a marked multivariate Hawkes process, where each component is discretized location, and text is mark. By using an adapted kernel function, as well as text embedding techniques, our proposed spatial-temporal text point process is able to incorporate the text similarity as part of the influence between events. The intensity function of the point process is shown below.
<img src=https://github.com/meowoodie/Event-Correlation-Detection/blob/master/imgs/intensity-function.png width="70%">

With the conditional intensity in hand, we explicitly denote the dependence of the likelihood function on the spatial-temporal coefficients in the presence of *spatial-temporal text*. The log likelihood function is shown as below.
<img src=https://github.com/meowoodie/Event-Correlation-Detection/blob/master/imgs/log-likelihood.png width="50%">

We then construct the linkage between crime events by introducing auxiliary variables that indicates the probability *i*-th event is linked to *j*-th event. Moreover, an EM algorithm for learning the parameters is presented. 
<img src=https://github.com/meowoodie/Event-Correlation-Detection/blob/master/imgs/e-step.png width="50%">
<img src=https://github.com/meowoodie/Event-Correlation-Detection/blob/master/imgs/m-step.png width="50%">

Usage
---
Below is an simple example for initialization and fitting of the model.
```python
# init MPPEM object
# - t: a sequence of time
# - u: a sequence of locations (indices)
# - l: a sequence of labels (optional)
# - l: a sequence of marks
# - d: dimension of components of the point process (number of discretized locations)
mppem = MPPEM(seq_t=t, seq_u=u, seq_l=l, seq_m=m, d=len(u_set))
# init A
distance_matrix = utils.calculate_beats_pairwise_distance(u_set, csv_filename)
mppem.init_A(distance_matrix, gamma=gamma)
# init Mu
mppem.init_Mu(gamma=gamma)
# fit model
# - ps:  a list of precisions over iterations
# - rs:  a list of recalls over iterations
# - lls: a list of loglikelihoods over iterations
# - lbs: a list of lower bounds over iterations
# - t, T:  start and end of the time window for fitting
# - iters: number of iterations
ps, rs, lls, lbs = em.fit(T=t[-1], tau=t[0], iters=iters)
# results
print(em.P)
print(em.A)
```

References
---

- [Preprint: Crime Linkage Detection by Spatial-Temporal Text Point Processes. S. Zhu and Y. Xie.]()

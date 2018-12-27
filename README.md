Event Correlation Detection
===

Introduction
---
Consider events represented by *spatial-temporal text*, a data tuple consists of time, location, and text. And we model the sequence of *spatial-temporal text* events using a multivariate Hawkes point process, called s*patial-temporal text* point process. *Spatial-temporal text* point process is essentially a marked multivariate Hawkes process, where each component is discretized location, and text is mark. By using an adapted kernel function, as well as text embedding techniques, our proposed spatial-temporal text point process is able to incorporate the text similarity as part of the influence between events. The intensity function of the point process is shown below.
![intensity-function](https://github.com/meowoodie/Event-Correlation-Detection/blob/master/imgs/intensity-function.png)

With the conditional intensity in hand, we explicitly denote the dependence of the likelihood function on the spatial-temporal coefficients in the presence of *spatial-temporal text*. The log likelihood function is shown as below.
![log-likelihood](https://github.com/meowoodie/Event-Correlation-Detection/blob/master/imgs/log-likelihood.png)

We then construct the linkage between crime events by introducing auxiliary variables that indicates the probability *i*-th event is linked to *j*-th event. Moreover, an EM algorithm for learning the parameters is presented. 
![e-step](https://github.com/meowoodie/Event-Correlation-Detection/blob/master/imgs/e-step.png)
![m-step](https://github.com/meowoodie/Event-Correlation-Detection/blob/master/imgs/m-step.png)

Usage
---



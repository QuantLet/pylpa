# pyLPA
This is a python repository implementing the Local Parametric Approach from Spokoiny, V. (2009a).

# Config

- `min_steps`: int, Mininum distance between two break point test
- `max_trial`: int, Maximum attempts to estimate MLE
- `maxiter`: int, Maximum number of iterations to perform to estimate MLE
- `num_sim`: int, Number of bootstrap simulation
- `njobs`: int, Number of parallel bootstrap estimation
- `generate`: str, "exponential" or "normal" distribution of the multiplier 
  bootstrap weights

*References*

*Spokoiny, V. (1998). Estimation of a function with discontinuities via 
local polynomial fit with an adaptive  window choice, The Annals of 
Statistics 26: 1356–78.*

*Spokoiny, V. (2009a). Multiscale local change point detection with 
applications to value-at-risk, The  Annals of Statistics 37: 1405–1436.*

*Spokoiny, V. and Zhilova, M. (2015). Bootstrap confidence sets under model 
misspecification, The Annals  of Statistics 43(1): 2653–2675*

*Spilak, Bruno and Härdle, Wolfgang Karl, Tail-Risk Protection: Machine 
Learning Meets Modern Econometrics (October 7, 2020). Spilak, B., Härdle, W.
K. (2021). In: Lee, CF., Lee, A.C. (eds) Encyclopedia of Finance. Springer, 
Cham. https://doi.org/10.1007/978-3-030-73443-5_94-1*
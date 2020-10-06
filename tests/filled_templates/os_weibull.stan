functions{
    real orderstatistics(int N, int M, vector q, vector U){
        real lpdf = 0;
        lpdf += lgamma(N+1) - lgamma(N*q[1]) - lgamma(N-N*q[M]+1);
        lpdf += (N*q[1]-1)*log(U[1]);
        lpdf += (N-N*q[M])*log(1-U[M]);
        for (m in 2:M){
            lpdf += -lgamma(N*q[m]-N*q[m-1]);
            lpdf += (N*q[m]-N*q[m-1]-1)*log(U[m]-U[m-1]);
        }
        return lpdf;
    }
}
data{
    int N;
    int M;
    vector[M] q;
    vector[M] X;
}
parameters{
    real<lower=0> alpha;
    real<lower=0> sigma;
}
transformed parameters{
    vector[M] U;
    for (m in 1:M)
        U[m] = weibull_cdf(X[m], alpha, sigma);
}
model{
    alpha ~ gamma(1.0, 1.2);
    sigma ~ weibull(2.1, 2.2);
    target += orderstatistics(N, M, q, U);
    for (m in 1:M)
        target += weibull_lpdf(X[m] | alpha, sigma);
}
generated quantities {
    real predictive_dist = weibull_rng(alpha, sigma);
    real log_prob = orderstatistics(N, M, q, U);
    for (m in 1:M)
        log_prob += weibull_lpdf(X[m] | alpha, sigma);
}

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
    real mu;
    real<lower=0> sigma;
}
transformed parameters{
    vector[M] U;
    for (m in 1:M)
        U[m] = lognormal_cdf(X[m], mu, sigma);
}
model{
    mu ~ normal(1.0, 1.2);
    sigma ~ lognormal(2.1, 2.2);
    target += orderstatistics(N, M, q, U);
    for (m in 1:M)
        target += lognormal_lpdf(X[m] | mu, sigma);
}
generated quantities {
    real predictive_dist = lognormal_rng(mu, sigma);
    real log_prob = orderstatistics(N, M, q, U);
    for (m in 1:M)
        log_prob += lognormal_lpdf(X[m] | mu, sigma);
}

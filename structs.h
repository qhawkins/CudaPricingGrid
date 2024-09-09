struct GridParams {
    float S0;      // Initial stock price
    float K;       // Strike price
    float T;       // Time to expiration
    float r;       // Risk-free rate
    float sigma;   // Volatility
    int M;         // Number of time steps
    int N;         // Number of stock price steps
    float dt;      // Time step size
    float dS;      // Stock price step size
    float Smax;    // Maximum stock price in the grid
};

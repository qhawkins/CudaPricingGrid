#ifndef STRUCTS_H
#define STRUCTS_H

#include <string>

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
    float contractType; // 0 for call, 1 for put
    float optionPrice; // Price of the option
};

struct OptionData {
    float market_price;
    float strike_price;
    float underlying_price;
    float years_to_expiration;
    float rfr;
    std::string option_type;
    std::string contract_id;
    std::string timestamp;
    // First-order Greeks
    float delta;
    float vega;
    float theta;
    float rho;
    float lambda;
    float epsilon;

    // Second-order Greeks
    float gamma;
    float vanna;
    float charm;
    float vomma;
    float veta;
    float vera;

    // Third-order Greeks
    float speed;
    float zomma;
    float color;
    float ultima;

    // Additional values
    float impliedVolatility;
    float modelPrice;
};

#endif // STRUCTS_H
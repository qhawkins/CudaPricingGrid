#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

enum class OptionType { Call, Put };

class AmericanOptionPricer {
private:
    double S, K, r, q, sigma, T;
    int steps;
    OptionType optionType;

public:
    AmericanOptionPricer(double S, double K, double r, double q, double sigma, double T, int steps, OptionType type)
        : S(S), K(K), r(r), q(q), sigma(sigma), T(T), steps(steps), optionType(type) {}

    double price() {
        double dt = T / steps;
        double u = std::exp(sigma * std::sqrt(dt));
        double d = 1 / u;
        double p = (std::exp((r - q) * dt) - d) / (u - d);

        std::vector<double> prices(steps + 1);
        std::vector<double> values(steps + 1);

        // Initialize asset prices at expiration
        for (int i = 0; i <= steps; ++i) {
            prices[i] = S * std::pow(u, steps - i) * std::pow(d, i);
            if (optionType == OptionType::Call) {
                values[i] = std::max(0.0, prices[i] - K);
            } else {
                values[i] = std::max(0.0, K - prices[i]);
            }
        }

        // Backward induction
        for (int step = steps - 1; step >= 0; --step) {
            for (int i = 0; i <= step; ++i) {
                double spotPrice = S * std::pow(u, step - i) * std::pow(d, i);
                double expectedValue = (p * values[i] + (1 - p) * values[i + 1]) * std::exp(-r * dt);
                
                if (optionType == OptionType::Call) {
                    values[i] = std::max(expectedValue, spotPrice - K);
                } else {
                    values[i] = std::max(expectedValue, K - spotPrice);
                }
            }
        }

        return values[0];
    }
};

int main() {
    double S = 100;  // Current stock price
    double K = 100;  // Strike price
    double r = 0.05;  // Risk-free rate
    double q = 0;     // Dividend yield
    double sigma = 0.2;  // Volatility
    double T = 1.0;   // Time to maturity in years
    int steps = 10000;  // Number of steps in the binomial tree

    AmericanOptionPricer putPricer(S, K, r, q, sigma, T, steps, OptionType::Put);
    double putPrice = putPricer.price();

    AmericanOptionPricer callPricer(S, K, r, q, sigma, T, steps, OptionType::Call);
    double callPrice = callPricer.price();

    std::cout << "The price of the American put option is: " << putPrice << std::endl;
    std::cout << "The price of the American call option is: " << callPrice << std::endl;

    return 0;
}
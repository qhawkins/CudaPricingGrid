#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

enum class OptionType { Call, Put };

class AmericanOptionPricer {
private:
    double S, K, r, q, T, tol;
    int steps, max_iter;
    OptionType optionType;

public:
    AmericanOptionPricer(double S, double K, double r, double q, double T, int steps, OptionType type, double tol, int max_iter)
        : S(S), K(K), r(r), q(q), T(T), steps(steps), optionType(type), tol(tol), max_iter(max_iter) {}

    double price(double sigma) {
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

    double computeIV(double optionPrice) {
        auto f = [this, optionPrice](double sigma) {
            return price(sigma) - optionPrice;
        };
        // Initial guess
        double a = 0.01;
        double b = 10.0;
        double fa = f(a);
        double fb = f(b);
        std::cout << "Initial fa: " << fa << std::endl;
        std::cout << "Initial fb: " << fb << std::endl;
        // If not bracketed, expand the interval
        int bracket_attempts = 0;
        while (fa * fb > 0 && bracket_attempts < 50) {
            if (std::abs(fa) < std::abs(fb)) {
                a -= (b - a);
                fa = f(a);
            } else {
                b += (b - a);
                fb = f(b);
            }
            bracket_attempts++;
        }

        if (fa * fb > 0) {
            return -1;  // Root not bracketed after attempts
        }

        double c = b, fc = fb;
        double d, e;

        for (int iter = 0; iter < max_iter; iter++) {
            if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
                c = a; fc = fa;
                d = b - a; e = d;
            }
            if (std::abs(fc) < std::abs(fb)) {
                a = b; b = c; c = a;
                fa = fb; fb = fc; fc = fa;
            }

            double tol1 = 2 * std::numeric_limits<double>::epsilon() * std::abs(b) + 0.5 * tol;
            double xm = 0.5 * (c - b);
            
            if (std::abs(xm) <= tol1 || fb == 0) {
                return b;  // Found a solution
            }
            
            if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
                double s = fb / fa;
                double p, q;
                if (a == c) {
                    p = 2 * xm * s;
                    q = 1 - s;
                } else {
                    q = fa / fc;
                    double r = fb / fc;
                    p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1));
                    q = (q - 1) * (r - 1) * (s - 1);
                }
                if (p > 0) q = -q;
                p = std::abs(p);
                
                if (2 * p < std::min(3 * xm * q - std::abs(tol1 * q), std::abs(e * q))) {
                    e = d;
                    d = p / q;
                } else {
                    d = xm;
                    e = d;
                }
            } else {
                d = xm;
                e = d;
            }

            a = b;
            fa = fb;
            b += (std::abs(d) > tol1) ? d : (xm > 0 ? tol1 : -tol1);
            fb = f(b);
        }

        return -2;  // Max iterations reached
    }
};

int main() {
    double S = 100;  // Current stock price
    double K = 100;  // Strike price
    double mP = 6.76; // Market price
    double r = 0.05;  // Risk-free rate
    double q = 0;     // Dividend yield
    double sigma = 0.2;  // Volatility
    double T = 1.0;   // Time to maturity in years
    int steps = 1000;  // Number of steps in the binomial tree

    AmericanOptionPricer putPricer(S, K, r, q, T, steps, OptionType::Put, .00001, 1000);
    double callIV = putPricer.computeIV(mP);

    AmericanOptionPricer callPricer(S, K, r, q, T, steps, OptionType::Call, .00001, 1000);
    double putIV = callPricer.computeIV(mP);

    std::cout << "The IV of the American put option is: " << callIV << std::endl;
    std::cout << "The IV of the American call option is: " << putIV << std::endl;

    return 0;
}
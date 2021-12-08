#ifndef CUTILS_CPP
#define CUTILS_CPP

#include "cutils.h"
#include <math.h>

namespace mabCutils {
    double catonialpha(const double &v, const int &itercount, const int &_size) {
        double lg4t = 4.0 * std::log(double(itercount));
        return std::sqrt(lg4t / (double(_size) * (v + v * lg4t / (double(_size) - lg4t))));
    }

    double psi(const double &x) {
        if (x < 0) return -psi(-x);
        if (x > 1) return std::log(2 * x - 1) / 4 + 5.0 / 6;
        return x - x * x * x / 6;
    }

    double dpsi(const double &x) {
        if (x < 0) return dpsi(-x);
        if (x > 1) return 1.0 / (4 * x - 2);
        return 1.0 - x * x / 2;
    }

    double sumpsi(const double &v, const int &itercount, const double &guess, mabnodescpp &nodes) {
        double ans = 0.0;
        auto a_d = catonialpha(v, itercount, nodes.size());
        for (auto p = nodes.gethead(); p; p = p->next) {
            ans += psi(a_d * (p->val - guess));
        }
        return ans;
    }

    double dsumpsi(const double &v, const int &itercount, const double &guess, mabnodescpp &nodes) {
        double ans = 0.0;
        auto a_d = catonialpha(v, itercount, nodes.size());
        for (auto p = nodes.gethead(); p; p = p->next) ans += dpsi(a_d * (p->val - guess));
        return -a_d * ans;
    }

    double nt_iter(const double &v, const int &itercount, const double &guess, mabnodescpp &nodes, const double &fguess) {
        return guess - fguess / dsumpsi(v, itercount, guess, nodes);
    }

    std::pair<double, int> getcatoni(const double &v, const int &itercount, double &guess, mabnodescpp &nodes, const double &tol) {
        auto a = sumpsi(v, itercount, guess, nodes);
        int nt_itercount = 0;
        auto a_d = catonialpha(v, itercount, nodes.size());
        auto realtol = tol * a_d * a_d;
        while ((a > realtol || a < -realtol) && nt_itercount < 50) {
            guess = nt_iter(v, itercount, guess, nodes, a);
            a = sumpsi(v, itercount, guess, nodes);
            ++nt_itercount;
        }
        return {guess, nt_itercount};
    }
} // namespace mabCutils

#endif // CUTILS_CPP
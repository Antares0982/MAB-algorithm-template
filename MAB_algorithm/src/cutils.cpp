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

    double sumpsi(const double &v, const int &itercount, const double &guess, mabarraycpp &arr) {
        double ans = 0.0;
        auto a_d = catonialpha(v, itercount, arr.size());
        for (int i = 0; i < arr.size(); ++i) ans += psi(a_d * (arr[i] - guess));
        return ans;
    }

    double dsumpsi(const double &v, const int &itercount, const double &guess, mabarraycpp &arr) {
        double ans = 0.0;
        auto a_d = catonialpha(v, itercount, arr.size());
        for (int i = 0; i < arr.size(); ++i) ans += dpsi(a_d * (arr[i] - guess));
        return -a_d * ans;
    }

    double nt_iter(const double &v, const int &itercount, const double &guess, mabarraycpp &arr, const double &fguess) {
        return guess - fguess / dsumpsi(v, itercount, guess, arr);
    }

    std::pair<double, int> getcatoni(const double &v, const int &itercount, double &guess, mabarraycpp &arr, const double &tol) {
        auto a = sumpsi(v, itercount, guess, arr);
        int nt_itercount = 0;
        auto a_d = catonialpha(v, itercount, arr.size());
        auto realtol = tol * a_d * a_d;
        while ((a > realtol || a < -realtol) && nt_itercount < 50) {
            guess = nt_iter(v, itercount, guess, arr, a);
            a = sumpsi(v, itercount, guess, arr);
            ++nt_itercount;
        }
        return {guess, nt_itercount};
    }

    // distns
    double heavytail_pdf(const double &alpha, const double &beta, const double &coef, const double &maxMomentOrder, double x) {
        x = alpha * x + beta;
        if (x < 2) return 0.0;
        double log_sq = std::log(x);
        log_sq *= log_sq;
        return alpha * coef / (std::pow(x, maxMomentOrder + 1) * log_sq);
    }
} // namespace mabCutils

#endif // CUTILS_CPP

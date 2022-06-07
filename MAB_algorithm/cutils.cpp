#include "cutils.h"
#include <algorithm>
#include <iostream>
#include <vector>

namespace mabCutils {
    // function declaration
    double findmedian(std::vector<double> &vec);

    // class member function definition
    double mabarraycpp::momentMedianOfMean(const double theta, const double q, const int k) const {
        int binsizeN = _len / k;

        if (binsizeN < 1) throw std::out_of_range("Invalid bin number " + std::to_string(k));
        int total = binsizeN * k;

        std::vector<double> means;
        means.reserve(k);

        for (auto p = begin(); p != begin() + total; ++p) {
            means[(p - begin()) / binsizeN] += std::pow(std::abs(*p - theta), q);
        }

        return findmedian(means) / binsizeN;
    }

    std::pair<medianOfMeanArrayCpp::leftQueue, medianOfMeanArrayCpp::rightQueue> &medianOfMeanArrayCpp::updateMedianMeanArray(int k, int binsizeN) const {
        auto &pr = avgmemory[binsizeN];
        int binCount = pr.first.size() + pr.second.size();

        while (binCount < k) {
            double topush = (presum_unique_ptr[(binCount + 1) * binsizeN] - presum_unique_ptr[binCount * binsizeN]) / binsizeN;

            if (pr.first.empty() || pr.first.top() > topush)
                pr.first.push(topush);
            else
                pr.second.push(topush);

            if (pr.first.size() > pr.second.size() + 1) {
                pr.second.push(pr.first.top());
                pr.first.pop();
            } else if (pr.first.size() + 1 < pr.second.size()) {
                pr.first.push(pr.second.top());
                pr.second.pop();
            }

            binCount++;
        }

        return pr;
    }

    double medianOfMeanArrayCpp::medianMeanWithMoment(int bins, double guess_avg, double p) const {
        int binsize = size() / bins;
        std::vector<double> medians(bins, 0.);
        auto ptr = &arr_unique_ptr[0];
        for (int i = 0; i < bins; ++i) {
            auto &median = medians[i];
            for (int j = 0; j < binsize; ++j) {
                median += std::pow(std::abs(*ptr - guess_avg), p);
                ++ptr;
            }
            median /= binsize;
        }
        return findmedian(medians);
    }
    // begin function def

    inline double catonialpha(const double v, const int itercount, const int _size) {
        double lg4t = 4.0 * std::log(double(itercount));
        return std::sqrt(lg4t / (double(_size) * (v + v * lg4t / (double(_size) - lg4t))));
    }

    inline double psi(const double x) {
        if (x < 0) return -psi(-x);
        if (x > 1) return std::log(2 * x - 1) / 4 + 5.0 / 6;
        return x - x * x * x / 6;
    }

    inline double dpsi(const double x) {
        if (x < 0) return dpsi(-x);
        if (x > 1) return 1.0 / (4 * x - 2);
        return 1.0 - x * x / 2;
    }

    inline double sumpsi(const double v, const int itercount, const double guess, mabarraycpp &arr) {
        double ans = 0.0;
        auto a_d = catonialpha(v, itercount, arr.size());
        for (int i = 0; i < arr.size(); ++i) ans += psi(a_d * (arr[i] - guess));
        return ans;
    }

    inline double dsumpsi(const double v, const int itercount, const double guess, mabarraycpp &arr) {
        double ans = 0.0;
        auto a_d = catonialpha(v, itercount, arr.size());
        for (int i = 0; i < arr.size(); ++i) ans += dpsi(a_d * (arr[i] - guess));
        return -a_d * ans;
    }

    inline double nt_iter(const double v, const int itercount, const double guess, mabarraycpp &arr, const double fguess) {
        return guess - fguess / dsumpsi(v, itercount, guess, arr);
    }

    // find the median of a given vector
    double findmedian(std::vector<double> &vec) {
        int mind = vec.size() / 2;
        std::nth_element(vec.begin(), vec.begin() + mind, vec.end());
        if (vec.size() & 1) return vec[mind];
        auto it = std::max_element(vec.begin(), vec.begin() + mind);
        return ((*it) + vec[mind]) / 2;
    }

    // implement interfaces
    // mean sestimator
    std::pair<double, int> _calculateCatoniMean(const double v, const int itercount, double guess, mabarraycpp &arr, const double tol) {
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

    double _calculateTruncatedMean(const double u, const double ve, const int itercount, mabarraycpp &arr) {
        const double ee = u / (2 * std::log(itercount));
        const double vinv = 1 / (ve + 1);

        double ans = 0.0;
        for (int i = 0; i < arr.size();) {
            double v = arr[i];
            if (std::abs(v) <= std::pow(ee * (++i), vinv)) ans += v;
        }
        return ans / arr.size();
    }

    double _calculateMedianMean(mabarraycpp &arr, const int bins) {
        int N = arr.size() / bins;
        std::vector<double> tmp(bins, 0.0);

        for (int i = 0; i < arr.size(); ++i) {
            int b = i / N;
            if (b == bins) break;
            double v = arr[i];
            tmp[b] += v;
        }

        return findmedian(tmp) / N;
    }

    // distns utils
    double heavytail_pdf(const double alpha, const double beta, const double coef, const double maxMomentOrder, double x) {
        x = alpha * x + beta;
        if (x < 2) return 0.0;
        double log_sq = std::log(x);
        log_sq *= log_sq;
        return alpha * coef / (std::pow(x, maxMomentOrder + 1) * log_sq);
    }
} // namespace mabCutils

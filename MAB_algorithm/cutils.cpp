#ifndef CUTILS_CPP
#define CUTILS_CPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mabCutils {
    class mabarraycpp {
    protected:
        int _cap;
        int _len;
        std::unique_ptr<double[]> arr_unique_ptr;

    public:
        mabarraycpp() : _cap(0), _len(0) {}

        virtual ~mabarraycpp() = default;

    public:
        virtual void startup(int cap) {
            arr_unique_ptr.reset(new double[cap]);
            _len = 0;
            _cap = cap;
        }

        double &operator[](int index) {
            return __get_index__(index);
        }

        const double &operator[](int index) const {
            return __get_index__(index);
        }

        virtual void append(double v) {
            if (_len >= _cap) {
                throw std::out_of_range("size is equal to capacity, cannot append new element.");
            }
            arr_unique_ptr[_len] = v;
            ++_len;
        }

        int size() const { return _len; }

        int cap() const { return _cap; }

        double *begin() { return &arr_unique_ptr[0]; }

        const double *begin() const { return &arr_unique_ptr[0]; }

        double *end() { return &arr_unique_ptr[0] + _len; }

        const double *end() const { return &arr_unique_ptr[0] + _len; }

        double avg() const {
            double ans = 0.0;
            for (int i = 0; i < _len; ++i) ans += arr_unique_ptr[i];
            return ans / _len;
        }

    private:
        double &__get_index__(int index) const {
            if (index < _len) return arr_unique_ptr[index];
            throw std::out_of_range("invalid index, expected less than " + std::to_string(_len));
        }
    };

    class medianOfMeanArrayCpp : public mabarraycpp {
    private: // typedefs
        using leftQueue = std::priority_queue<double>;
        using rightQueue = std::priority_queue<double, std::vector<double>, std::greater<double>>;
        using memory_iterator = std::unordered_map<int, std::pair<leftQueue, rightQueue>>::iterator;

    private:
        std::unique_ptr<double[]> presum_unique_ptr;
        // run-time evaluate mutable object `avgmemory`
        mutable std::unordered_map<int, std::pair<leftQueue, rightQueue>> avgmemory;

    public:
        medianOfMeanArrayCpp() : mabarraycpp() {}

        void startup(int cap) override {
            mabarraycpp::startup(cap);
            presum_unique_ptr.reset(new double[cap + 1]);
        }

        void append(double v) override {
            if (_len >= _cap) {
                throw std::out_of_range("size is equal to capacity, cannot append new element.");
            }
            arr_unique_ptr[_len] = v;
            ++_len;
            presum_unique_ptr[_len] = presum_unique_ptr[_len - 1] + v;
        }

        double getMedianMean(int binsizeN) const {
            auto &pr = updateMedianMeanArray(binsizeN);
            if ((pr.first.size() + pr.second.size()) & 1) return (pr.first.size() > pr.second.size()) ? pr.first.top() : pr.second.top();
            return (pr.first.top() + pr.second.top()) / 2.0;
        }

    private:
        std::pair<leftQueue, rightQueue> &updateMedianMeanArray(int binsizeN) const;
    };

    // mean estimator

    // Returns: catoni mean, number of times iterated
    // std::pair<double, int> getcatoni(const double, const int, double, mabarraycpp &, const double);

    // double truncatedMean(const double, const double, const int, mabarraycpp &);

    // double medianMean(const double, const double, const int, mabarraycpp &);

    // distns utils
    // double heavytail_pdf(const double, const double, const double, const double, double);

    // begin function def

    double catonialpha(const double v, const int itercount, const int _size) {
        double lg4t = 4.0 * std::log(double(itercount));
        return std::sqrt(lg4t / (double(_size) * (v + v * lg4t / (double(_size) - lg4t))));
    }

    double psi(const double x) {
        if (x < 0) return -psi(-x);
        if (x > 1) return std::log(2 * x - 1) / 4 + 5.0 / 6;
        return x - x * x * x / 6;
    }

    double dpsi(const double x) {
        if (x < 0) return dpsi(-x);
        if (x > 1) return 1.0 / (4 * x - 2);
        return 1.0 - x * x / 2;
    }

    double sumpsi(const double v, const int itercount, const double guess, mabarraycpp &arr) {
        double ans = 0.0;
        auto a_d = catonialpha(v, itercount, arr.size());
        for (int i = 0; i < arr.size(); ++i) ans += psi(a_d * (arr[i] - guess));
        return ans;
    }

    double dsumpsi(const double v, const int itercount, const double guess, mabarraycpp &arr) {
        double ans = 0.0;
        auto a_d = catonialpha(v, itercount, arr.size());
        for (int i = 0; i < arr.size(); ++i) ans += dpsi(a_d * (arr[i] - guess));
        return -a_d * ans;
    }

    double nt_iter(const double v, const int itercount, const double guess, mabarraycpp &arr, const double fguess) {
        return guess - fguess / dsumpsi(v, itercount, guess, arr);
    }

    double findmedian(std::vector<double> &vec) {
        int mind = vec.size() / 2;
        std::nth_element(vec.begin(), vec.begin() + mind, vec.end());
        if (vec.size() & 1) return vec[mind];
        auto it = std::max_element(vec.begin(), vec.begin() + mind);
        return ((*it) + vec[mind]) / 2;
    }

    // mean sestimator
    std::pair<double, int> getcatoni(const double v, const int itercount, double guess, mabarraycpp &arr, const double tol) {
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

    double truncatedMean(const double u, const double ve, const int itercount, mabarraycpp &arr) {
        double ee = u / (2 * std::log(itercount));
        double vinv = 1 / (ve + 1);
        auto bd = [&](const double &x) {
            return std::pow(ee * x, vinv);
        };
        double ans = 0.0;
        for (int i = 0; i < arr.size();) {
            double v = arr[i];
            if (std::abs(v) <= bd(++i)) ans += v;
        }
        return ans / arr.size();
    }

    double medianMean(const double v, const double ve, const int itercount, mabarraycpp &arr) {
        double ee = v / (2 * std::log(itercount));
        double vinv = 1 / (ve + 1);
        auto bd = [&](const int &x) {
            return std::pow(ee * x, vinv);
        };
        int k = int(std::floor(std::min(1 + 16 * std::log(itercount), double(arr.size()) / 2)));
        if (k < 1) k = 1;
        int N = int(std::floor(double(arr.size()) / k));
        std::vector<double> tmp(k, 0.0);

        for (int i = 0; i < arr.size(); ++i) {
            int b = i / N;
            if (b == k) break;
            double v = arr[i];
            if (std::abs(v) <= bd((i % N) + 1)) {
                if (b >= k) throw std::out_of_range("Out of range");
                tmp[b] += v;
            }
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

    std::pair<medianOfMeanArrayCpp::leftQueue, medianOfMeanArrayCpp::rightQueue> &medianOfMeanArrayCpp::updateMedianMeanArray(int binsizeN) const {
        auto &pr = avgmemory[binsizeN];
        int binCount = pr.first.size() + pr.second.size();

        int maxBinCount = _len / binsizeN;

        while (binCount < maxBinCount) {
            double topush = presum_unique_ptr[(binCount + 1) * binsizeN] - presum_unique_ptr[binCount * binsizeN];

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
} // namespace mabCutils

#endif // CUTILS_CPP

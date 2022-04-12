#ifndef CUTILS_H
#define CUTILS_H


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

        void operator=(mabarraycpp &) = delete;

        virtual ~mabarraycpp() = default;

    public:
        virtual void startup(int cap) {
            arr_unique_ptr.reset(new double[cap]);
            _len = 0;
            _cap = cap;
        }

        double &operator[](int index) const {
            if (index < _len) return arr_unique_ptr[index];
            throw std::out_of_range("invalid index, expected less than " + std::to_string(_len));
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

        double *end() { return &arr_unique_ptr[0] + _len; }

        double avg() const {
            double ans = 0.0;
            for (int i = 0; i < _len; ++i) ans += arr_unique_ptr[i];
            return ans / _len;
        }
    };

    class medianOfMeanArrayCpp : public mabarraycpp {
    private:
        using leftQueue = std::priority_queue<double>;
        using rightQueue = std::priority_queue<double, std::vector<double>, std::greater<double>>;
        std::unordered_map<int, std::pair<leftQueue, rightQueue>> avgmemory;
        std::unique_ptr<double[]> presum_unique_ptr;
        double *presum;
        double medianMean;

    public:
        medianOfMeanArrayCpp() : mabarraycpp(), presum(nullptr) {}

        void startup(int cap) override {
            mabarraycpp::startup(cap);
            presum_unique_ptr.reset(new double[cap + 1]);
            presum = &presum_unique_ptr[0];
            *presum = 0;
        }

        void append(double v) override {
            if (_len >= _cap) {
                throw std::out_of_range("size is equal to capacity, cannot append new element.");
            }
            arr_unique_ptr[_len] = v;
            ++_len;
            presum[_len] = presum[_len - 1] + v;
        }

        double getMedianMean(int binsizeN) {
            // TODO
            updateMedianMeanArray(binsizeN);
            auto &&pr = avgmemory[binsizeN];
            if ((pr.first.size() + pr.second.size()) & 1) return pr.first.size() > pr.second.size() ? pr.first.top() : pr.second.top();
            return (pr.first.top() + pr.second.top()) / 2.0;
        }

    private:
        void updateMedianMeanArray(int binsizeN);
    };

    // mean estimator

    // Returns: catoni mean, number of times iterated
    std::pair<double, int> getcatoni(const double, const int, double, mabarraycpp &, const double);

    double truncatedMean(const double, const double, const int, mabarraycpp &);

    double medianMean(const double, const double, const int, mabarraycpp &);

    // distns utils
    double heavytail_pdf(const double, const double, const double, const double, double);
} // namespace mabCutils

#endif // CUTILS_H

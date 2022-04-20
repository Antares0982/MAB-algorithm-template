#ifndef CUTILS_H
#define CUTILS_H


#include <cmath>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>

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
            throw std::out_of_range("invalid index, expected less than " + std::to_string(_len) + ", got " + std::to_string(index));
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

        double getMedianMean(int iteration) const {
            int k = std::max(1, int(std::floor(std::min(1 + 16 * std::log(iteration), double(_len) / 2))));
            int binsizeN = _len / k;
            auto &pr = updateMedianMeanArray(k, binsizeN);
            if ((pr.first.size() + pr.second.size()) & 1) return (pr.first.size() > pr.second.size()) ? pr.first.top() : pr.second.top();
            return (pr.first.top() + pr.second.top()) / 2.0;
        }

    private:
        std::pair<leftQueue, rightQueue> &updateMedianMeanArray(int k, int binsizeN) const;
    };

    // mean estimator

    // Returns: catoni mean, number of times iterated
    std::pair<double, int> getcatoni(const double, const int, double, mabarraycpp &, const double);

    double truncatedMean(const double, const double, const int, mabarraycpp &);

    double medianMean(const int, mabarraycpp &);

    // distns utils
    double heavytail_pdf(const double, const double, const double, const double, double);

} // namespace mabCutils
#endif // CUTILS_H
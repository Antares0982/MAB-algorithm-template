#ifndef CUTILS_H
#define CUTILS_H

#include <stdexcept>
#include <utility>

namespace mabCutils {
    class mabarraycpp {
    private:
        int _cap;
        int _len;
        double *arr;

    public:
        mabarraycpp() : _cap(0), _len(0), arr(nullptr) {}

        void startup(const int &cap) {
            delete[] arr;
            _len = 0;
            _cap = cap;
            arr = new double[cap];
        }

        double &operator[](const int &index) const {
            if (index < _len) return arr[index];
            throw std::out_of_range("invalid index, expected less than " + std::to_string(_len));
        }

        void append(double v) {
            arr[_len] = v;
            ++_len;
        }

        int size() const { return _len; }

        int cap() const { return _cap; }

        double *begin() { return arr; }

        double *end() { return &arr[_len]; }

        double avg() const {
            double ans = 0.0;
            for (int i = 0; i < _len; ++i) ans += arr[i];
            return ans / _len;
        }

        ~mabarraycpp() {
            delete[] arr;
        }
    };

    // mean estimator

    // Returns: catoni mean, number of times iterated
    std::pair<double, int> getcatoni(const double &, const int &, double &, mabarraycpp &, const double &);

    double truncatedMean(const double &, const double &, const int &, mabarraycpp &);

    double meadianMean(const double &, const double &, const int &, mabarraycpp &);

    // distns utils
    double heavytail_pdf(const double &, const double &, const double &, const double &, double);
} // namespace mabCutils

#endif // CUTILS_H

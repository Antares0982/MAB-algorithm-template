#ifndef CUTILS_H
#define CUTILS_H

// #include <iostream>
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

        double avg() const {
            double ans = 0.0;
            for (int i = 0; i < _len; ++i) ans += arr[i];
            return ans / _len;
        }

        ~mabarraycpp() {
            delete[] arr;
        }
    };

    double catonialpha(const double &v, const int &itercount, const int &_size);

    double psi(const double &x);

    double dpsi(const double &x);

    double sumpsi(const double &v, const int &itercount, const double &guess, mabarraycpp &arr);

    double dsumpsi(const double &v, const int &itercount, const double &guess, mabarraycpp &arr);

    double nt_iter(const double &v, const int &itercount, const double &guess, mabarraycpp &arr, const double &fguess);

    std::pair<double, int> getcatoni(const double &v, const int &itercount, double &guess, mabarraycpp &arr, const double &tol);

    // distns utils
    double heavytail_pdf(const double &alpha, const double &beta, const double &coef, const double &maxMomentOrder, double x);
} // namespace mabCutils

#endif // CUTILS_H

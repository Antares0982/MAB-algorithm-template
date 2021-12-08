#ifndef CUTILS_H
#define CUTILS_H

#include <utility>

namespace mabCutils {
    class node {
    public:
        node *next;
        double val;

    public:
        node() : next(nullptr), val(0){};
        node(double _val) : next(nullptr), val(_val){};
    };

    class mabnodescpp {
    private:
        int _size;
        node *head;
        node *tail;

    public:
        mabnodescpp() : _size(0), head(nullptr), tail(nullptr){};

        mabnodescpp(double _val) : _size(1) {
            head = new node(_val);
            tail = head;
        }

        mabnodescpp(node &_h) : _size(1) {
            head = &_h;
            tail = head;
        }

        mabnodescpp(node *_h) : _size(1) {
            head = _h;
            tail = head;
        }

        mabnodescpp(const mabnodescpp &ns) : _size(ns._size) {
            head = ns.head;
            tail = ns.tail;
        }

        node *gethead() { return head; }

        node *gettail() { return tail; }

        int size() { return _size; }

        void append(double _val) {
            if (!tail) {
                _size = 1;
                head = new node(_val);
                tail = head;
                return;
            }
            tail->next = new node(_val);
            tail = tail->next;
            ++_size;
        }

        double avg() {
            double ans = 0.0;
            for (auto p = gethead(); p; p = p->next) ans += p->val;
            return ans / size();
        }

        ~mabnodescpp() {
            if (!head) return;
            auto p = head;
            node *p1;
            while (p) {
                p1 = p->next;
                delete p;
                p = p1;
            }
        }
    };

    double catonialpha(const double &v, const int &itercount, const int &_size);

    double psi(const double &x);

    double dpsi(const double &x);

    double sumpsi(const double &v, const int &itercount, const double &guess, mabnodescpp &nodes);

    double dsumpsi(const double &v, const int &itercount, const double &guess, mabnodescpp &nodes);

    double nt_iter(const double &v, const int &itercount, const double &guess, mabnodescpp &nodes, const double &fguess);

    std::pair<double, int> getcatoni(const double &v, const int &itercount, double &guess, mabnodescpp &nodes, const double &tol);
} // namespace mabCutils

#endif // CUTILS_H
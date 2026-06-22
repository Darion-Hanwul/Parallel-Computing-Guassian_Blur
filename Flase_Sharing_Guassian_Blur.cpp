#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

const long long ITER = 100000000;

struct PaddedCounter
{
    long long value;
    char padding[64];
};

int main()
{
    int threads =
        omp_get_max_threads();

    cout << "Threads : "
         << threads
         << endl;

    /*FALSE SHARING*/
    vector<long long>
        counter(threads);

    double start1 =
        omp_get_wtime();

#pragma omp parallel
    {
        int tid =
            omp_get_thread_num();

        for (long long i = 0;
             i < ITER;
             i++)
        {
            counter[tid]++;
        }
    }

    double end1 =
        omp_get_wtime();

    /*TANPA FALSE SHARING*/
    vector<PaddedCounter>
        padded(threads);

    double start2 =
        omp_get_wtime();

#pragma omp parallel
    {
        int tid =
            omp_get_thread_num();

        for (long long i = 0;
             i < ITER;
             i++)
        {
            padded[tid].value++;
        }
    }

    double end2 =
        omp_get_wtime();

    cout << endl;
    cout << "FALSE SHARING"
         << endl;

    cout << "Execution Time : "
         << end1 - start1
         << " seconds"
         << endl;

    cout << endl;

    cout << "TANPA FALSE SHARING"
         << endl;

    cout << "Execution Time : "
         << end2 - start2
         << " seconds"
         << endl;

    return 0;
}
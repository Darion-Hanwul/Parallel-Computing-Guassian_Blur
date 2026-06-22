#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;

int main()
{
    Mat input = imread("images/Gambar.jpeg");

    if (input.empty())
    {
        cout << "Gagal membaca gambar." << endl;
        return -1;
    }

    Mat output = input.clone();

    omp_set_num_threads(8);

    double start = omp_get_wtime();

#pragma omp parallel for schedule(static)
    for (int y = 1; y < input.rows - 1; y++)
    {
        for (int x = 1; x < input.cols - 1; x++)
        {
            int kernel[7][7] =
            {
                {0,0,1,2,1,0,0},
                {0,3,13,22,13,3,0},
                {1,13,59,97,59,13,1},
                {2,22,97,159,97,22,2},
                {1,13,59,97,59,13,1},
                {0,3,13,22,13,3,0},
                {0,0,1,2,1,0,0}
            };

            int blue = 0;
            int green = 0;
            int red = 0;

            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    Vec3b pixel =
                        input.at<Vec3b>(
                            y + ky,
                            x + kx
                        );

                    int w =
                        kernel[ky + 1][kx + 1];

                    blue += pixel[0] * w;
                    green += pixel[1] * w;
                    red += pixel[2] * w;
                }
            }

            output.at<Vec3b>(y, x)[0] = blue / 1003;

            output.at<Vec3b>(y, x)[1] = green / 1003;

            output.at<Vec3b>(y, x)[2] = red / 1003;
        }
    }

    double end = omp_get_wtime();

    double elapsed = end - start;

    cout << "OPENMP GAUSSIAN BLUR"
         << endl;

    cout << "Image      : Gambar.jpeg"
         << endl;

    cout << "Resolution : "
         << input.cols
         << " x "
         << input.rows
         << endl;

    cout << "Threads    : "
         << omp_get_max_threads()
         << endl;

    cout << "Execution Time : "
         << elapsed
         << " seconds"
         << endl;

    imwrite(
        "output/OpenMPGaussianBlur_output8.jpeg",
        output
    );

    cout << "Output disimpan." << endl;

    return 0;
}
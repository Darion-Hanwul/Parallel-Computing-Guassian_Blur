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

    int kernel[7][7] =
    {
        {0, 0, 1, 2, 1, 0, 0},
        {0, 3, 13, 22, 13, 3, 0},
        {1, 13, 59, 97, 59, 13, 1},
        {2, 22, 97, 159, 97, 22, 2},
        {1, 13, 59, 97, 59, 13, 1},
        {0, 3, 13, 22, 13, 3, 0},
        {0, 0, 1, 2, 1, 0, 0}
    };

    const int totalWeight = 1003;

    double start = omp_get_wtime();

    for (int y = 3; y < input.rows - 3; y++)
    {
        for (int x = 3; x < input.cols - 3; x++)
        {
            int blue = 0;
            int green = 0;
            int red = 0;

            for (int ky = -3; ky <= 3; ky++)
            {
                for (int kx = -3; kx <= 3; kx++)
                {
                    Vec3b pixel =
                        input.at<Vec3b>(y + ky, x + kx);

                    int w =
                        kernel[ky + 3][kx + 3];

                    blue += pixel[0] * w;
                    green += pixel[1] * w;
                    red += pixel[2] * w;
                }
            }

            output.at<Vec3b>(y, x)[0] =
                blue / totalWeight;

            output.at<Vec3b>(y, x)[1] =
                green / totalWeight;

            output.at<Vec3b>(y, x)[2] =
                red / totalWeight;
        }
    }

    double end = omp_get_wtime();

    double elapsed = end - start;

    cout << "SERIAL GAUSSIAN BLUR"
         << endl;

    cout << "Image      : Gambar.jpeg"
         << endl;

    cout << "Resolution : "
         << input.cols
         << " x "
         << input.rows
         << endl;

    cout << "Warna  : "
         << input.channels()
         << endl;

    cout << "Execution Time : "
         << elapsed
         << " seconds"
         << endl;

    imwrite(
        "output/SerialGaussianBlur_output.jpeg",
        output
    );

    cout << "Output disimpan."
         << endl;

    return 0;
}
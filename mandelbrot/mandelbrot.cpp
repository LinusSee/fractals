#include <iostream>
#include <complex>
#include <tuple>

using namespace std;


int mandelbrot_point(complex<double> c, int max_iterations)
{
	std::complex<double> z(0, 0);
	for(int i = 0; i < max_iterations; i++)
	{
		z = z * z + c;
		if(abs(z) > 4)
		{
			return i;
		}
	}
	return max_iterations;
}

int** mandelbrot_set(double start_x, double end_x, double start_y, double end_y, int num_points, int max_iterations)
{
	double spacing_x = abs(end_x - start_x) / (num_points - 1);
	double spacing_y = abs(end_y - start_y) / (num_points - 1);

	cout << "SpacingX: " << spacing_x << endl;
	cout << "SpacingY: " << spacing_y << endl;

	double current_x = start_x;
	double current_y = start_y;

	int** m_set = new int*[num_points];
	for(int i = 0; i < num_points; i++)
	{
		m_set[i] = new int[num_points];
		for(int j = 0; j < num_points; j++)
		{

			complex<double> c(current_x, current_y);
			int iterations = mandelbrot_point(c, max_iterations);

			m_set[i][j] = iterations;
			current_x = current_x + spacing_x;
		}
		current_x = start_x;
		current_y = current_y + spacing_y;
	}

	return m_set;
}


int main()
{
	int num_points = 10;
	int** m_set = mandelbrot_set(-2.25, 0.75, -1.5, 1.5, num_points, 120);

	for(int i = 0; i < num_points; i++)
	{
		for(int j = 0; j < num_points; j++)
		{
			cout << m_set[i][j] << " - ";
		}
		cout << endl;
	}

	return 0;
}

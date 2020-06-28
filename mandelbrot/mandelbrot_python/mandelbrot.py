import cmath
import numpy as np
import matplotlib.pyplot as plt



def mandelbrot_point(c, max_iterations):
	z = complex(0, 0)
	for x in range(max_iterations):
		z = z**2 + c
		if abs(z) > 4:
			return x

	return max_iterations


def mandelbrot_set(start_x, end_x, start_y, end_y, num_points, max_iterations):
	xs = np.linspace(start_x, end_x, num_points)
	ys = np.linspace(start_y, end_y, num_points)

	result = []
	for x in xs:
		row = []
		for y in ys:
			c = complex(x, y)
			iterations = mandelbrot_point(c, max_iterations)
			row.append(iterations)
		result.append(row)
	return np.array(result)


def main():
	result = mandelbrot_set(-2.25, 0.75, -1.5, 1.5, 1000, 120)
	#scaled_result = (result / 120) * 255

	plt.figure()
	extent = (-2.25, 0.75, -1.5, 1.5)
	plt.imshow(result.T, cmap="inferno", extent=extent, interpolation="nearest")
	plt.show()


if __name__ == "__main__":
	main()

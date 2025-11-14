import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_1d(input_array, kernel_array):
    return len(input_array) - len(kernel_array) + 1


# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array, kernel_array):
    # Tip: start by initializing an empty output array (you can use your function above to calculate the correct size).
    # Then fill the cells in the array with a loop.
    output_size = compute_output_size_1d(input_array, kernel_array)
    output = np.empty(output_size)

    kernel_len = len(kernel_array)

    for i in range(output_size):
        output[i] = np.sum(input_array[i : i + kernel_len] * kernel_array)

    return output


# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

input_array_2 = np.array([1,2,1,3,5,6])
kernel_array_2 = np.array([2,4])
print(convolve_1d(input_array_2, kernel_array_2))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

two_dim = np.array([[1,2,3],
                     [4,4,5]])

a = np.shape(two_dim)
print(a[0])


# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
    shape_input = np.shape(input_matrix)
    shape_kernel = np.shape(kernel_matrix)
    input_height = shape_input[0]
    input_width = shape_input[1]
    kernel_height = shape_kernel[0]
    kernel_width = shape_kernel[1]

    return input_height - kernel_height + 1, input_width - kernel_width + 1


# test
example_input = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
example_kernel = np.array([[1, 0],[0, 1]])

print(compute_output_size_2d(example_input, example_kernel))

# -----------------------------------------------
lajos = compute_output_size_2d(example_input, example_kernel)
print(lajos[0])

# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix, kernel_matrix):
    # Tip: same tips as above, but you might need a nested loop here in order to
    # define which parts of the input matrix need to be multiplied with the kernel matrix.
    output_shape = compute_output_size_2d(input_matrix, kernel_matrix)
    output_hight = output_shape[0]
    output_width = output_shape[1]
    
    output = np.empty((output_hight, output_width))

    kernel_shape = np.shape(kernel_matrix)
    kernel_height = kernel_shape[0]
    kernel_width = kernel_shape[1]
    

    for i in range(output_hight):
        for j in range(output_width):
            output[i, j] = np.sum(input_matrix[i : i + kernel_height, j : j + kernel_width] * kernel_matrix)

    return output

# checking:
test_input_matrix = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
test_kernel_matrix = np.array([[1, 0],[0, 1]])

print(convolute_2d(test_input_matrix, test_kernel_matrix))

# -----------------------------------------------
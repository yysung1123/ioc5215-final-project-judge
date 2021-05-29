import numpy as np
import itertools

acc0 = 61.878882
latency0 = 16.30464
acc_array = np.array([0.308031, 0.246401, 0.385460, 0.549228, 0.355705, 0.020765, 0.041229, 0.034954, 0.067013, 0.135598, 0.066404, -0.012060, 0.054933, 0.009343, 0.014635, 0.057120, 0.043826, 0.056370, 0.089531, 0.113646, 0.005593, 0.025202, 0.027489, 0.003217, 0.013580, 0.088727, 0.116691, 0.120979, 0.203725, 0.153543, -0.020289, -0.026291, -0.027556, -0.040638, -0.031194, -0.014607, 0.006102, -0.006670, 0.004494, 0.002630])
latency_array = np.array([-8.06211, -2.69638, -1.13344, -0.22013, -0.55329, -2.32253, -0.90273, -1.01517, -0.54179, -0.30544, -2.33407, -0.94015, -0.11907, 0.01387, -0.27301, 2.85597, 1.29418, 0.65506, 0.68359, 0.62364, 1.17867, 0.49333, 0.10513, 0.23057, 0.01159, 2.15201, 1.01542, 0.70170, 0.87146, 0.45674, 1.91980, 0.70742, 0.16173, 0.21330, 0.17648])

def get_acc(x):
    return acc0 + sum(x * acc_array)

def get_latency(x):
    return latency0 + sum(x * latency_array)

def predict(array):
    d1, d2, d3, d4, d5 = array[0:5]
    avg_e1, avg_e2, avg_e3, avg_e4, avg_e5 = array[5:10]
    avg_k1, avg_k2, avg_k3, avg_k4, avg_k5 = array[10:15]
    e1, e2, e3, e4, e5 = array[15:20]
    k1, k2, k3, k4, k5 = array[20:25]

    # use the accuracy predictor to predict the accuracy of a model (Do NOT modify!)
    x_acc = np.array([d1, d2, d3, d4, d5,
                    avg_e1, avg_e2, avg_e3, avg_e4, avg_e5,
                    avg_k1, avg_k2, avg_k3, avg_k4, avg_k5,
                    e1, e2, e3, e4, e5,
                    k1, k2, k3, k4, k5,
                    (d1 - 1) * avg_e1,
                    (d2 - 1) * avg_e2,
                    (d3 - 1) * avg_e3,
                    (d4 - 1) * avg_e4,
                    (d5 - 1) * avg_e5,
                    d1 * (d1 - 1) * avg_e1,
                    d2 * (d2 - 1) * avg_e2,
                    d3 * (d3 - 1) * avg_e3,
                    d4 * (d4 - 1) * avg_e4,
                    d5 * (d5 - 1) * avg_e5,
                    d1 * avg_k1,
                    d2 * avg_k2,
                    d3 * avg_k3,
                    d4 * avg_k4,
                    d5 * avg_k5])

    predicted_acc = get_acc(x_acc)

    # use the latency predictor to predict the latency of a model (Do NOT modify!)
    x_latency = np.array([d1, d2, d3, d4, d5,
                        avg_e1, avg_e2, avg_e3, avg_e4, avg_e5,
                        avg_k1, avg_k2, avg_k3, avg_k4, avg_k5,
                        e1, e2, e3, e4, e5,
                        k1, k2, k3, k4, k5,
                        d1 * avg_e1,
                        d2 * avg_e2,
                        d3 * avg_e3,
                        d4 * avg_e4,
                        d5 * avg_e5,
                        d1 * avg_k1,
                        d2 * avg_k2,
                        d3 * avg_k3,
                        d4 * avg_k4,
                        d5 * avg_k5])

    predicted_latency = get_latency(x_latency)

    return (predicted_acc, predicted_latency)

def check_range(array):
    error = False
    error_messages = []

    # 檢查depth、expand ration、kernel size是否合規定
    if array.shape != (35,):
        return (True, [f"Array shape should be (35,). Got {array.shape}"])

    column_name = np.array(['d1', 'd2', 'd3', 'd4', 'd5',
                            'avg_e1', 'avg_e2', 'avg_e3', 'avg_e4', 'avg_e5',
                            'avg_k1', 'avg_k2', 'avg_k3', 'avg_k4', 'avg_k5',
                            'e1', 'e2', 'e3', 'e4', 'e5',
                            'k1', 'k2', 'k3', 'k4', 'k5'])
    depth_error = np.isin(array[0:5], [2, 3, 4], invert=True).tolist()
    for x in column_name[0:5][depth_error]:
        print(x, 'is not in [2, 3, 4]')
        error_messages.append(f'{x} is not in [2, 3, 4]')
        error = True
    expand_ratio_error = np.isin(array[5:10], [2, 3, 4, 6], invert=True).tolist()
    for x in column_name[5:10][expand_ratio_error]:
        print(x, 'is not in [2, 3, 4, 6]')
        error_messages.append(f'{x} is not in [2, 3, 4, 6]')
        error = True
    kernel_size_error = np.isin(array[11:15], [3, 5, 7], invert=True).tolist()
    for x in column_name[11:15][kernel_size_error]:
        print(x, 'is not in [3, 5, 7]')
        error_messages.append(f'{x} is not in [3, 5, 7]')
        error = True
    expand_ratio_error = np.isin(array[16:20], [2, 3, 4, 6], invert=True).tolist()
    for x in column_name[16:20][expand_ratio_error]:
        print(x, 'is not in [2, 3, 4, 6]')
        error_messages.append(f'{x} is not in [2, 3, 4, 6]')
        error = True
    kernel_size_error = np.isin(array[21:25], [3, 5, 7], invert=True).tolist()
    for x in column_name[21:25][kernel_size_error]:
        print(x, 'is not in [3, 5, 7]')
        error_messages.append(f'{x} is not in [3, 5, 7]')
        error = True

    return (error, error_messages)

if __name__ == '__main__':

    # 1. choose depth from [2, 3, 4]:
    #    d1, d2, d3, d4, d5
    # 2. choose expand ratio from [2, 3, 4, 6]:
    #    avg_e1, avg_e2, avg_e3, avg_e4, avg_e5, e1, e2, e3, e4, e5
    # 3. choose kernel size from [3, 5, 7]:
    #    avg_k1, avg_k2, avg_k3, avg_k4, avg_k5, k1, k2, k3, k4, k5

    array = np.load('legal.npy')
    error = check_range(array)

    if not error :
        # example architecture (you need to search the best architecture yourself!)
        d1, d2, d3, d4, d5 = array[0:5]
        avg_e1, avg_e2, avg_e3, avg_e4, avg_e5 = array[5:10]
        avg_k1, avg_k2, avg_k3, avg_k4, avg_k5 = array[10:15]
        e1, e2, e3, e4, e5 = array[15:20]
        k1, k2, k3, k4, k5 = array[20:25]

        # use the accuracy predictor to predict the accuracy of a model (Do NOT modify!)
        x_acc = np.array([d1, d2, d3, d4, d5,
                        avg_e1, avg_e2, avg_e3, avg_e4, avg_e5,
                        avg_k1, avg_k2, avg_k3, avg_k4, avg_k5,
                        e1, e2, e3, e4, e5,
                        k1, k2, k3, k4, k5,
                        (d1 - 1) * avg_e1,
                        (d2 - 1) * avg_e2,
                        (d3 - 1) * avg_e3,
                        (d4 - 1) * avg_e4,
                        (d5 - 1) * avg_e5,
                        d1 * (d1 - 1) * avg_e1,
                        d2 * (d2 - 1) * avg_e2,
                        d3 * (d3 - 1) * avg_e3,
                        d4 * (d4 - 1) * avg_e4,
                        d5 * (d5 - 1) * avg_e5,
                        d1 * avg_k1,
                        d2 * avg_k2,
                        d3 * avg_k3,
                        d4 * avg_k4,
                        d5 * avg_k5])

        predicted_acc = get_acc(x_acc)

        # use the latency predictor to predict the latency of a model (Do NOT modify!)
        x_latency = np.array([d1, d2, d3, d4, d5,
                            avg_e1, avg_e2, avg_e3, avg_e4, avg_e5,
                            avg_k1, avg_k2, avg_k3, avg_k4, avg_k5,
                            e1, e2, e3, e4, e5,
                            k1, k2, k3, k4, k5,
                            d1 * avg_e1,
                            d2 * avg_e2,
                            d3 * avg_e3,
                            d4 * avg_e4,
                            d5 * avg_e5,
                            d1 * avg_k1,
                            d2 * avg_k2,
                            d3 * avg_k3,
                            d4 * avg_k4,
                            d5 * avg_k5])

        predicted_latency = get_latency(x_latency)

        print('predicted acc:', predicted_acc) # 73.904113
        print('predicted latency:', predicted_latency) # 130.53857
    else:
        print('\nPlease modify the architecture and try again!')
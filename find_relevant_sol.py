import os
import pickle

import numpy as np

# loop over the files in the directory to find the solution that has a g value 1e-6 with the lowest f value
# folder = "results/temporary_results_11-04-17-25-47"
folder = "results/temporary_results_13-04-15-35-15_more_nodes"
lowest_cost_function_file = None
lowest_cost_function_value = np.inf
best_solution_g = None
for i_file in os.listdir(folder):
    if i_file.endswith(".pkl") and not i_file.endswith("_integrated.pkl"):
        with open(folder + "/" + i_file, "rb") as f:
            data = pickle.load(f)
            g = np.array(data["g"])
            max_g = np.max(np.abs(g))
            f = float(data["f"])
            if max_g > 1e-6:
                continue
            if f < lowest_cost_function_value:
                lowest_cost_function_value = f
                lowest_cost_function_file = i_file
                best_solution_g = max_g

print("The file with the lowest cost function value is: ", lowest_cost_function_file)
print("The lowest cost function value is: ", lowest_cost_function_value)
print("The g value of the best solution is: ", best_solution_g)

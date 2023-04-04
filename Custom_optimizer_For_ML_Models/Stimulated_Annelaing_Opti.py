import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# load the Boston Housing dataset
data = load_boston()
X, y = data.data, data.target

# split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# define the objective function
def svr_mse(params, X_train, y_train, X_val, y_val):
    C, gamma, epsilon = params
    svr = SVR(C=C, gamma=gamma, epsilon=epsilon)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_val)
    mse = np.mean((y_pred - y_val) ** 2)
    return mse

# define the bounds for the hyperparameters
bounds = [(0.01, 100), # C
          (0.0001, 10), # gamma
          (0.001, 1)]  # epsilon

# define the initial temperature and cooling rate for simulated annealing
T0 = 10
cooling_rate = 0.95

# define the maximum number of iterations for simulated annealing
max_iter = 3000

# define a function to perform simulated annealing optimization
def custom_simulated_annealing(X_train, y_train, X_val, y_val, T0, cooling_rate, max_iter):
    current_sol = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds])
    current_mse = svr_mse(current_sol, X_train, y_train, X_val, y_val)
    best_sol = current_sol
    best_mse = current_mse
    for i in range(max_iter):
        # print("Optimizing!!!!!")
        new_sol = np.clip(current_sol + np.random.normal(size=len(bounds), scale=0.1), [b[0] for b in bounds], [b[1] for b in bounds])
        new_mse = svr_mse(new_sol, X_train, y_train, X_val, y_val)
        delta_mse = new_mse - current_mse
        prob_accept = np.exp(-delta_mse / T0)
        if np.random.rand() < prob_accept:
            current_sol = new_sol
            current_mse = new_mse
        if current_mse < best_mse:
            best_sol = current_sol
            best_mse = current_mse
        T0 *= cooling_rate
    return best_sol, best_mse

# run simulated annealing to optimize the hyperparameters
best_params, best_mse = custom_simulated_annealing(X_train, y_train, X_val, y_val, T0, cooling_rate, max_iter)

# train an SVR model with the best hyperparameters on the entire training set
C, gamma, epsilon = best_params
svr = SVR(C=C, gamma=gamma, epsilon=epsilon)
svr.fit(X_train, y_train)

#without optimization
svr_no = SVR()
svr_no.fit(X_train, y_train)


svr_at = SVR(C=50.639, gamma=5.204, epsilon=0.001)
svr_at.fit(X_train, y_train)
# evaluate the model on the test set
X_test, y_test = X_val, y_val # just for demonstration purposes
y_pred = svr.predict(X_test)
y_pred_no = svr_no.predict(X_test)
y_pred_at = svr_at.predict(X_test)

mse = np.mean((y_pred - y_test) ** 2)
mse_no = np.mean((y_pred_no - y_test) ** 2)
mse_at = np.mean((y_pred_at - y_test) ** 2)
print("Best hyperparameters: C=%.3f, gamma=%.3f, epsilon=%.3f" % (C, gamma, epsilon))
print(mse, mse_no, mse_at)

#C=26.696, gamma=0.001, epsilon=0.614
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Load the Boston Housing dataset
boston = datasets.load_boston()
X = boston.data
y = boston.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the objective function
def objective_function(params):
    c, gamma, epsilon = params
    svr = SVR(C=c, gamma=gamma, epsilon=epsilon)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Define the bounds for each hyperparameter
lower_bound = [1e-3, 1e-3, 1e-3]
upper_bound = [10, 10, 10]

# Define the PSO algorithm
n_particles = 10
n_dimensions = 3
max_iter = 50
w = 0.9
c1 = 0.5
c2 = 0.3

# Define the PSO function
def pso(objective_function, lower_bound, upper_bound, n_particles, n_dimensions, max_iter, w, c1, c2):
    # Initialize the particles randomly within the search space
    particles = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_particles, n_dimensions))

    # Initialize the personal best positions and global best position
    personal_best_positions = particles
    global_best_position = particles[np.argmin([objective_function(p) for p in particles])]

    # Initialize the velocities
    velocities = np.zeros((n_particles, n_dimensions))

    # Perform optimization
    for i in range(max_iter):
        # Update the velocities
        r1 = np.random.rand(n_particles, n_dimensions)
        r2 = np.random.rand(n_particles, n_dimensions)
        velocities = w * velocities + c1 * r1 * (personal_best_positions - particles) + c2 * r2 * (global_best_position - particles)

        # Update the positions
        particles = particles + velocities

        # Enforce the bounds
        particles = np.clip(particles, lower_bound, upper_bound)

        # Update the personal best positions and global best position
        for j in range(n_particles):
            if objective_function(particles[j]) < objective_function(personal_best_positions[j]):
                personal_best_positions[j] = particles[j]
            if objective_function(personal_best_positions[j]) < objective_function(global_best_position):
                global_best_position = personal_best_positions[j]

    return global_best_position

# Call the PSO function with the specified hyperparameters
hyperparameters = pso(objective_function, lower_bound, upper_bound, n_particles, n_dimensions, max_iter, w, c1, c2)

# Train the SVM model with the best hyperparameters
c, gamma, epsilon = hyperparameters
svr = SVR(C=c, gamma=gamma, epsilon=epsilon)
svr.fit(X_train, y_train)

# Evaluate the performance on the test set
y_pred = svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

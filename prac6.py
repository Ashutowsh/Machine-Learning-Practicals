import numpy as np

def activation_function(x):
    return 1 if x >= 0 else -1

def fit(X, y, learning_rate=0.1, n_iterations=5):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0


    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}")
        for idx, x_i in enumerate(X):
            linear_output = np.dot(x_i, weights) + bias
            y_predicted = activation_function(linear_output)

            update = learning_rate * (y[idx] - y_predicted)
            weights += update * x_i
            bias += update
            
            print(f"  Sample {idx + 1}: Predicted={y_predicted}, Actual={y[idx]}, Weights={weights}, Bias={bias}")
            
    return weights, bias

def predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    return np.array([activation_function(x) for x in linear_output])


X_and = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y_and = np.array([1, -1, -1, -1])                       

X_or = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])  
y_or = np.array([1, 1, 1, -1])                          

weights_and, bias_and = fit(X_and, y_and, learning_rate=0.1, n_iterations=5)
print("\nFinal Weights after training AND:", weights_and)
print("Final Bias after training AND:", bias_and)

predictions_and = predict(X_and, weights_and, bias_and)
print("Predictions for AND:", predictions_and)

weights_or, bias_or = fit(X_or, y_or, learning_rate=0.1, n_iterations=5)
print("\nFinal Weights after training OR:", weights_or)
print("Final Bias after training OR:", bias_or)

predictions_or = predict(X_or, weights_or, bias_or)
print("Predictions for OR:", predictions_or)

# Bothe AND and OR Gate
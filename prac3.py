import numpy as np
import matplotlib.pyplot as plt

heights = np.array([5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8])
weights = np.array([53, 55, 59, 61, 65, 68, 70, 74, 76])

mean_x = np.mean(heights)
mean_y = np.mean(weights)

numerator = np.sum((heights - mean_x) * (weights - mean_y))
denominator = np.sum((heights - mean_x) ** 2)


w1 = numerator / denominator
w0 = mean_y - w1 * mean_x

print(f'Regression Coefficient w0 (intercept): {w0:.2f}')
print(f'Regression Coefficient w1 (slope): {w1:.2f}')

h = 5.9 # Calculate ans for this height
predicted_weight = w0 + w1 * h
print(f'Predicted weight for height h = 5.9 : {predicted_weight:.2f}')

# Plot the line and the points
plt.scatter(heights, weights, label='Data Points')
plt.plot(heights, w0 + w1 * heights, color='red', label='Regression Line')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Linear Regression')
plt.legend()
plt.show()

predicted_weights = w0 + w1 * heights
points_on_line = np.sum(predicted_weights == weights)
print(f'\n\nPredicted weights are : {predicted_weights}')
print(f'\nNumber of points exactly on the line: {points_on_line}')


# If we round off the predicted weights
print("After Rounding off")
predicted_weights_rounded = np.round(predicted_weights)
points_on_line = np.sum(predicted_weights_rounded == weights)
print(f'\n\nPredicted weights are : {predicted_weights_rounded }')
print(f'\nNumber of points exactly on the line: {points_on_line}')

# Calculating Errors

errors = weights - predicted_weights
distances = np.abs(errors)
print(f"\nErrors for each value : {errors}")
print(f"\nDistances for each value : {distances}") # Taking absolutes for each errors
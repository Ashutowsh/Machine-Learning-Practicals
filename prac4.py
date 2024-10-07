import math

def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i]-point2[i])**2
    return math.sqrt(distance)

def knn_classification(train_data, test_point, k):
    distances = []
    for train_point in train_data:
        distance = euclidean_distance(train_point[:-1], test_point)
        distances.append((distance, train_point[-1]))
    distances.sort()
    neighbors = [point[-1] for point in distances[:k]]
    ans = max(set(neighbors), key=neighbors.count)
    return ans

def knn_regression(train_data, test_point, k):
    distances = []
    for train_point in train_data:
        distance = euclidean_distance(train_point[:-1], test_point)
        distances.append((distance, train_point[-1]))
    distances.sort()
    neighbors = [point[-1] for point in distances[:k]]
    ans = sum(neighbors)/len(neighbors)
    return ans

def mean_squared_error(actual, predicted):
    return sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)


length = [10.0, 11.0, 12.0, 7.0, 9.0, 8.0,6.0,15.0,14.0,7.0,10.0,13.0,9.0,5.0,5.0]
weight = [15.0, 6.0 , 14.0 , 9.0 ,14.0,12.0,11.0,10.0,8.0,12.0,6.0,8.0,7.0,8.0,10.0]
cost = [45,37,48,33,38,40,35,50,46,35,36,44,32,30,30]

train_data = list(zip(length, weight, cost))

test_point = [7, 8]
k = 7

predicted_class = knn_classification(train_data, test_point, k)
print("Predicted class for test point", test_point, "is:", predicted_class)

predicted_regression_ans = knn_regression(train_data, test_point, k)
print("\nPredicted regression value for test point", test_point, "is:", predicted_regression_ans)

test_points = [[10, 15], [7, 9], [6, 11]]
actual_costs = [45, 33, 35]  

predicted_costs = [knn_regression(train_data, test, k) for test in test_points]
print("\nPredicted costs:", predicted_costs)

mse = mean_squared_error(actual_costs, predicted_costs)
print("Mean Squared Error for regression:", mse)

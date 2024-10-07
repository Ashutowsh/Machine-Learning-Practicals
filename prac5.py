import pandas as pd
import numpy as np

df = pd.read_csv('PlayTennis.csv')

df = df.drop(columns=['day'])


def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value


def information_gain(df, attribute, target):
    original_entropy = entropy(df[target])
    values = df[attribute].unique()
    weighted_entropy = 0.0
    for value in values:
        subset = df[df[attribute] == value]
        weight = len(subset) / len(df)
        weighted_entropy += weight * entropy(subset[target])
    info_gain = original_entropy - weighted_entropy
    return info_gain


def attribute_entropy(df, attribute, target):
    values = df[attribute].unique()
    entropies = {}
    for value in values:
        subset = df[df[attribute] == value]
        entropies[value] = entropy(subset[target])
    return entropies


target_entropy = entropy(df['play'])
print(f'Entropy of the overall sample: {target_entropy:.4f}')


attributes = ['outlook', 'temp', 'humidity', 'wind']
info_gains = {attr: information_gain(df, attr, 'play') for attr in attributes}
entropies_per_attribute = {attr: attribute_entropy(df, attr, 'play') for attr in attributes}


print('\nInformation Gain for each attribute:')
for attr, gain in info_gains.items():
    print(f'{attr}: {gain:.4f}')


print('\nEntropy for each attribute value:')
for attr, entropies in entropies_per_attribute.items():
    print(f'{attr}:')
    for value, entropy_value in entropies.items():
        print(f'  {value}: {entropy_value:.4f}')


root_attribute = max(info_gains, key=info_gains.get)
print(f'\nRoot attribute: {root_attribute}')


def select_best_attribute(df, attributes, target):
    gains = {attribute: information_gain(df, attribute, target) for attribute in attributes}
    return max(gains, key=gains.get)

def build_decision_tree(df, target, attributes):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if not attributes:
        return df[target].value_counts().idxmax()

    best_attribute = select_best_attribute(df, attributes, target)
    tree = {best_attribute: {}}
    for value in df[best_attribute].unique():
        subset_df = df[df[best_attribute] == value].drop(columns=best_attribute)
        subset_target = subset_df[target]
        if subset_df.empty:
            tree[best_attribute][value] = df[target].value_counts().idxmax()
        else:
            remaining_attributes = attributes.copy()
            remaining_attributes.remove(best_attribute)
            tree[best_attribute][value] = build_decision_tree(subset_df, target, remaining_attributes)
    return tree


attributes = list(df.columns[df.columns != 'play'])
decision_tree = build_decision_tree(df, 'play', attributes)


def print_decision_tree(tree, depth=0, indent='  '):
    """Print the decision tree in a readable format."""
    if isinstance(tree, dict):
        for attribute, subtree in tree.items():
            print(indent * depth + f"If {attribute}:")
            for value, subsubtree in subtree.items():
                print(indent * (depth + 1) + f"Then {value}:")
                print_decision_tree(subsubtree, depth + 2, indent)
    else:
        print(indent * (depth + 1) + f"Predict: {tree}")


print("\nDecision Tree:")
print_decision_tree(decision_tree)

# --------------------- Using Inbuilt Functions ---------------------

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
for col in df.columns[:-1]:
    df[col] = le.fit_transform(df[col])


X = df.iloc[:, :-1]
y = df.iloc[:, -1]

clf = DecisionTreeClassifier()
clf.fit(X, y)

tree_rules = export_text(clf, feature_names=list(X.columns))
print("\nInbuilt Decision Tree rules:")
print(tree_rules)

root_feature = clf.tree_.feature[0]
root_feature_name = X.columns[root_feature]
print("\nRoot of the tree:", root_feature_name)

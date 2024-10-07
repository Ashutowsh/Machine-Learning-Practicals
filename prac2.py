def candidate_elimination(training_data, attributes):
    S = [['Ø'] * len(attributes)] 

    G = [['?' for _ in range(len(attributes))]] 

    print("Initial S:", S)
    print("Initial G:", G)

    for i, example in enumerate(training_data):
        print(f"\nProcessing example {i + 1}: {example[:-1]} -> {example[-1]}")

        if example[-1] == 'Yes':

            G = [g for g in G if all((g[i] == '?' or g[i] == example[i]) for i in range(len(example) - 1))]

            for s in S:
                for i in range(len(s)):
                    if s[i] == 'Ø': 
                        s[i] = example[i]
                    elif s[i] != example[i]:
                        s[i] = '?'

            print("Updated S (after generalizing for positive example):", S)
            print("Updated G (after removing inconsistent G hypotheses):", G)

        else:
            S = [s for s in S if not all((s[i] == '?' or s[i] == example[i]) for i in range(len(example) - 1))]

            new_G = []
            for g in G:
                for i in range(len(g)):
                    if g[i] == '?':
                        new_hypothesis = g.copy()
                        new_hypothesis[i] = example[i]
                        new_G.append(new_hypothesis)

            G = [g for g in new_G if any((g[i] == '?' or g[i] != example[i]) for i in range(len(example) - 1))]

            print("Updated S (after removing inconsistent S hypotheses):", S)
            print("Updated G (after specializing for negative example):", G)

    return S, G


training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

attributes = ['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast']

S_final, G_final = candidate_elimination(training_data, attributes)

print("\nFinal Specific Boundary (S):", S_final)
print("Final General Boundary (G):", G_final)

def find_s_algorithm(training_data):

    hypothesis = ['Ø'] * len(training_data[0])
    
    print("Initial hypothesis:", hypothesis)
    
    for i, example in enumerate(training_data):
        if example[-1] == "Yes": 
            if hypothesis == ['Ø'] * len(example):
                hypothesis = example[:-1] 
                print(f"\nFirst positive example (Day {i+1}): {example}")
                print("Updated hypothesis:", hypothesis)
            else:
                print(f"\nNext positive example (Day {i+1}): {example}")
                for attr in range(len(hypothesis)):
                    if hypothesis[attr] != example[attr]:
                        hypothesis[attr] = '?' 
                print("Updated hypothesis:", hypothesis)
        else:
            print(f"\nNegative example (Day {i+1}): {example} -> Ignored")
    
    return hypothesis

training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

final_hypothesis = find_s_algorithm(training_data)

print("\nFinal Hypothesis after applying FIND-S:", final_hypothesis)

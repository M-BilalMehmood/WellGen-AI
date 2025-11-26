import json

with open('data/training_data.json', 'r') as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")
print(f"Sources: {set([d['source'] for d in data])}")

# Sample a few examples
print("\nSample examples:")
for i in range(min(3, len(data))):
    text = data[i]['text']
    preview = text[:200] + "..." if len(text) > 200 else text
    print(f"\nExample {i+1}:")
    print(preview)

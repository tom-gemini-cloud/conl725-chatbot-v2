from transformers import pipeline

classifier = pipeline("sentiment-analysis")

print(classifier("I love you"))
print(classifier("I hate you"))
# Ungraded quiz

- Explore the Hub and look for the roberta-large-mnli checkpoint. What task does it perform?
  Ans:- Text Classification.
- What will following code return?
```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

Ans:- it will return word representation persons, organizations & locations.

- What will be the output of code?

```python
from transformers import pipeline

filler = pipeline("fill-mask", model="bert-base-cased")

result = filler("I want to [MASK] a new phone.")
print(result)
```

Answer: 5 probability output:

```
[
  {'score': 0.950, 'token_str': 'buy',    'sequence': 'I want to buy a new phone.'},
  {'score': 0.015, 'token_str': 'get',    'sequence': 'I want to get a new phone.'},
  {'score': 0.008, 'token_str': 'purchase', 'sequence': 'I want to purchase a new phone.'},
  {'score': 0.005, 'token_str': 'have',   'sequence': 'I want to have a new phone.'},
  {'score': 0.003, 'token_str': 'own',    'sequence': 'I want to own a new phone.'}
] 
```

- What will be the output?

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier("This is a course about the Transformers library")
```

Answer: It will show error because we don't pass any label here.

- What does “transfer learning” mean?
    - Ans:- First model transfer its trained weight to new model.

- True or False? A language model usually does not need labels for its pretraining.
    - Pre-training model usually self supervised, which means the labels are created automatically from the inputs
      (like predicting the next word or filling in some masked words.)

- Select the sentence that best describes the terms "model", "architecture" & "weight".
    - The same set of mathematical functions(architecture) can be used to build different models by using different 
  parameters.

- Which of these types of models would you use for completing prompts with generated text?
    - A decoder model.

- Which of those types of models would you use for summarizing texts?
    - A sequence to sequence model.


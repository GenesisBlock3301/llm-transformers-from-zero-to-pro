- What is the order of the language modeling pipeline?
    - Ans: The tokenizer handle text and returns IDs, the model handles these IDs and outputs prediction. The tokenizer
      can then be once again to convert these predictions back to some text. 

- How many dimensions does the tensor output by the base Transformer model have, and what are they?
    - Ans: The sequence length, the batch size, and the hidden size

- Which of the following is an example of subword tokenization?
    - Ans: WordPiece.

- What is a model head?
    - An additional component, usually made up of one or a few layers, to convert the transformer predictions to a
      task-specific output.

- What is an AutoModel?
    - A model that automatically trains on your data.

- What are the techniques to be aware of when batching sequences of different lengths together?
    - Padding

- What is the point of applying a SoftMax function to the logits output by a sequence classification model?
    - It applies a lower and upper bound so that they're understandable.
      Correct! The resulting values are bound between 0 and 1. That's not the only reason we use a SoftMax function,
      though.
- What method is most of the tokenizer API centered around?

  Calling the tokenizer object directly.
  Correct! Exactly! The `__call__` method of the tokenizer is a very powerful method which can handle pretty much anything.
  It is also the method used to retrieve predictions from a model.


- What does the result variable contain in this code sample?

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
result = tokenizer.tokenize("Hello!")
```
A list of strings, each string being a token
Correct! Absolutely! Convert this to IDs, and send them to a model!

- Is there something wrong with the following code?
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("gpt2")

encoded = tokenizer("Hey!", return_tensors="pt")
result = model(**encoded)
```

The tokenizer and model should always be from the same checkpoint.

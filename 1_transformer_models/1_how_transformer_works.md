# How transformer model work?

### Transformers are language models

- All transformers model firstly trained as a language models. All have been trained on large amount of data in a self
  supervised fashion.
- Self supervised learning means human are not needed to label the data.
- In general tasks pretrained models works well, but critical and domain specific tasks need transformer learning or
  fine-tuning.
- A task is predicting the next word in a sentence having read the n previous words. This is called `casual language
modeling` the output depends on past and present inputs not the future.
- `Mask language modeling` It predicts a masked word in the sentence. e.g; My <mask>(name) is SIfat.

### What is Transfer learning?

- Training from scratch gives you less accuracy rather fine-tuning with large model give you good accuracy.

Training from scratch is costly and time-consuming, on the other hand fine-tuning is better than training from scratch.

### General Transformer Architecture:



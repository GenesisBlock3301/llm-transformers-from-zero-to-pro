
### Managing Repetition: Keeping Output Fresh

LLMs love to repeat words/phrases. To stop that, we use two simple "punishment" tricks:

**Presence Penalty** → "If you already used this word ONCE, make it a bit harder to use again."
(Discourages reusing any word, even if it appeared just one time.)

**Frequency Penalty** → "The MORE you’ve already used a word, the HEAVIER the punishment."
(Really hates super-repeated words like saying “amazing” 10 times.)

Both work by quietly lowering the chance of picking repeated words → forces the model to say things in fresh, new ways.
Simple rule:
Higher presence = less topic repetition
Higher frequency = less word/spam repetition
Turn them up = fresher output
Turn them down = more natural but possibly repetitive

## Controlling Generation Length: Setting Boundaries

Three simple ways to stop the model from talking forever:

#### Token Limits:
`max_tokens` = “Don’t write more than X words”

`min_tokens` = “Write at least X words”

(Example: max_tokens = 100 → short paragraph)

#### Stop Sequences
Tell it: “When you see this exact text, STOP immediately.”

Common ones: \n\n, ###, END, </s>

Perfect for tweets, bullet points, or chat replies.
Let it finish naturally
Just don’t set anything → model stops when it thinks it’s done (usually good for open-ended answers).

#### Quick cheat-sheet:

Want a tweet? → max_tokens 60 + stop at \n

Want one paragraph? → max_tokens 150 + stop at \n\n

Want a full answer? → high/no max_tokens, no stop sequence

## Practical Challenges and Optimization

"F-F-S-M" (Fast First Slow More)

F = TTFT → First token latency (prefill speed)
"How fast do I start talking?"

F = TPOT → Follow-up token speed (decode speed)
"How fast do I keep talking?"

S = Throughput → Simultaneous requests per second
"How many people can I talk to at once?"

M = VRAM → Memory usage
"How much GPU RAM do I eat?"

## The Context Length Challenge

One of the most significant challenges in LLM inference is managing context length effectively. 
Longer contexts provide more information but come with significant costs:

Memory Usage: Grows quadratically with context length.

Processing Speed: Decreases linearly with longer contexts

Resource Allocation: Requires careful balancing of VRAM usage

## The KV Cache Optimization

To address these challenges, one of the most powerful optimizations is KV (Key-Value) caching. This technique significantly improves inference speed by storing and reusing intermediate calculations. This optimization:

- Reduces repeated calculations
- Improves generation speed
- Makes long-context generation practical
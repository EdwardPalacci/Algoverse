# Data Notes

## Dataset Origins

### GSM8K: OpenAI, October 2021

### TruthfulQA: OpenAI, September 2021

### SimpleQA: OpenAI, October 2024

## Dataset Grading

### SimpleQA
- Answers are graded after they become lowercase and spaces are eliminated
- No partial credit: answers are either correct or incorrect.

### GSM8K
- At first, only looks for number so ignore reasoning
- Units should be ignored
- When there are multiple numbers, only last one should be used
- Number has to match correct number

### TruthfulQA
- Based on semantic correctness, not exactly like correct answers
- If contains hallucinations, should not be correct
- We could see if strings match


## Known Difficulties

### GSM8k - needs to extract final answer from what the model gives

### SimpleQA - model can give answer in multiple different answer forms

### TruthfulQA - model can give correct answer but may not be listed in correct answers, questions can be seen in many ways

## SimpleQA vs TriviaQA

### Pros of SimpleQA
- Short Answers, better for evaluation
- Cleaner integration of calibration metrics
- newer and more up to date
### Cons of TriviaQA
- Consists more noise
- Formatting is off compared to other datasets used


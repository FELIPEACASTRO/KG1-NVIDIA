# V373 OpenRouter/Kaggle API audit - Discussion 690307

- source_url: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307
- title: Strategy to solve 85% of bit manipulation
- author: Tong Hui Kang
- postDate: 2026-04-11T07:43:56.656Z
- votes: 96
- commentCount: 11

## Extracted topic evidence
### keyword: 35/160
```
This is part of my
publication
for the Open Progress Prize.
I read the 0.73 scoring
notebook
from
@llkh0a
/ Kh0a.
The approach described in Kh0a's notebook is actually very similar to mine
Use code to write synthetic CoT traces
Train SFT on the synthetic CoT traces
Make the submission
Kh0a reports the following validation score.
Per-category:
  bit_manipulation: 35/160 = 21.88%
  gravity_physics: 160/160 = 100.00%
  numeral_system: 158/158 = 100.00%
  numeric_equation: 51/73 = 69.86%
  symbol_transform: 0/82 = 0.00%
  text_decryption: 145/158 = 91.77%
  unit_conversion: 159/159 = 100.00%
Overall: 708/950 = 74.53%
Weighted CV score: 74.76%
Kh0a's algorithm solves only 35/160 of bit manipulation problems.
I have an algorithm that solves 1364 of 1602 bit manipulation problems (85.1%).
85.1% of 160 is around 136. The additional 136 - 35 = 101 correct solutions will bring the overall score from 708/950 to 809/950 which is approximately 85%, which is the same as my winning submission score.
If Kh0a was actually able to perfectly train the model to generate exactly the chain of thought, Kh0a would have won the progress prize.
I describe my algorithm for bit manipulation here in a separate post.
I do not want my main post to have 50% bit manipulation co
```
### keyword: 1364
```
rts the following validation score.
Per-category:
  bit_manipulation: 35/160 = 21.88%
  gravity_physics: 160/160 = 100.00%
  numeral_system: 158/158 = 100.00%
  numeric_equation: 51/73 = 69.86%
  symbol_transform: 0/82 = 0.00%
  text_decryption: 145/158 = 91.77%
  unit_conversion: 159/159 = 100.00%
Overall: 708/950 = 74.53%
Weighted CV score: 74.76%
Kh0a's algorithm solves only 35/160 of bit manipulation problems.
I have an algorithm that solves 1364 of 1602 bit manipulation problems (85.1%).
85.1% of 160 is around 136. The additional 136 - 35 = 101 correct solutions will bring the overall score from 708/950 to 809/950 which is approximately 85%, which is the same as my winning submission score.
If Kh0a was actually able to perfectly train the model to generate exactly the chain of thought, Kh0a would have won the progress prize.
I describe my algorithm for bit manipulation here in a separate post.
I do not want my main post to have 50% bit manipulation content, even though it accounts for more than half of the difference with Kh0a's notebook.
This also allows me to elaborate more on the bit manipulation problem here.
This task asks to discover a per-bit transformation rule from input-output examples of 8-bit binary numbers.
I consider three possible transformations, each with seven possible values
ROT
(rotation)
SHR
(shift righ
```
### keyword: 85.1
```
category:
  bit_manipulation: 35/160 = 21.88%
  gravity_physics: 160/160 = 100.00%
  numeral_system: 158/158 = 100.00%
  numeric_equation: 51/73 = 69.86%
  symbol_transform: 0/82 = 0.00%
  text_decryption: 145/158 = 91.77%
  unit_conversion: 159/159 = 100.00%
Overall: 708/950 = 74.53%
Weighted CV score: 74.76%
Kh0a's algorithm solves only 35/160 of bit manipulation problems.
I have an algorithm that solves 1364 of 1602 bit manipulation problems (85.1%).
85.1% of 160 is around 136. The additional 136 - 35 = 101 correct solutions will bring the overall score from 708/950 to 809/950 which is approximately 85%, which is the same as my winning submission score.
If Kh0a was actually able to perfectly train the model to generate exactly the chain of thought, Kh0a would have won the progress prize.
I describe my algorithm for bit manipulation here in a separate post.
I do not want my main post to have 50% bit manipulation content, even though it accounts for more than half of the difference with Kh0a's notebook.
This also allows me to elaborate more on the bit manipulation problem here.
This task asks to discover a per-bit transformation rule from input-output examples of 8-bit binary numbers.
I consider three possible transformations, each with seven possible values
ROT
(rotation)
SHR
(shift right)
SHL
(shift left)
There are 7 + 7 + 7
```
### keyword: 21 possible
```
have 50% bit manipulation content, even though it accounts for more than half of the difference with Kh0a's notebook.
This also allows me to elaborate more on the bit manipulation problem here.
This task asks to discover a per-bit transformation rule from input-output examples of 8-bit binary numbers.
I consider three possible transformations, each with seven possible values
ROT
(rotation)
SHR
(shift right)
SHL
(shift left)
There are 7 + 7 + 7 = 21 possible transformations.
I consider six possible operations
AND
and
AND-NOT
OR
and
OR-NOT
XOR
and
XOR-NOT
I consider up to three transformations per expression
One-transformation:
ROT(4)
Two-transformation:
SHL(3) XOR NOT SHL(6)
Three-transformation:
(ROT(5) AND NOT SHR(4)) XOR NOT SHL(4)
In the training data alone, there are 622 expressions
One-transformation: 20
Two-transformation: 318
Three-transformation: 284
However, the 622 expressions do not cover all possible expressions.
Consider the following template
(ROT(X) AND NOT SHR(Y)) XOR NOT SHL(Z)
There are already 12,348 possible expressions for this template
7 possible values for
ROT(X)
6 possible operations for
ROT
and
SHR
7 possible values for
SHR(Y)
6 possible operations for
ROT
+
SHR
and
SHL
7 possible values for
SHL(Z)
You are only allowed 7680 tokens for your completion.
Even if you spend only one token testing one expressi
```
### keyword: six possible
```
though it accounts for more than half of the difference with Kh0a's notebook.
This also allows me to elaborate more on the bit manipulation problem here.
This task asks to discover a per-bit transformation rule from input-output examples of 8-bit binary numbers.
I consider three possible transformations, each with seven possible values
ROT
(rotation)
SHR
(shift right)
SHL
(shift left)
There are 7 + 7 + 7 = 21 possible transformations.
I consider six possible operations
AND
and
AND-NOT
OR
and
OR-NOT
XOR
and
XOR-NOT
I consider up to three transformations per expression
One-transformation:
ROT(4)
Two-transformation:
SHL(3) XOR NOT SHL(6)
Three-transformation:
(ROT(5) AND NOT SHR(4)) XOR NOT SHL(4)
In the training data alone, there are 622 expressions
One-transformation: 20
Two-transformation: 318
Three-transformation: 284
However, the 622 expressions do not cover all possible expressions.
Consider the following template
(ROT(X) AND NOT SHR(Y)) XOR NOT SHL(Z)
There are already 12,348 possible expressions for this template
7 possible values for
ROT(X)
6 possible operations for
ROT
and
SHR
7 possible values for
SHR(Y)
6 possible operations for
ROT
+
SHR
and
SHL
7 possible values for
SHL(Z)
You are only allowed 7680 tokens for your completion.
Even if you spend only one token testing one expression, you will run out of tokens.
Insight

```
### keyword: up to three
```
s also allows me to elaborate more on the bit manipulation problem here.
This task asks to discover a per-bit transformation rule from input-output examples of 8-bit binary numbers.
I consider three possible transformations, each with seven possible values
ROT
(rotation)
SHR
(shift right)
SHL
(shift left)
There are 7 + 7 + 7 = 21 possible transformations.
I consider six possible operations
AND
and
AND-NOT
OR
and
OR-NOT
XOR
and
XOR-NOT
I consider up to three transformations per expression
One-transformation:
ROT(4)
Two-transformation:
SHL(3) XOR NOT SHL(6)
Three-transformation:
(ROT(5) AND NOT SHR(4)) XOR NOT SHL(4)
In the training data alone, there are 622 expressions
One-transformation: 20
Two-transformation: 318
Three-transformation: 284
However, the 622 expressions do not cover all possible expressions.
Consider the following template
(ROT(X) AND NOT SHR(Y)) XOR NOT SHL(Z)
There are already 12,348 possible expressions for this template
7 possible values for
ROT(X)
6 possible operations for
ROT
and
SHR
7 possible values for
SHR(Y)
6 possible operations for
ROT
+
SHR
and
SHL
7 possible values for
SHL(Z)
You are only allowed 7680 tokens for your completion.
Even if you spend only one token testing one expression, you will run out of tokens.
Insight
I am still able to solve a significant majority of the three-transformation expre
```
### keyword: 354
```
nary operator, I will just default to answering with bit value 1.
Algorithm
I described that the number of expressions is too large to enumerate directly.
I need to test 18 possible unary combinations
8 possible positions
8 possible negated positions
2 possible constants
I need to test 336 possible binary combinations
8 possible positions for the first input
7 possible positions for the second input
6 possible operations
In total, I need to test 354 possible combinations.
I spend around 17 tokens to test each combination
2 tokens to denote the input bit positions
10 tokens for up to 10 possible example test cases. For symmetric operators like
AND
, I only print half the binary strings.
1 bitsum to make matching easier
3 spaces for formatting
1 newline
more if there is a match
The section looks like this
AND
01 10 1001000 2
12 21 1000000 1
23 32 0010000 1
34 43 0011000 2
45 54 0001100 2
56 65 0000100 1
67 76 0000100 1
70 07 1001010 3

02 20 1000000 1
13 31 0001000 1 match 5
24 42 0010000 1
35 53 0001010 2
46 64 0000100 1
57 75 1001110 4 match 0
60 06 0000001 1 match 1
71 17 1001100 3 match 2 6

03 30 0001010 2
14 41 0001100 2
25 52 1000000 1
36 63 0000000 a match 3 4 7
47 74 0001100 2
50 05 1001010 3
61 16 0000100 1
72 27 1000000 1

04 40 0001000 1 match 5
15 51 1001100 3 match 2 6
26 62 0000000 a match 3 4 7
37 73 0001010 2

Mat
```
### keyword: bitsum
```
ons
2 possible constants
I need to test 336 possible binary combinations
8 possible positions for the first input
7 possible positions for the second input
6 possible operations
In total, I need to test 354 possible combinations.
I spend around 17 tokens to test each combination
2 tokens to denote the input bit positions
10 tokens for up to 10 possible example test cases. For symmetric operators like
AND
, I only print half the binary strings.
1 bitsum to make matching easier
3 spaces for formatting
1 newline
more if there is a match
The section looks like this
AND
01 10 1001000 2
12 21 1000000 1
23 32 0010000 1
34 43 0011000 2
45 54 0001100 2
56 65 0000100 1
67 76 0000100 1
70 07 1001010 3

02 20 1000000 1
13 31 0001000 1 match 5
24 42 0010000 1
35 53 0001010 2
46 64 0000100 1
57 75 1001110 4 match 0
60 06 0000001 1 match 1
71 17 1001100 3 match 2 6

03 30 0001010 2
14 41 0001100 2
25 52 1000000 1
36 63 0000000 a match 3 4 7
47 74 0001100 2
50 05 1001010 3
61 16 0000100 1
72 27 1000000 1

04 40 0001000 1 match 5
15 51 1001100 3 match 2 6
26 62 0000000 a match 3 4 7
37 73 0001010 2

Matching output
0 57 75
1 60 06
2 71 17 15 51
3 36 63 26 62
4 36 63 26 62
5 13 31 04 40
6 71 17 15 51
7 36 63 26 62
Notice that I compute the bitsum instead of simply matching two binary strings that are far apart from each other.
The bitsum acts as
```
### keyword: longest stride
```
. Otherwise, return the whitespace token and continue calculating the next pair of indices. It seems that the model is able to reliably produce the
x
token, but the model could not reliably differentiate whether to produce the
y
token or the whitespace token.
The best match is the longest sequence. For tie-breaking, I simply choose the sequence that appears earlier.
After iterating over all the combinations, I choose the operations.
I choose the longest stride, starting from whichever side is having more matches.
To tie break, I have a priority order of operators.
Left longest: 3
Right longest: 3

Left winner: Identity no, NOT no, Constant no, AND yes, OR no, XOR no, AND-NOT no, OR-NOT no, XOR-NOT no
Right winner: Identity no, NOT no, Constant no, AND yes, OR no, XOR no, AND-NOT no, OR-NOT no, XOR-NOT no

Best left: AND57 60 71: 3
Best right: AND26 15 04: 3

Truncated left: AND57 60 71: 3
Truncated right: AND26 15 04: 3
If the left stride and the right stride do not fill up the full sequence, I fill in the middle with a stride-compliant sequence.
Preferred
0 AND57
1 AND60
2 AND71
3 ?02 ?20
4 ?13 ?31
5 AND04
6 AND15
7 AND26

Matching
0 AND57
1 AND60
2 AND71
3 ?02 ?20 - Identity absent, NOT absent, Constant C0, AND absent, OR absent, XOR absent, AND-NOT absent, OR-NOT absent, XOR-NOT absent
4 ?13 ?31 - Identity absent, NOT absent,
```
### keyword: fill in the middle
```
operators.
Left longest: 3
Right longest: 3

Left winner: Identity no, NOT no, Constant no, AND yes, OR no, XOR no, AND-NOT no, OR-NOT no, XOR-NOT no
Right winner: Identity no, NOT no, Constant no, AND yes, OR no, XOR no, AND-NOT no, OR-NOT no, XOR-NOT no

Best left: AND57 60 71: 3
Best right: AND26 15 04: 3

Truncated left: AND57 60 71: 3
Truncated right: AND26 15 04: 3
If the left stride and the right stride do not fill up the full sequence, I fill in the middle with a stride-compliant sequence.
Preferred
0 AND57
1 AND60
2 AND71
3 ?02 ?20
4 ?13 ?31
5 AND04
6 AND15
7 AND26

Matching
0 AND57
1 AND60
2 AND71
3 ?02 ?20 - Identity absent, NOT absent, Constant C0, AND absent, OR absent, XOR absent, AND-NOT absent, OR-NOT absent, XOR-NOT absent
4 ?13 ?31 - Identity absent, NOT absent, Constant C0, AND absent, OR absent, XOR absent, AND-NOT absent, OR-NOT absent, XOR-NOT absent
5 AND04
6 AND15
7 AND26

Perfect match
Identity no
NOT no
Constant yes
AND no
OR no
XOR no
AND-NOT no
OR-NOT no
XOR-NOT no
In this case, the constant value fits the middle section.
Then I construct the result.
Selected
0 AND57
1 AND60
2 AND71
3 C0
4 C0
5 AND04
6 AND15
7 AND26

Applying to 10001001
Input
0 1
1 0
2 0
3 0
4 1
5 0
6 0
7 1
Output
0 AND57 = AND(0,1) = 0
1 AND60 = AND(0,1) = 0
2 AND71 = AND(1,0) = 0
3 C0 = 0
4 C0 = 0
5 AND04 = AND(1,1) = 1
6 AND15 = A
```

## Comments evidence
### comment 3445372 by Taha votes=-3
```
Category
Found
Total
Accuracy
Avg ms
bit_manipulation
1584
1602
98.9%
7.7
cipher
1576
1576
100.0%
0.0
cryptarithm_deduce
98
659
14.9%
41.1
cryptarithm_guess
14
164
8.5%
39.8
equation_numeric_deduce
553
596
92.8%
0.9
equation_numeric_guess
21
136
15.4%
0.9
gravity
1597
1597
100.0%
0.0
numeral
1576
1576
100.0%
0.0
unit_conversion
1594
1594
100.0%
0.0
--------------------------
-------
-------
----------
--------
TOTAL
8613
9500
90.7%
33.0
Guess what?
```
### comment 3440047 by Giovanny Rodríguez votes=-1
```
It worked—thanks (It went up almost 20%.):```
 (.venv) dreuxx@dreuxx-HP-ZBook-Fury-15-6-inch-G8-Mobile-Workstation-PC:~/Documents/data$ python3 solver.py "train(7).csv"
Verifying 9500 samples from train(7).csv
bitwise             :   918/ 1602 ( 57.3%) [unsolvable: 0]
  cipher              :  1576/ 1576 (100.0%) [unsolvable: 0]
  physics             :  1597/ 1597 (100.0%) [unsolvable: 0]
  symbol_transform    :   168/ 1555 ( 10.8%) [unsolvable: 0]
  symbolic            :  1576/ 1576 (100.0%) [unsolvable: 0]
  unit_conversion     :  1594/ 1594 (100.0%) [unsolvable: 0]
Total verified: 7429/9500 (78.2%)
Generating curated dataset…
Saved 7429 solved rows to train_curated.csv.
(.venv) dreuxx@dreuxx-HP-ZBook-Fury-15-6-inch-G8-Mobile-Workstation-PC:~/Documents/data$ python3 solver.py "train(7).csv"
Verifying 9500 samples from train(7).csv
bitwise             :  1157/ 1602 ( 72.2%) [unsolvable: 0]
  cipher              :  1576/ 1576 (100.0%) [unsolvable: 0]
  physics             :  1597/ 1597 (100.0%) [unsolvable: 0]
  symbol_transform    :   166/ 1555 ( 10.7%) [unsolvable: 0]
  symbolic            :  1576/ 1576 (100.0%) [unsolvable: 0]
  unit_conversion     :  1594/ 1594 (100.0%) [unsolvable: 0]
```
```

## Actionable interpretation
- Confirmed by Kaggle API, not only screenshot/paste: bit gains come from deterministic bit relation search plus trace design, not generic more-epochs SFT.
- The solver side already surpassed Tong weak-scale target on our weak split (`V366 bit=159/160`), so the current bottleneck is transfer into adapter-only behavior.
- V372 is direct evidence that trace-style smoke still did not transfer: checkpoint-1 weak `191/315`, `equation=56`, `bit=135`, truncation `0`. FinOps gate rejects checkpoint-2/full/package/submit.
- New CPU work should be residual-only and focus on the remaining `1` bit miss plus equation DSL. New GPU work must be blocked until a transfer-specific change is proven, not just another SFT run.
# Snake
An AI that plays Snake by itself

## Example Test:
[Test 1] snake_head_x=5, snake_head_y=5, food_x=2, food_y=2, Ne=40, C=40, gamma=0.7
[Test 2] snake_head_x=5, snake_head_y=5, food_x=2, food_y=2, Ne=20, C=60, gamma=0.5
[Test 3] snake_head_x=3, snake_head_y=3, food_x=10, food_y=4, Ne=30, C=30, gamma=0.6

## Run:
To see the available parameters you can set for the game, run:

python mp6.py --help
To train and test your agent, run:

python mp6.py [parameters]
For example, to run Test 1 above, run:

python mp6.py --snake_head_x 5 --snake_head_y 5 --food_x 2 --food_y 2 --Ne 40 --C 40 --gamma 0.7

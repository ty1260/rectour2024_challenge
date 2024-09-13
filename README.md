# rectour2024_challenge

This repository contains the solution code that won 1st place in the RecTour 2024 Challenge by Team ringo.

## Approach
The solution consists of two main steps:
- Candidate generation: join `Users` and `Reviews` tables.
- Ranking: Predict the probability using LightGBM and get the top 10 candidates.

## Environment Setup
```
$ pip install -r requirements.txt
```

## Usage
```
$ chmod +x run.sh
$ ./run.sh
```

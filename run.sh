#!/bin/bash

notebooks=(
    "./notebook/dataset/load_dataset.ipynb"
    "./notebook/feature/01_reviews.ipynb"
    "./notebook/feature/02_accommodations.ipynb"
    "./notebook/feature/03_tfidf.ipynb"
)
script="./src/main.py"

# run Jupyter notebooks
for notebook in "${notebooks[@]}"
do
    echo "Running $notebook"
    jupyter nbconvert --execute "$notebook" --to notebook --inplace
done

# run Python script
echo "Running $script"
python3 "$script"

#!/bin/bash

# Define paths
FIBER_PATHS=("/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D1 drug fiber/D1 drug fiber - genotype 0"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D1 drug fiber/D1 drug fiber - genotype 1"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D1 vehicle fiber/D1 vehicle fiber - genotype 0"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D1 vehicle fiber/D1 vehicle fiber - genotype 1"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D2 drug fiber/D2 drug fiber - genotype 0"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D2 drug fiber/D2 drug fiber - genotype 1"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D2 vehicle fiber/D2 vehicle fiber - genotype 0"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D2 vehicle fiber/D2 vehicle fiber - genotype 1")
BEHAVIOR_PATHS=("/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D1 drug behavior/D1 drug behavior - genotype 0"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D1 drug behavior/D1 drug behavior - genotype 1"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D1 vehicle behavior/D1 vehicle behavior - genotype 0"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D1 vehicle behavior/D1 vehicle behavior - genotype 1"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D2 drug behavior/D2 drug behavior - genotype 0"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D2 drug behavior/D2 drug behavior - genotype 1"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D2 vehicle behavior/D2 vehicle behavior - genotype 0"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D2 vehicle behavior/D2 vehicle behavior - genotype 1")
OUTPUT_FOLDER_PATHS=("/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D1 drug out/genotype 0"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D1 drug out/genotype 1"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D1 vehicle out/genotype 0"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D1 vehicle out/genotype 1"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D2 drug out/genotype 0"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D2 drug out/genotype 1"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D2 vehicle out/genotype 0"
 "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/For Jake/D2 vehicle out/genotype 1")
MODEL_PATH="/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/epoch=34-step=314687.ckpt"

# Check to ensure the number of fiber, behavior, and output paths are the same
if [ ${#FIBER_PATHS[@]} -ne ${#BEHAVIOR_PATHS[@]} ] || [ ${#FIBER_PATHS[@]} -ne ${#OUTPUT_FOLDER_PATHS[@]} ]; then
    echo "The number of fiber, behavior, and output paths must be the same."
    exit 1
fi

# Define the constant model parameters
WINDOW_SIZE=1000
STRIDE=100
BATCH_SIZE=8

# Loop through the paths
for i in "${!FIBER_PATHS[@]}"; do
    # Run the Python script with the current combination of paths and constant model parameters
    python latent_generator_1.py "${FIBER_PATHS[$i]}" "${BEHAVIOR_PATHS[$i]}" "${OUTPUT_FOLDER_PATHS[$i]}" "$MODEL_PATH" \
    --window_size "$WINDOW_SIZE" --stride "$STRIDE" --batch_size "$BATCH_SIZE" --pca --plot --pca_components=4
done

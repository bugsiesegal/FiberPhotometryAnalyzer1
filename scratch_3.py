import os
import glob
from latent_generator_2 import *

basepath = "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/test data"

if os.path.exists(basepath + "/inference_results.pkl"):
    print("Loading inference results")
    data = pd.read_pickle(basepath + "/inference_results.pkl")
else:
    print("Performing inference")
    data = get_data(glob.glob(os.path.join(basepath, "*/*/*_evts.txt")) +
                    glob.glob(os.path.join(basepath, "*/*/Animal*")))

    model = load_model("/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/model.ckpt").cuda()

    data = add_sliding_windows(data, window_size=1000, step_size=100)

    behaviors = []
    for idx, animal_row in data.iterrows():
        if animal_row["behavior_or_fiber"] == "behavior":
            behaviors += animal_row["data"]["TrackName"].tolist()

    behaviors = list(set(behaviors))

    print(behaviors)

    data = perform_inference_and_update_df(data, model, batch_size=32)

    # Save inference results
    data.to_pickle(basepath + "/inference_results.pkl")

cebra = get_cebra(data, n_components=4, max_iterations=10_000, batch_size=128, learning_rate=1e-7)

calculate_centroid_distance(perform_segmentation(data), ["None"],
                            "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/outputs/centroid_distance.csv",
                            "/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/outputs/centroids.csv")

data = perform_cebra_on_inference_results(data, cebra)

data = perform_segmentation(data, "cebra_results")

for idx, animal_row in data.iterrows():
    if animal_row["behavior_or_fiber"] == "fiber":
        base_output_path = f"/home/bugsie/PycharmProjects/FiberPhotometryAnalyzer/outputs/" \
                           f"{animal_row['animal']}-{animal_row['drug_or_vehicle']}-{animal_row['pathway']}-" \
                           f"genotype_{animal_row['genotype']}"

        os.makedirs(base_output_path, exist_ok=True)

        plot_combinations_of_components_of_animal(data, animal_row["animal"], num_components=4,
                                                  exclude_behaviors=["None"], save_path=os.path.join(base_output_path,
                                                                                                     "scatter.png"))

        plot_segmented_time_series(animal_row["segmentation_results"],
                                   os.path.join(base_output_path, 'plot.png'))

        plot_density_based_combinations(animal_row["segmentation_results"], ["None"],
                                        save_path=os.path.join(base_output_path,
                                                               "density.png"))

        plot_autoencoder_results(animal_row["sliding_windows"], animal_row["reconstructed_results"],
                                 save_path=os.path.join(base_output_path,
                                                        "autoencoder.png"))

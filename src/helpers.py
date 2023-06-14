import numpy as np


def print_box(message: str) -> None:
    print("{:-^50}".format(""))
    print("{:-^50}".format("  " + message + "  "))
    print("{:-^50}".format(""))
    print()

def generate_bootstrap_samples(data: np.ndarray, bootstrap_name: str, n_samples: int):
    """Generates bootstramp samples from the provided data in the INPUT/data folder.
    Places them in the INPUT/data/bootstrap_name folder.
    """
    print("Generating bootstrap samples...")
    samples = []
    for i in range(n_samples):
        bootstrap_sample = data[np.random.randint(0, data.shape[0])]
        samples.append(bootstrap_sample)
    all_zeros = np.zeros(len(samples[0]), dtype=int)
    all_ones = np.ones(len(samples[0]), dtype=int)
    samples.append(all_zeros)
    samples.append(all_ones)
    np.savetxt(
        "INPUT/data/{}.dat".format(bootstrap_name),
        samples,
        fmt="%d",
        delimiter="",
    )
    print("Done!")
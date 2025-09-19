ALL_WORKS = ["D911-{:02d}".format(x) for x in range(1, 24+1)]
ALL_VERSIONS = ["HU33", "SC06", "AL98", "FI55", "FI66", "FI80", "OL06", "QU98", "TR99"]

PREFIX = "Schubert"

def get_split_list():
    train_versions = ["HU33", "SC06", "QU98", "FI55"]
    val_versions = ["FI66", "FI80"]
    test_versions = ["OL06", "AL98", "TR99"]

    train_works = ALL_WORKS[:16] + ALL_WORKS[18:19]
    val_works = ALL_WORKS[20:22]
    test_works = ALL_WORKS[16:18] + ALL_WORKS[19:20] + ALL_WORKS[22:]

    train_list = [f"{PREFIX}_{work}_{version}" for version in train_versions for work in train_works]
    val_list = [f"{PREFIX}_{work}_{version}" for version in val_versions for work in val_works]
    test_list = [f"{PREFIX}_{work}_{version}" for version in test_versions for work in test_works]

    return train_list, val_list, test_list

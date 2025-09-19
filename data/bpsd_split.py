ALL_WORKS = ["Op002No1-01", "Op002No2-01", "Op002No3-01", "Op007-01", "Op010No1-01", "Op010No2-01", "Op010No3-01",
             "Op013-01", "Op014No1-01", "Op014No2-01", "Op022-01", "Op026-01", "Op027No1-01", "Op027No2-01",
             "Op028-01", "Op031No1-01", "Op031No2-01", "Op031No3-01", "Op049No1-01", "Op049No2-01", "Op053-01",
             "Op054-01", "Op057-01", "Op078-01", "Op079-01", "Op081a-01", "Op090-01", "Op101-01", "Op106-01",
             "Op109-01", "Op110-01", "Op111-01"]
ALL_VERSIONS = ["WK64", "FJ62", "FG58", "AS35", "MC22", "MB97", "AB96", "JJ90", "DB84", "VA81", "FG67"]
PREFIX = "Beethoven"

def get_split_list():

    train_versions = ["JJ90", "DB84", "VA81", "FG67", "AB96"]
    val_versions = ["MB97", "MC22"]
    test_versions = ["WK64", "FG58", "FJ62", "AS35"]

    train_works = ["Op002No1-01", "Op002No2-01", "Op002No3-01", "Op007-01", "Op010No1-01", "Op010No2-01",
                    "Op010No3-01", "Op013-01", "Op014No1-01", "Op014No2-01", "Op022-01", "Op026-01",
                    "Op027No1-01", "Op027No2-01", "Op028-01", "Op031No1-01", "Op031No2-01", "Op031No3-01",
                    "Op049No1-01", "Op049No2-01"]
    val_works = ["Op053-01", "Op054-01", "Op057-01", "Op078-01"]
    test_works = ["Op079-01", "Op081a-01", "Op090-01", "Op101-01", "Op106-01", "Op109-01", "Op110-01", "Op111-01"]

    train_list = [f"{PREFIX}_{work}_{version}" for version in train_versions for work in train_works]
    val_list = [f"{PREFIX}_{work}_{version}" for version in val_versions for work in val_works]
    test_list = [f"{PREFIX}_{work}_{version}" for version in test_versions for work in test_works]

    return train_list, val_list, test_list

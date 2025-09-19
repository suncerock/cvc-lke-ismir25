ALL_WORKS = [
    ["Op018No1-01", "Op018No1-02", "Op018No1-03", "Op018No1-04"],
    ["Op018No2-01", "Op018No2-02", "Op018No2-03", "Op018No2-04"],
    ["Op018No3-01", "Op018No3-02", "Op018No3-03", "Op018No3-04"],
    ["Op018No4-01", "Op018No4-02", "Op018No4-03", "Op018No4-04"],
    ["Op018No5-01", "Op018No5-02", "Op018No5-03", "Op018No5-04"],
    ["Op018No6-01", "Op018No6-02", "Op018No6-03", "Op018No6-04"],
    ["Op059No1-01", "Op059No1-02", "Op059No1-03", "Op059No1-04"],
    ["Op059No2-01", "Op059No2-02", "Op059No2-03", "Op059No2-04"],
    ["Op059No3-01", "Op059No3-02", "Op059No3-03", "Op059No3-04"],
    ["Op074-01", "Op074-02", "Op074-03", "Op074-04"],
    ["Op095-01", "Op095-02", "Op095-03", "Op095-04"],
    ["Op127-01", "Op127-02", "Op127-03", "Op127-04"],
    ["Op130-01", "Op130-02", "Op130-03", "Op130-04", "Op130-05", "Op130-06"],
    ["Op131-01", "Op131-02", "Op131-03", "Op131-04", "Op131-05", "Op131-06", "Op131-07"],
    ["Op132-01", "Op132-02", "Op132-03", "Op132-04", "Op132-05"],
    ["Op135-01", "Op135-02", "Op135-03", "Op135-04"],
]
ALL_VERSIONS = [
    "AlbanBergQuartet_WC",
    "AmadeusQuartet_DG",
    "BudapestStringQuartet_SONY",
    "EmersonStringQuartet_DG",
    "GuarneriQuartet_BC",  # 42 works - Op059, Op074, Op127, Op130, Op131, Op132, Op135
    "SharonQuartet_BC",  # 4 works - Op095
    "SuskeQuartett_BC",
    "TokyoQuartet_SONY",
    "VeghQuartet_TIM",
]
ALL_COMPLETE_VERSIONS = [
    "AlbanBergQuartet_WC",
    "AmadeusQuartet_DG",
    "BudapestStringQuartet_SONY",
    "EmersonStringQuartet_DG",
    "SuskeQuartett_BC",
    "TokyoQuartet_SONY",
    "VeghQuartet_TIM",
]

PREFIX = "Beethoven"

def get_split_list():
    train_versions = ["AlbanBergQuartet_WC", "AmadeusQuartet_DG"] # , "GuarneriQuartet_BC", "SharonQuartet_BC"
    val_versions = ["BudapestStringQuartet_SONY", "VeghQuartet_TIM"]
    test_versions = ["SuskeQuartett_BC", "TokyoQuartet_SONY", "EmersonStringQuartet_DG"]

    train_works = sum(ALL_WORKS[:4] + ALL_WORKS[12:14]+ ALL_WORKS[9:11], start=[]) + ALL_WORKS[6]
    val_works = ALL_WORKS[4] + ALL_WORKS[7] + ALL_WORKS[14]
    test_works = ALL_WORKS[5] + ALL_WORKS[8] + ALL_WORKS[11] + ALL_WORKS[15]

    train_list = [f"{PREFIX}_{work}_{version}" for version in train_versions for work in train_works]
    val_list = [f"{PREFIX}_{work}_{version}" for version in val_versions for work in val_works]
    test_list = [f"{PREFIX}_{work}_{version}" for version in test_versions for work in test_works]

    for work in ALL_WORKS[6] + ALL_WORKS[9] + ALL_WORKS[12] + ALL_WORKS[13]:
        train_list.append(f"{PREFIX}_{work}_GuarneriQuartet_BC")
    for work in ALL_WORKS[10]:
        train_list.append(f"{PREFIX}_{work}_SharonQuartet_BC")
    
    return train_list, val_list, test_list

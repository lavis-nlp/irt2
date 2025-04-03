import irt2
from irt2.loader import LOADER

p_data = irt2.ENV.DIR.DATA


config = {
    "datasets": [
        {
            "path": p_data / "irt2" / "irt2-cde-tiny",
            "loader": "irt2",
            "percentage": {
                "validation": 0.17,
                "test": 0.02,
            },
        },
        {
            "path": p_data / "irt2" / "irt2-cde-small",
            "loader": "irt2",
            "percentage": {
                "validation": 0.08,
                "test": 0.02,
            },
        },
        {
            "path": p_data / "irt2" / "irt2-cde-medium",
            "loader": "irt2",
            "percentage": {
                "validation": 0.04,
                "test": 0.01,
            },
        },
        {
            "path": p_data / "irt2" / "irt2-cde-large",
            "loader": "irt2",
            "percentage": {
                "validation": 0.05,
                "test": 0.02,
            },
        },
        {
            "path": p_data / "blp" / "WN18RR",
            "loader": "blp/wn18rr",
            "percentage": {
                "validation": 0.06,
                "test": 0.06,
            },
        },
        {
            "path": p_data / "blp" / "FB15k-237",
            "loader": "blp/fb15k237",
            "percentage": {
                "validation": 0.03,
                "test": 0.03,
            },
        },
        {
            "path": p_data / "blp" / "Wikidata5M",
            "loader": "blp/wikidata5m",
            "percentage": {
                "validation": 0.09,
                "test": 0.08,
            },
        },
    ],
    "splits": [
        "validation",
        "test",
    ],
    "seed": 31189,
}


idx = 23  # where test cells start


for dataset_config in config["datasets"]:
    ds = LOADER[dataset_config["loader"]](dataset_config["path"])

    rows = {}
    for split in config["splits"]:
        percentage = dataset_config["percentage"][split]
        sub_ds = ds.tasks_subsample_kgc(percentage, seed=config["seed"])
        rows[split] = sub_ds.table_row

    row = rows["validation"][:idx] + rows["test"][idx:]
    print(", ".join(map(str, row)))

import json
import random

import irt2
import yaml
from irt2.dataset import IRT2
from irt2.loader import from_config_file
from irt2.types import Split
from rich.progress import track


def create_prompts_with_examples(datasets: dict[str, IRT2]):
    data = []

    rng = random.Random(31189)
    k = 3

    for dataset in datasets.values():
        print(f"{dataset}")

        conf = dict(
            datasets=[dataset.name],
            prompts=dict(
                head={},
                tail={},
            ),
        )

        for rid, raw in track(list(dataset.relations.items())):
            rel = rel_head = rel_tail = raw.split(":")[1]

            if dataset.name == "BLP/FB15K237" and "." in rel:
                rel_tail, rel_head = map(str.strip, rel.split("."))

            candidates = [(h, t) for h, t, r in dataset.closed_triples if r == rid]
            head_vids, tail_vids = zip(*candidates)

            head_mentions = {
                dataset.idmap.mid2str[mid]
                for vid in head_vids
                for mid in dataset.idmap.vid2mids[Split.train][vid]
            }

            tail_mentions = {
                dataset.idmap.mid2str[mid]
                for vid in tail_vids
                for mid in dataset.idmap.vid2mids[Split.train][vid]
            }

            if len(head_mentions) > k:
                head_candidates = rng.sample(sorted(head_mentions), k=k)
            else:
                head_candidates = list(head_mentions)

            if len(tail_mentions) > k:
                tail_candidates = rng.sample(sorted(tail_mentions), k=k)
            else:
                tail_candidates = list(tail_mentions)

            tail_prompt = f"head={{mention}}, relation={rel_head.lower()}. "
            if len(tail_candidates):
                tail_prompt += "For example, a possible answer would be "
                tail_prompt += json.dumps({"answer": tail_candidates})

            head_prompt = f"tail={{mention}}, relation={rel_tail.lower()}. "
            if len(head_candidates):
                head_prompt += "For example, a possible answer would be "
                head_prompt += json.dumps({"answer": head_candidates})

            conf["prompts"]["head"][rel] = head_prompt
            conf["prompts"]["tail"][rel] = tail_prompt

        data.append(conf)

    with open("prompts.yaml", mode="w") as fd:
        fd.write(yaml.safe_dump(data))


if __name__ == "__main__":
    datasets = from_config_file(
        irt2.ENV.DIR.CONF / "datasets" / "original.yaml",
        # only=["blp/fb15k237"],
    )

    create_prompts_with_examples(dict(datasets))

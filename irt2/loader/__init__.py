from irt2.loader import blp, irt2

LOADER = {
    "irt2": irt2.load_irt2,
    "blp/umls": blp.load_umls,
    "blp/wn18rr": blp.load_wn18rr,
    "blp/fb15k237": blp.load_fb15k237,
    "blp/wikidata5m": blp.load_wikidata5m,
}

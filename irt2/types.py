# -*- coding: utf-8 -*-

"""Project wide type definitions."""

VID = int  # vertex id
MID = int  # mention id
RID = int  # relation id
EID = str  # upstream entity id (e.g. Wikidata ID for CodEx)

Mention = str
Triple = tuple[VID, VID, RID]

"""Tests for model-name matching against the real hardware CSVs.

Run with:  python -m pytest tests
"""

import pandas as pd
import pytest

from pc_build_assistant.config import BASE_DIR, PRICE_PATH
from pc_build_assistant.data import find_price_row
from pc_build_assistant.matching import ModelIndex, same_model, tokenize

GPUS = ModelIndex(pd.read_csv(BASE_DIR / "all_gpus.csv")["Model"])
CPUS = ModelIndex(pd.read_csv(BASE_DIR / "cpus.csv")["Model"])
ALL = ModelIndex(GPUS.models + CPUS.models)


@pytest.mark.parametrize("query, expected", [
    # A shorter name must not resolve to a longer model that contains it.
    ("Ryzen 9 9950X", "Ryzen 9 9950X"),
    ("Ryzen 9 7950X", "Ryzen 9 7950X"),
    ("Ryzen 7 7700", "Ryzen 7 7700"),
    ("Ryzen 5 7600", "Ryzen 5 7600"),
    ("Ryzen 5 5600", "Ryzen 5 5600"),
    ("Ryzen 9 9950X3D", "Ryzen 9 9950X3D"),
    # Brand prefixes, missing family words, spacing and punctuation.
    ("AMD Ryzen 7 9800X3D", "Ryzen 7 9800X3D"),
    ("9800X3D", "Ryzen 7 9800X3D"),
    ("Intel Core i7-14700K", "Core i7-14700K"),
    ("i7 14700K", "Core i7-14700K"),
    ("14700k", "Core i7-14700K"),
    ("250K", "Core Ultra 5 250K Plus"),
])
def test_resolve_cpu(query, expected):
    assert CPUS.resolve(query).best == expected


@pytest.mark.parametrize("query, expected", [
    ("RTX 5070", "RTX 5070"),
    ("RTX 5070 Ti", "RTX 5070 Ti"),
    ("RTX 5070Ti", "RTX 5070 Ti"),
    ("rtx-5070-ti", "RTX 5070 Ti"),
    ("GeForce RTX 5080", "RTX 5080"),
    ("NVIDIA GeForce RTX 5080 Founders Edition", "RTX 5080"),
    ("RTX 4060", "RTX 4060"),
    ("RTX 4070 Super", "RTX 4070 Super"),
    ("RTX 4070 Ti Super", "RTX 4070 Ti Super"),
    ("Radeon RX 9070 XT", "RX 9070 XT"),
    ("RX 9070", "RX 9070"),
    ("9070 XT", "RX 9070 XT"),
    ("RX 7600", "RX 7600"),
    ("RX 7900 XTX", "RX 7900 XTX"),
])
def test_resolve_gpu(query, expected):
    assert GPUS.resolve(query).best == expected


def test_ambiguous_name_lists_every_variant():
    mention = GPUS.resolve("RTX 4060 Ti")
    assert not mention.exact
    assert set(mention.models) == {"RTX 4060 Ti 8GB", "RTX 4060 Ti 16GB"}


@pytest.mark.parametrize("query", [
    "Ryzen 7",            # family only, no model number
    "RTX",
    "Ryzen 5 7600X3D",    # doesn't exist; must not fall back to the 7600X
    "RTX 6090",
    "best GPU under $600 for 4K",
])
def test_unknown_names_do_not_match(query):
    assert ALL.resolve(query) is None


@pytest.mark.parametrize("query, expected", [
    ("Compare the RTX 5080 and RX 9070 XT for 4K gaming",
     ["RTX 5080", "RX 9070 XT"]),
    ("Ryzen 7 9800X3D vs Core i7-14700K for gaming",
     ["Ryzen 7 9800X3D", "Core i7-14700K"]),
    ("Will a Ryzen 5 5600X bottleneck an RTX 5080?",
     ["Ryzen 5 5600X", "RTX 5080"]),
    ("RTX 5080 specs", ["RTX 5080"]),
    ("Ryzen 5 7600 vs RX 7600", ["Ryzen 5 7600", "RX 7600"]),
])
def test_find_all_in_free_text(query, expected):
    assert [m.best for m in ALL.find_all(query)] == expected


def test_tokenize_and_same_model():
    assert tokenize("Core i7-14700K") == ("core", "i", "7", "14700", "k")
    assert same_model("RTX 5070Ti", "rtx 5070 ti")
    assert not same_model("Ryzen 9 9950X", "Ryzen 9 9950X3D")


@pytest.mark.parametrize("model, has_price", [
    ("Ryzen 9 9950X", True),
    ("Ryzen 9 9950X3D", True),
    ("Ryzen 7 7700X", True),
    ("Ryzen 7 7700", False),   # must not borrow the 7700X's price
    ("RX 7600", False),
])
def test_price_lookup_is_exact(model, has_price):
    row = find_price_row(pd.read_csv(PRICE_PATH), model)
    assert (row is not None) == has_price
    if row is not None:
        assert same_model(row["Model"], model)

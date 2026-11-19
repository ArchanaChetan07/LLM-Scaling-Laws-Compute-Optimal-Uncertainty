"""Parameter counts must reproduce the published tables exactly."""
import pytest

from scalingci.data import RAW
from scalingci.params import from_config_file

# Biderman et al. (2023) Table 1: (total, non-embedding)
PYTHIA = {"70m": (70_426_624, 18_915_328), "160m": (162_322_944, 85_056_000),
          "410m": (405_334_016, 302_311_424), "1.4b": (1_414_647_808, 1_208_602_624),
          "2.8b": (2_775_208_960, 2_517_652_480), "6.9b": (6_857_302_016, 6_444_163_072),
          "12b": (11_846_072_320, 11_327_027_200)}
# DataDecide model card "Model size" = non-embedding + input embedding, in millions.
DATADECIDE = {"4M": 3.7, "6M": 6.0, "8M": 8.5, "10M": 9.9, "14M": 14.4, "16M": 16.0,
              "20M": 19.1, "60M": 57.1, "90M": 97.9, "150M": 151.9, "300M": 320.0,
              "530M": 530.1, "750M": 681.3, "1B": 1176.8}


@pytest.mark.parametrize("size", PYTHIA)
def test_pythia(size):
    pc = from_config_file(RAW / "configs" / f"pythia_{size}.json")
    assert (pc.total, pc.nonembed) == PYTHIA[size]


@pytest.mark.parametrize("size", DATADECIDE)
def test_datadecide(size):
    pc = from_config_file(RAW / "configs" / f"dd_{size}.json")
    reported = (pc.nonembed + pc.embed / 2) / 1e6
    assert reported == pytest.approx(DATADECIDE[size], abs=0.051)

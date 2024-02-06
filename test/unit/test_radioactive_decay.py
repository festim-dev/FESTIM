from festim import RadioactiveDecay
import pytest


def test_init():
    rd = RadioactiveDecay(0.5, 100)
    assert rd.decay_constant == 0.5
    assert rd.volume == 100


def test_decay_constant_setter():
    rd = RadioactiveDecay(0.5, 100)
    rd.decay_constant = 0.7
    assert rd.decay_constant == 0.7


def test_decay_constant_setter_invalid_type():
    rd = RadioactiveDecay(0.5, 100)
    with pytest.raises(TypeError):
        rd.decay_constant = "invalid"


def test_decay_constant_setter_negative_value():
    rd = RadioactiveDecay(0.5, 100)
    with pytest.raises(ValueError):
        rd.decay_constant = -0.5


def test_form():
    rd = RadioactiveDecay(0.5, 100)
    assert rd.form(200) == -100

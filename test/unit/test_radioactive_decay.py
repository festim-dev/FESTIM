from festim import RadioactiveDecay
import pytest


class TestRadioactiveDecay:
    """
    General test for the festim.RadioactiveDecay class
    """

    rd = RadioactiveDecay(0.5, 100)

    def test_init(self):
        assert self.rd.decay_constant == 0.5
        assert self.rd.volume == 100

    def test_decay_constant_setter(self):
        self.rd.decay_constant = 0.7
        assert self.rd.decay_constant == 0.7

    def test_decay_constant_setter_invalid_type(self):
        with pytest.raises(TypeError):
            self.rd.decay_constant = "invalid"

    def test_decay_constant_setter_negative_value(self):
        with pytest.raises(ValueError):
            self.rd.decay_constant = -0.5

    def test_form(self):
        self.rd.decay_constant = 0.5
        assert self.rd.form(200) == -100

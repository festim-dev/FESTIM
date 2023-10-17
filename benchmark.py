from test.test_permeation_problem import test_permeation_problem


def test_festim_benchmark(benchmark):
    benchmark(test_permeation_problem)

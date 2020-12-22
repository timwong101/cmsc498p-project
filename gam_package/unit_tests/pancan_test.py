import pytest

from gam_package.gam import GAM

class TestClass:

    def test_gam_parallel_pancan_runtime(self):
        local_attribution_path = '../data/pancan_clean.csv'
        g = GAM(attributions_path=local_attribution_path, k=3, cluster_method="parallel medoids")
        g.generate()
        print("Parallel Medoids Runtime: ", g.duration)

    def test_gam_bandit_pancan_runtime_700_samples(self):
        local_attribution_path = '../data/pancan_clean.csv'
        g = GAM(attributions_path=local_attribution_path, k=3, cluster_method="banditpam", num_samp=700)
        g.generate()
        print("BanditPAM 700 Runtime: ", g.duration)


import pytest

from gam_package.gam import GAM

class TestClass:

    def test_gam_bandit_crime_runtime_700_samples(self):
        local_attribution_path = '../data/crime.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="bandit pam", num_samp=700, dataset='crime')
        g.generate()
        print("BanditPAM 500 Runtime: ", g.duration)
        print("---------------------------------------")
    def test_gam_spectral_crime_runtime_1000_samples(self):
        local_attribution_path = '../data/crime.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="spectral", num_samp=1000, dataset='crime')
        g.generate()
        print("Spectral Runtime: ", g.duration)
        print("---------------------------------------")

    def test_gam_nystrom_crime_runtime_1000_samples(self):
        local_attribution_path = '../data/crime.csv'
        g = GAM(attributions_path=local_attribution_path, n_clusters=3, cluster_method="kernel medoids", num_samp=1000, dataset='crime')
        g.generate()
        print("Kernel Nystrom Runtime: ", g.duration)
        print("---------------------------------------")


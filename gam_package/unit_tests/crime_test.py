import pytest

from gam_package.gam import GAM

class TestClass:
    """
    def test_gam_parallel_crime_runtime(self):
        local_attribution_path = '../data/crime-communities-formatted.csv'
        g = GAM(attributions_path=local_attribution_path, k=3, cluster_method="parallel medoids")
        g.generate()
        print("Parallel Medoids Runtime: ", g.duration)

    """
    def test_gam_ranked_crime_runtime(self):
        local_attribution_path = '../data/crime-communities-formatted.csv'
        g = GAM(attributions_path=local_attribution_path, k=3, cluster_method="ranked medoids")
        g.generate()
        print("Ranked Medoids Runtime: ", g.duration)
    """
    def test_gam_bandit_crime_runtime_200_samples(self):
        local_attribution_path = '../data/crime-communities-formatted.csv'
        g = GAM(attributions_path=local_attribution_path, k=3, cluster_method="banditpam", num_samp=200)
        g.generate()
        print("BanditPAM 200 Runtime: ", g.duration)

    def test_gam_bandit_crime_runtime_700_samples(self):
        local_attribution_path = '../data/crime-communities-formatted.csv'
        g = GAM(attributions_path=local_attribution_path, k=3, cluster_method="banditpam", num_samp=700)
        g.generate()
        print("BanditPAM 700 Runtime: ", g.duration)
    """
    def test_gam_parallel_crime_bestk(self):
        local_attribution_path = '../data/Data_Cortex_Nuclear.csv'
        bestClusterNumber = 0
        bestScore = -2
        for k in range(2,5):
            g = GAM(attributions_path=local_attribution_path, k=k, cluster_method="parallel medoids")
            g.generate()
            if g.avg_silhouette_score > bestScore:
                bestScore = g.avg_silhouette_score
                bestClusterNumber = k
        print("Best Number of Clusters: ", bestClusterNumber)
        print("Best Silhouette Score:, ", bestScore)
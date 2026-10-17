import unittest

import numpy as np

from evaluation.proposal_alignment_audit import pseudo_quality, paired_mmd, balanced_indices
from evaluation.proposal_latent_knn_audit import agreements


class AlignmentAuditTests(unittest.TestCase):
    def test_pseudo_precision_and_coverage_have_distinct_denominators(self):
        p = np.array([[.05,.95],[.08,.92],[.95,.05],[.4,.6]])
        result = pseudo_quality(p,np.array([1,0,0,1]),.9)
        self.assertEqual(result['accepted'],3)
        self.assertEqual(result['accepted_attack'],2)
        self.assertEqual(result['attack_precision'],.5)
        self.assertEqual(result['attack_recall'],.5)
        self.assertAlmostEqual(result['accepted_accuracy'],2/3)

    def test_no_accepted_attack_has_undefined_precision(self):
        result = pseudo_quality(np.array([[.95,.05],[.6,.4]]),np.array([0,1]),.9)
        self.assertIsNone(result['attack_precision'])
        self.assertEqual(result['attack_recall'],0)

    def test_identical_models_have_zero_paired_mmd_change(self):
        rng=np.random.default_rng(4)
        s=rng.normal(size=(20,3)).astype('float32')
        t=rng.normal(size=(25,3)).astype('float32')
        result=paired_mmd(s,t,s,t,10,3,42)
        self.assertAlmostEqual(result['paired_delta'],0,places=8)

    def test_balancing_and_self_exclusion(self):
        labels=np.array([0]*8+[1]*3)
        indices=balanced_indices(labels,5,np.random.default_rng(42))
        self.assertEqual(np.bincount(labels[indices]).tolist(),[3,3])
        x=np.array([[0.],[.01],[10.],[10.01]])
        values=agreements(x,x,np.array([0,0,1,1]),np.array([0,0,1,1]),[1],'euclidean',True)[1]
        np.testing.assert_array_equal(values,np.ones(4))


if __name__ == '__main__':
    unittest.main()

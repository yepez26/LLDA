import unittest
import numpy as np
from modelagem_simples import class_spread

class DescriptiveTests(unittest.TestCase):
    def test_coverage_and_multilabel(self):
        assignments=np.array([0]*5+[1]*3+[2]*2)
        Y=np.ones((10,2),dtype=int)
        counts,required=class_spread(assignments,Y,3)
        np.testing.assert_array_equal(counts,[[5,5],[3,3],[2,2]])
        np.testing.assert_array_equal(required,[2,2])
        self.assertEqual(class_spread(assignments,Y,3,coverage=0.5)[1][0],1)
        self.assertEqual(class_spread(assignments,Y,3,coverage=1)[1][0],3)
    def test_empty_class_and_topic(self):
        counts,required=class_spread(np.array([1,1]),np.array([[1,0],[1,0]]),3)
        np.testing.assert_array_equal(required,[1,0])
        self.assertEqual(counts[0].sum(),0)
        with self.assertRaises(ValueError):class_spread([0],np.ones((1,1)),1,coverage=0)
if __name__=='__main__':unittest.main()

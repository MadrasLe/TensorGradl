import numpy as np
import unittest
from minigrad.tensor import Tensor

class TestTensor(unittest.TestCase):
    def test_add(self):
        t1 = Tensor([1.0, 2.0, 3.0])
        t2 = Tensor([4.0, 5.0, 6.0])
        t3 = t1 + t2
        t3.backward()
        
        np.testing.assert_array_equal(t3.data, [5.0, 7.0, 9.0])
        np.testing.assert_array_equal(t1.grad, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(t2.grad, [1.0, 1.0, 1.0])

    def test_mul(self):
        t1 = Tensor([2.0])
        t2 = Tensor([3.0])
        t3 = t1 * t2
        t3.backward()
        
        self.assertEqual(t3.data[0], 6.0)
        self.assertEqual(t1.grad[0], 3.0)
        self.assertEqual(t2.grad[0], 2.0)
        
    def test_matmul(self):
        # y = x @ w
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        w = Tensor([[1.0, 0.0], [0.0, 1.0]]) # Identity
        y = x.matmul(w)
        
        # To test gradients easily, let's sum y to get a scalar, 
        # so implicit gradient is 1.0 everywhere
        z = y.sum() 
        z.backward()
        
        # z = sum(x_ij * w_jk)
        # dy/dx is w.T broadcasted essentially.
        # Since w is Identity, and we sum output, grad w.r.t input is all ones.
        
        np.testing.assert_array_equal(y.data, x.data)
        np.testing.assert_array_equal(x.grad, np.ones_like(x.data))

    def test_broadcasting(self):
        t1 = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t2 = Tensor([10.0, 20.0, 30.0])
        t3 = t1 + t2
        t3.sum().backward() # Reduce to scalar to call backward easily
        
        np.testing.assert_array_equal(t3.data, [[11., 22., 33.], [14., 25., 36.]])
        np.testing.assert_array_equal(t1.grad, np.ones_like(t1.data))
        # t2 grad should be [2, 2, 2] because it was added to two rows
        np.testing.assert_array_equal(t2.grad, [2.0, 2.0, 2.0])

if __name__ == '__main__':
    unittest.main()

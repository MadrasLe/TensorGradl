import numpy as np
import unittest
from minigrad.tensor import Tensor
from minigrad.loss import cross_entropy

class TestLoss(unittest.TestCase):
    def test_cross_entropy_gradient(self):
        # Simple case: 1 sample, 2 classes
        # Logits: [0, 0] -> Softmax [0.5, 0.5]
        # Target: 0
        logits_data = np.array([[[0.0, 0.0]]]) # (B=1, T=1, V=2)
        logits = Tensor(logits_data)
        target = Tensor(np.array([[0]])) # (B=1, T=1)
        
        loss = cross_entropy(logits, target)
        loss.backward()
        
        # Check Loss Value
        expected_loss = -np.log(0.5)
        self.assertAlmostEqual(loss.data, expected_loss, places=4)
        
        # Check Gradients
        # dLoss/dLogit_i = p_i - y_i
        # p = [0.5, 0.5], y = [1, 0] (one-hot)
        # grad = [0.5 - 1, 0.5 - 0] = [-0.5, 0.5]
        
        grads = logits.grad[0, 0]
        print("Gradients:", grads)
        
        self.assertAlmostEqual(grads[0], -0.5, places=2)
        self.assertAlmostEqual(grads[1], 0.5, places=2)

if __name__ == '__main__':
    unittest.main()

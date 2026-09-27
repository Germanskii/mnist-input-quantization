import unittest
import torch
from src.benchmark import preprocess, DigitMLP

class PreprocessingTest(unittest.TestCase):
    def test_baseline_and_quantized_levels(self):
        x=torch.arange(28*28).remainder(256).to(torch.uint8).reshape(1,28,28)
        self.assertTrue(torch.equal(preprocess(x,28),x.float().unsqueeze(1)/255))
        out=preprocess(x,16,4)
        self.assertEqual(out.shape,(1,1,16,16))
        self.assertEqual(out.dtype,torch.float32)
        self.assertLessEqual(out.unique().numel(),16)
        self.assertTrue(torch.allclose(out*15,(out*15).round(),atol=1e-5))
        self.assertEqual(DigitMLP(16)(out).shape,(1,10))

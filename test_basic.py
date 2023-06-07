import torch, unittest
from networks import SelfDistillUNETRWithDictOutput as SelfDistilUNETR


class TestBasic(unittest.TestCase):
    def test_basic(self) -> None:
        model = SelfDistilUNETR(1, 3, img_size=(96, 96, 96), feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0)
        x = torch.randn((1, 1, 96, 96, 96))
        y: torch.Tensor = model(x)
        self.assertEqual(y.shape, (1, 3, 96, 96, 96))
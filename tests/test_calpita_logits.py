"""Run with python -m unittest discover -s tests -p 'test_calpita_logits.py'.

Requires the training environment: torch, torchvision, Lightning, calpit,
NumPy, SciPy, scikit-learn, and monotonicnetworks. Uses synthetic CPU data.
"""
import unittest

import calpit
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.interpolate import PchipInterpolator
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pita_z.models import pita_model, pita_model_gradnorm, pita_model_pcgrad


MODEL_MODULES = (pita_model, pita_model_pcgrad, pita_model_gradnorm)


def make_model(module):
    projection = nn.Linear(3, 2)
    projection.output_dim = 2
    return module.CalPITALightning(
        encoder=nn.Linear(4, 3),
        encoder_mlp=nn.Sequential(nn.Linear(3, 3)),
        projection_head=projection,
        redshift_mlp=calpit.nn.models.MLP(4, [8], sigmoid=False),
        color_mlp=nn.Linear(3, 2),
        loss_type='bce' if module is pita_model_gradnorm else 'monotonic_bce',
        alpha_grid=np.linspace(.01, .99, 5, dtype=np.float32),
        y_grid=np.linspace(0, 4, 9, dtype=np.float32),
        transforms=nn.Identity(),
        transforms_z_metric=nn.Identity(),
        queue_size=8,
        lr=1e-4,
    )


class TestCalpitaLogits(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_saturated_logits_keep_corrective_gradients(self):
        for module in MODEL_MODULES:
            with self.subTest(module=module.__name__):
                model = make_model(module)
                logits = torch.tensor([20., -100.], requires_grad=True)
                loss = model.redshift_loss_fn(logits, torch.tensor([0., 1.]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                torch.testing.assert_close(logits.grad, torch.tensor([.5, -.5]))

    def test_forward_returns_logits(self):
        for module in MODEL_MODULES:
            with self.subTest(module=module.__name__):
                model = make_model(module)
                with torch.no_grad():
                    model.redshift_mlp.layers[-1].weight.zero_()
                    model.redshift_mlp.layers[-1].bias.fill_(20.)
                for goal in ('train', 'validate'):
                    _, logits, _, _ = model(torch.randn(4, 4), goal=goal)
                    torch.testing.assert_close(logits, torch.full_like(logits, 20.))

    def test_ordinary_logits_match_probability_bce(self):
        for module in MODEL_MODULES:
            with self.subTest(module=module.__name__):
                model = make_model(module)
                logits = torch.tensor([-3., 0., 3.])
                target = torch.tensor([0., 1., 1.])
                expected = nn.functional.binary_cross_entropy(logits.sigmoid(), target)
                torch.testing.assert_close(model.redshift_loss_fn(logits, target), expected)

    def test_cde_inference_matches_previous_sigmoid_head(self):
        for module in MODEL_MODULES:
            with self.subTest(module=module.__name__):
                model = make_model(module)
                previous_head = calpit.nn.models.MLP(4, [8], sigmoid=True)
                previous_head.load_state_dict(model.redshift_mlp.state_dict())
                images = torch.randn(4, 4)
                with torch.no_grad():
                    latent = model.encoder_mlp(model.encoder(images))
                    alphas = model.y_grid / 4
                    features = torch.cat([
                        alphas.repeat(4)[:, None],
                        latent.repeat_interleave(len(alphas), dim=0),
                    ], dim=1)
                    cdf = previous_head(features).reshape(4, -1).numpy()
                expected = PchipInterpolator(model.y_grid.numpy(), cdf, axis=1).derivative()(model.y_grid.numpy())
                np.testing.assert_allclose(model.transform_cde(images).numpy(), expected, atol=1e-6)

    def test_short_training_and_validation(self):
        cases = [(module, True) for module in MODEL_MODULES]
        cases.append((pita_model_pcgrad, False))
        for module, labeled in cases:
            with self.subTest(module=module.__name__, labeled=labeled):
                model = make_model(module)
                # The production queue requires an initialized DDP process group.
                # A fixed queue isolates the loss calculation for this CPU check.
                model.dequeue_and_enqueue = lambda keys: None
                dataset = TensorDataset(
                    torch.randn(8, 4), torch.linspace(.1, .9, 8),
                    torch.linspace(.4, 3.6, 8),
                    torch.ones(8) if labeled else torch.zeros(8), torch.randn(8, 2),
                )
                loader = DataLoader(dataset, batch_size=4)
                trainer = pl.Trainer(
                    accelerator='cpu', devices=1, max_epochs=1,
                    limit_train_batches=2, limit_val_batches=1,
                    num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
                    enable_progress_bar=False, enable_model_summary=False,
                )
                trainer.fit(model, loader, loader)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                self.assertTrue(all(torch.isfinite(v).all() for v in trainer.callback_metrics.values()))


if __name__ == '__main__':
    unittest.main()

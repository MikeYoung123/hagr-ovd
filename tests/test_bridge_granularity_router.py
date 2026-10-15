#!/usr/bin/env python3
"""Focused tensor tests for primary-preserving granularity routing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    print("SKIP: torch is not installed")
    sys.exit(0)

from ultralytics.models.yolo.world.train_granularity_router import (
    PrimaryPreservingGranularityLoss,
    PrimaryPreservingGranularityTrainer,
)
from ultralytics.nn.modules.block import ContrastiveHead
from ultralytics.nn.modules.granularity_router import attach_granularity_router
from ultralytics.nn.modules.head import WorldDetect


class RouterHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cv2 = nn.ModuleList([nn.Sequential(nn.Conv2d(4, 4, 1))])
        self.cv3 = nn.ModuleList([nn.Sequential(nn.Conv2d(4, 8, 1))])
        self.cv4 = nn.ModuleList([ContrastiveHead()])


class RouterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.ModuleList([RouterHead()])


class LossHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.nc = 6
        self.reg_max = 16


class LossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.model = nn.ModuleList([LossHead()])
        self.args = SimpleNamespace(
            partial_mp_rank_gain=0.0,
            partial_mp_rank_margin=0.2,
            partial_mp_rank_temperature=1.0,
            partial_mp_rank_topk=4,
            granularity_category_gain=1.0,
            granularity_attribute_gain=1.0,
            granularity_affordance_gain=1.0,
            granularity_attribute_preserve_gain=1.0,
            granularity_train_stage="full",
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )


class GranularityRouterTest(unittest.TestCase):
    def test_world_detect_exposes_immutable_base_scores(self):
        head = WorldDetect(nc=3, embed=8, ch=(4, 8, 16))
        attach_granularity_router(head, hidden_dim=4)
        head.granularity_prompt_roles = torch.tensor([[3, 2, 1]])
        head.train()
        features = [torch.randn(1, 4, 8, 8), torch.randn(1, 8, 4, 4), torch.randn(1, 16, 2, 2)]
        text = torch.randn(1, 3, 8)
        output = head(features, text)
        self.assertIn("granularity_base_scores", output)
        self.assertTrue(torch.equal(output["scores"], output["granularity_base_scores"]))
        self.assertEqual(tuple(output["scores"].shape), (1, 3, 84))

    def test_zero_initialization_and_per_query_routing(self):
        torch.manual_seed(7)
        head = RouterHead()
        router = attach_granularity_router(head, hidden_dim=4)
        visual = torch.randn(1, 8, 3, 3)
        text = torch.randn(1, 3, 8)
        roles = torch.tensor([[3, 2, 1]])
        routed, base = router.forward_scale(0, visual, text, head.cv4[0], roles, "attribute")
        self.assertTrue(torch.equal(routed, base))

        with torch.no_grad():
            router.category[0].up.bias.copy_(torch.linspace(-0.5, 0.5, 8))
        routed, base = router.forward_scale(0, visual, text, head.cv4[0], roles, "attribute")
        self.assertTrue(torch.equal(routed[:, 0], base[:, 0]))
        self.assertFalse(torch.equal(routed[:, 1], base[:, 1]))
        self.assertTrue(torch.equal(routed[:, 2], base[:, 2]))

    def test_extreme_adapter_projection_remains_finite_and_bounded(self):
        head = RouterHead()
        router = attach_granularity_router(head, hidden_dim=4)
        visual = torch.randn(1, 8, 3, 3)
        with torch.no_grad():
            router.shared[0].up.weight.fill_(1e6)
            router.shared[0].up.bias.fill_(1e6)
        residual = router.shared[0].residual(visual)
        self.assertTrue(torch.isfinite(residual).all())
        self.assertLessEqual(float(residual.abs().max()), 0.5)

    def test_direct_inheritance_gate_is_zero_initialized_and_bounded(self):
        head = RouterHead()
        router = attach_granularity_router(head, hidden_dim=4, inheritance_max=0.5)
        category_delta = torch.ones(1, 8, 2, 2)
        self.assertEqual(float(router.attribute_inheritance[0](category_delta).abs().max()), 0.0)
        with torch.no_grad():
            router.attribute_inheritance[0].logits.fill_(100.0)
        inherited = router.attribute_inheritance[0](category_delta)
        self.assertTrue(torch.isfinite(inherited).all())
        self.assertLessEqual(float(inherited.abs().max()), 0.5)

    def test_both_children_directly_inherit_detached_category_residual(self):
        torch.manual_seed(11)
        head = RouterHead()
        router = attach_granularity_router(head, hidden_dim=4, category_prior=1.0)
        visual = torch.randn(1, 8, 3, 3)
        text = torch.randn(1, 2, 8)
        roles = torch.tensor([[3, 1]])
        with torch.no_grad():
            router.category[0].up.bias.copy_(torch.linspace(-0.4, 0.4, 8))
            router.attribute_inheritance[0].logits.fill_(0.25)
            router.affordance_inheritance[0].logits.fill_(-0.25)

        router.category_prior = 0.0
        flat, _ = router.forward_scale(0, visual, text, head.cv4[0], roles, "attribute")
        router.category_prior = 1.0
        hierarchical, _ = router.forward_scale(0, visual, text, head.cv4[0], roles, "attribute")
        self.assertFalse(torch.equal(flat[:, 0], hierarchical[:, 0]))
        self.assertFalse(torch.equal(flat[:, 1], hierarchical[:, 1]))

        loss = hierarchical.sum()
        loss.backward()
        shared_grads = [parameter.grad for parameter in router.shared[0].parameters()]
        category_grads = [parameter.grad for parameter in router.category[0].parameters()]
        attribute_grads = [parameter.grad for parameter in router.attribute[0].parameters()]
        affordance_grads = [parameter.grad for parameter in router.affordance[0].parameters()]
        attribute_gate_grads = [parameter.grad for parameter in router.attribute_inheritance[0].parameters()]
        affordance_gate_grads = [parameter.grad for parameter in router.affordance_inheritance[0].parameters()]
        self.assertTrue(any(grad is not None and float(grad.abs().sum()) > 0.0 for grad in shared_grads))
        self.assertTrue(all(grad is None or float(grad.abs().sum()) == 0.0 for grad in category_grads))
        self.assertTrue(any(grad is not None and float(grad.abs().sum()) > 0.0 for grad in attribute_grads))
        self.assertTrue(any(grad is not None and float(grad.abs().sum()) > 0.0 for grad in affordance_grads))
        self.assertTrue(any(grad is not None and float(grad.abs().sum()) > 0.0 for grad in attribute_gate_grads))
        self.assertTrue(any(grad is not None and float(grad.abs().sum()) > 0.0 for grad in affordance_gate_grads))

    def test_disabling_stop_gradient_allows_child_loss_to_update_category_root(self):
        torch.manual_seed(13)
        head = RouterHead()
        router = attach_granularity_router(
            head, hidden_dim=4, category_prior=1.0, stop_gradient=False
        )
        visual = torch.randn(1, 8, 3, 3)
        text = torch.randn(1, 2, 8)
        child_roles = torch.tensor([[3, 1]])
        with torch.no_grad():
            router.category[0].up.bias.copy_(torch.linspace(-0.4, 0.4, 8))
            router.attribute_inheritance[0].logits.fill_(0.25)
            router.affordance_inheritance[0].logits.fill_(-0.25)

        child_scores, _ = router.forward_scale(
            0, visual, text, head.cv4[0], child_roles, "attribute"
        )
        child_scores.sum().backward()
        category_grads = [parameter.grad for parameter in router.category[0].parameters()]
        self.assertFalse(router.stop_gradient)
        self.assertTrue(any(grad is not None and float(grad.abs().sum()) > 0.0 for grad in category_grads))

    def test_v4_router_upgrade_adds_zero_gates_without_changing_output(self):
        torch.manual_seed(17)
        head = RouterHead()
        router = attach_granularity_router(head, hidden_dim=4, category_prior=1.0)
        visual = torch.randn(1, 8, 3, 3)
        text = torch.randn(1, 2, 8)
        roles = torch.tensor([[3, 1]])
        with torch.no_grad():
            router.category[0].up.bias.copy_(torch.linspace(-0.4, 0.4, 8))
            router.attribute[0].up.weight.normal_(std=0.2)
            router.affordance[0].up.weight.normal_(std=0.2)
        del router.attribute_inheritance
        del router.affordance_inheritance
        del router.inheritance_max
        router.router_version = 4
        legacy, _ = router.forward_scale(0, visual, text, head.cv4[0], roles, "attribute")
        upgraded = attach_granularity_router(head, hidden_dim=4, category_prior=1.0, inheritance_max=0.5)
        current, _ = upgraded.forward_scale(0, visual, text, head.cv4[0], roles, "attribute")
        self.assertEqual(upgraded.router_version, 5)
        self.assertTrue(torch.equal(legacy, current))

    def test_loss_updates_all_roles_but_not_frozen_base(self):
        criterion = PrimaryPreservingGranularityLoss(LossModel())
        boxes = torch.zeros(1, 64, 21, requires_grad=True)
        scores = torch.zeros(1, 6, 21, requires_grad=True)
        base_scores = torch.zeros(1, 6, 21, requires_grad=True)
        predictions = {
            "boxes": boxes,
            "scores": scores,
            "granularity_base_scores": base_scores,
            "feats": [torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 1, 1)],
        }
        batch = {
            "batch_idx": torch.tensor([0, 0]),
            "cls": torch.tensor([[0.0], [3.0]]),
            "bboxes": torch.tensor([[0.3, 0.3, 0.4, 0.4], [0.7, 0.7, 0.4, 0.4]]),
            "prompt_roles": torch.tensor([[3, 2, 1, 3, 1, 0]]),
            "multi_positive": torch.tensor(
                [
                    [[0, 3.0], [1, 2.0], [2, 1.0], [3, -1.0], [-1, 0.0]],
                    [[3, 3.0], [1, 2.0], [4, 1.0], [0, -1.0], [-1, 0.0]],
                ],
                dtype=torch.float32,
            ),
        }
        total, components = criterion.loss(predictions, batch)
        self.assertEqual(components.shape, (6,))
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(components).all())
        total.backward()
        self.assertIsNotNone(scores.grad)
        self.assertGreater(float(scores.grad[:, [0, 1, 2, 3, 4]].abs().sum()), 0.0)
        self.assertEqual(float(scores.grad[:, 5].abs().sum()), 0.0)
        self.assertIsNone(base_scores.grad)
        self.assertIsNone(boxes.grad)

    def test_attribute_preservation_penalizes_only_attribute_logits(self):
        pred = torch.zeros(1, 4, 3, requires_grad=True)
        base = torch.zeros_like(pred, requires_grad=True)
        roles = torch.tensor([[3, 2, 1]])
        unchanged = PrimaryPreservingGranularityLoss._attribute_preserve_loss(pred, base, roles)
        self.assertEqual(float(unchanged), 0.0)

        shifted = pred.clone()
        shifted[:, :, 0] += 1.0
        penalty = PrimaryPreservingGranularityLoss._attribute_preserve_loss(shifted, base, roles)
        self.assertGreater(float(penalty), 0.0)
        penalty.backward()
        self.assertGreater(float(pred.grad[:, :, 0].abs().sum()), 0.0)
        self.assertEqual(float(pred.grad[:, :, 1:].abs().sum()), 0.0)
        self.assertIsNone(base.grad)

    def test_shared_residual_uses_lower_learning_rate(self):
        model = RouterModel()
        attach_granularity_router(model.model[-1], hidden_dim=4)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ".granularity_router." in f".{name}"
        trainer = object.__new__(PrimaryPreservingGranularityTrainer)
        trainer.args = SimpleNamespace(granularity_shared_lr_mult=0.25, granularity_box_lr_mult=0.25)
        optimizer = trainer.build_optimizer(model, lr=0.001, momentum=0.9, decay=0.0001)
        by_name = {group["param_group"]: group["lr"] for group in optimizer.param_groups}
        self.assertAlmostEqual(by_name["inheritance_bias"], 0.001)
        self.assertAlmostEqual(by_name["role_weight"], 0.001)
        self.assertAlmostEqual(by_name["role_bias"], 0.001)
        self.assertAlmostEqual(by_name["shared_weight"], 0.00025)
        self.assertAlmostEqual(by_name["shared_bias"], 0.00025)

    def test_gate_stage_freezes_everything_except_direct_inheritance(self):
        model = RouterModel()
        router = attach_granularity_router(model.model[-1], hidden_dim=4)
        with torch.no_grad():
            router.category[0].up.bias.fill_(0.1)
        trainer = object.__new__(PrimaryPreservingGranularityTrainer)
        trainer.model = model
        trainer.args = SimpleNamespace(granularity_train_stage="gate")
        trainer.configure_trainable_parameters()
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all("_inheritance." in name for name in trainable))

        trainer.args.granularity_train_stage = "gate_child"
        trainer.configure_trainable_parameters()
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(any(".attribute." in name for name in trainable))
        self.assertTrue(any(".affordance." in name for name in trainable))
        self.assertFalse(any(".shared." in name or ".category." in name for name in trainable))


if __name__ == "__main__":
    unittest.main()

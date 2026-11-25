import copy
import unittest

from core.runner import Runner
from core.schema import ACTIONS, ReviewTask, RunnerConfig
from rl.adapter import step_with_masks
from rl.state_encoder import FEATURES, StateEncoder


class EncoderTests(unittest.TestCase):
    def setUp(self):
        papers = [{"id": p, "title": p, "abstract": "Synthetic abstract."} for p in ("A", "B")]
        self.runner = Runner({"papers": papers}, config=RunnerConfig(retrieval_batch_size=1))
        self.start = self.runner.reset(ReviewTask("id", "Synthetic abstract"))
        self.encoder = StateEncoder(paper_scale=2)

    def fields(self, state, mask):
        values = self.encoder.encode(state, mask)
        self.assertEqual(len(values), len(FEATURES))
        return dict(zip(FEATURES, values))

    def test_frozen_layout_and_cold_start(self):
        spec = self.encoder.spec()
        self.assertEqual(spec["action_order"], list(ACTIONS))
        self.assertEqual(spec["features"], list(FEATURES))
        self.assertEqual(len(set(FEATURES)), len(FEATURES))
        f = self.fields(self.start, self.runner.valid_actions())
        self.assertEqual((f["difficulty.medium"], f["can_retrieve_more"], f["token_remaining_fraction"]), (1, 1, 1))
        self.assertEqual((f["verifier_fresh"], f["verifier_stale"], f["score_valid.coverage"]), (0, 0, 0))
        self.assertEqual(f["summarize_fast.delta_valid"], 0)
        with self.assertRaises(ValueError):
            self.encoder.encode(self.start, [1] * 5)
        with self.assertRaises(ValueError):
            self.encoder.encode({**self.start, "config_version": -1}, [False] * 5)

    def test_comparable_zero_missing_delta_and_stale_score(self):
        r = self.runner
        retrieval = step_with_masks(r, "retrieve_more")
        self.assertEqual(retrieval["action_mask"], [True, False, False, False, False])
        self.assertEqual(retrieval["next_mask"], [True] * 5)
        self.assertNotIn("next_mask", r.transitions[-1])
        step_with_masks(r, "summarize_fast")
        t = step_with_masks(r, "summarize_fast")
        before = self.fields(t["state_before"], t["action_mask"])
        after = self.fields(t["state_after"], t["next_mask"])
        self.assertEqual(before["summarize_fast.delta_valid"], 0)
        self.assertEqual((after["summarize_fast.delta_valid"], after["summarize_fast.last_delta"]), (1, 0))
        state = copy.deepcopy(t["state_after"])
        state["verifier"]["scores"]["coverage"] = 0.0
        fresh = self.fields(state, t["next_mask"])
        state["material_revision"] += 1
        stale = self.fields(state, t["next_mask"])
        self.assertEqual((fresh["score.coverage"], fresh["score_valid.coverage"]), (0, 1))
        self.assertEqual((stale["score.coverage"], stale["score_valid.coverage"], stale["verifier_stale"]), (0, 0, 1))
        t = step_with_masks(r, "retrieve_more")
        f = self.fields(t["state_after"], t["next_mask"])
        self.assertEqual((f["summary_available"], f["can_retrieve_more"], f["retrieve_more.delta_valid"]), (0, 0, 0))
        final = step_with_masks(r, "write_deep")
        self.assertTrue(final["done"])
        self.assertEqual(final["next_mask"], [False] * 5)
        self.assertEqual(self.fields(final["state_after"], final["next_mask"])["write_deep.delta_valid"], 0)

    def test_encoder_does_not_read_identity_labels_or_future(self):
        mask = self.runner.valid_actions()
        polluted = {**self.start, "review_id": "elsewhere", "overall_score": 1,
                    "gold_paper_ids": ["A"], "state_after": {"success": True}}
        self.assertEqual(self.encoder.encode(polluted, mask), self.encoder.encode(self.start, mask))
        unknown = {**self.start, "workflow_tokens_used": None}
        self.assertEqual(self.fields(unknown, [False] * 5)["tokens_known"], 0)
        self.assertEqual(self.fields(self.start, mask)["tokens_known"], 1)


if __name__ == "__main__":
    unittest.main()

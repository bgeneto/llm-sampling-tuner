import json
import unittest
from unittest.mock import patch

import runner
from config import expand_param_combos
from run_coarse import build_phase_name
from prompts.planner_prompts import PLANNER_PROMPTS


BASE_COMBO = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
}


class FakeResponse:
    status_code = 200
    headers = {"content-type": "application/json"}
    text = '{"choices":[{"message":{"content":"ok"}}],"usage":{}}'

    def json(self):
        return json.loads(self.text)


class ReasoningBudgetTests(unittest.TestCase):
    def capture_payload(self, reasoning_profiles, thinking_token_budget=None):
        params = expand_param_combos(
            [BASE_COMBO],
            reasoning_profiles,
            thinking_token_budget=thinking_token_budget,
        )[0]
        with patch("runner.requests.post", return_value=FakeResponse()) as post:
            runner.call_lmstudio([{"role": "user", "content": "hello"}], params, 2048)
        return post.call_args.kwargs["json"]

    def test_dynamic_thinking_budget_is_top_level_sampling_param(self):
        payload = self.capture_payload(["thinking_2048"])

        self.assertEqual(payload["thinking_token_budget"], 2048)
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})

    def test_custom_thinking_budget_is_top_level_sampling_param(self):
        payload = self.capture_payload(["thinking_custom"], thinking_token_budget=2048)

        self.assertEqual(payload["thinking_token_budget"], 2048)
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})

    def test_full_prompt_phase_name_includes_concrete_reasoning_budget(self):
        phase_name = build_phase_name(
            explicit_phase_name=None,
            mode="planner",
            prompts=PLANNER_PROMPTS,
            analysis_path=None,
            top_n=5,
            param_hashes=[],
            reasoning_profiles=["thinking_custom"],
            thinking_token_budget=2048,
        )

        self.assertEqual(phase_name, "coarse_v2_thinking_2048")


if __name__ == "__main__":
    unittest.main()

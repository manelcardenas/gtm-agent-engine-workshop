import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service, gtm_agent


class ProspectPrivacyTests(unittest.TestCase):
    def test_get_prospect_returns_only_contact_and_routing_fields(self):
        record = {
            "name": "Ari Example",
            "email": "ari@example.com",
            "disqualified": False,
            "billing_qualification": {"tax_id": "sensitive"},
            "annual_revenue": 1000000,
            "engagement_history": [],
        }
        with patch.object(data_service, "get_prospect_record", return_value=record):
            result = gtm_agent.get_prospect.invoke({"prospect_id": "LEAD-1"})

        self.assertEqual(
            result,
            {
                "prospect": {
                    "prospect_id": "LEAD-1",
                    "name": "Ari Example",
                    "email": "ari@example.com",
                    "disqualified": False,
                },
                "found": True,
            },
        )

    def test_new_profile_excludes_billing_and_preserves_supported_fields(self):
        record = {
            "name": "Ari Example",
            "email": "ari@example.com",
            "annual_revenue": 1000000,
            "disqualified": False,
            "billing_qualification": {"tax_id": "sensitive"},
            "unexpected": "not supported",
        }
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(data_service, "get_prospect_record", return_value=record),
            patch.object(data_service, "fetch_engagement_history", return_value=[{"channel": "demo"}]),
            patch.object(data_service, "fetch_account_details", return_value=[{"account": "Example"}]),
            patch.object(data_service, "fetch_tech_stack", return_value=["Python"]),
            patch.object(data_service, "save_profile_to_db") as save_profile,
        ):
            result = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

        profile = result["prospect_profile"]
        self.assertEqual(
            profile,
            {
                "prospect_id": "LEAD-1",
                "name": "Ari Example",
                "email": "ari@example.com",
                "annual_revenue": 1000000,
                "disqualified": False,
                "engagement_history": [{"channel": "demo"}],
                "account_details": [{"account": "Example"}],
                "tech_stack": ["Python"],
            },
        )
        save_profile.assert_called_once_with("LEAD-1", profile)

    def test_cached_profile_is_sanitized_before_return(self):
        stored_profile = {
            "prospect_id": "LEAD-1",
            "name": "Ari Example",
            "email": "ari@example.com",
            "annual_revenue": 1000000,
            "disqualified": False,
            "engagement_history": [],
            "account_details": [],
            "tech_stack": ["Python"],
            "billing_qualification": {"tax_id": "sensitive"},
            "unexpected": "not supported",
        }
        with patch.object(
            data_service,
            "get_profile_from_db",
            return_value={"prospect_profile": stored_profile},
        ):
            result = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

        profile = result["prospect_profile"]
        self.assertNotIn("billing_qualification", profile)
        self.assertNotIn("unexpected", profile)
        self.assertEqual(profile["name"], "Ari Example")
        self.assertEqual(profile["annual_revenue"], 1000000)
        self.assertEqual(profile["tech_stack"], ["Python"])

    def test_profile_store_read_sanitizes_traced_result(self):
        stored_profile = {
            "prospect_id": "LEAD-1",
            "name": "Ari Example",
            "billing_qualification": {"tax_id": "sensitive"},
            "unexpected": "not supported",
        }
        with patch.dict(data_service._PROFILES, {"LEAD-1": stored_profile}, clear=True):
            result = data_service.get_profile_from_db("LEAD-1")

        self.assertEqual(result["prospect_profile"], {
            "prospect_id": "LEAD-1",
            "name": "Ari Example",
        })

    def test_scoring_payload_contains_only_supported_fields(self):
        profile = {
            "prospect_id": "LEAD-1",
            "name": "Ari Example",
            "annual_revenue": 1000000,
            "tech_stack": ["stale"],
            "account_details": [{"account": "Example"}],
            "engagement_history": [{"channel": "demo"}],
            "email": "ari@example.com",
            "billing_qualification": {"tax_id": "sensitive"},
            "unexpected": "not supported",
        }
        offering = {
            "required_tech_stack": ["Python"],
            "min_annual_revenue": 500000,
            "description": "Example offering",
        }
        llm_result = Mock()
        llm_result.model_dump.return_value = {"score": 80}
        with (
            patch.object(data_service, "fetch_tech_stack", return_value=["Python"]),
            patch.object(gtm_agent, "_scoring_llm") as scoring_llm,
        ):
            scoring_llm.invoke.return_value = llm_result
            gtm_agent.score_prospect.invoke({
                "prospect_profile": profile,
                "offering": offering,
            })

        user_message = scoring_llm.invoke.call_args.args[0][1]["content"]
        scoring_payload = json.loads(user_message.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(scoring_payload, {
            "prospect_id": "LEAD-1",
            "name": "Ari Example",
            "annual_revenue": 1000000,
            "tech_stack": ["Python"],
            "account_details": [{"account": "Example"}],
            "engagement_history": [{"channel": "demo"}],
        })
        self.assertNotIn("billing_qualification", user_message)


if __name__ == "__main__":
    unittest.main()

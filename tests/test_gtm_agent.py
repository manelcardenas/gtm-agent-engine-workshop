import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from gtm_agent import gtm_agent as module


class QueuedChatModel(BaseChatModel):
    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "queued-test-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def make_agent(*responses):
    return create_agent(
        QueuedChatModel(responses=list(responses)),
        tools=module.DOMAIN_TOOLS,
        system_prompt=module.SYSTEM_PROMPT,
    )


class AgentConstructionTests(unittest.TestCase):
    def test_agent_exposes_only_domain_tools_without_harness_nodes(self):
        graph = module.gtm_agent.get_graph()
        tools = graph.nodes["tools"].data.tools_by_name

        self.assertEqual(
            set(tools),
            {
                "lookup_offering",
                "build_prospect_profile",
                "get_prospect",
                "get_current_rep",
                "send_prospect_email",
                "score_prospect",
                "update_prospect_info",
            },
        )
        self.assertEqual(set(graph.nodes), {"__start__", "model", "tools", "__end__"})


class ToolRuntimeTests(unittest.TestCase):
    def test_get_current_rep_receives_runtime_metadata(self):
        agent = make_agent(
            AIMessage(
                content="",
                tool_calls=[{"name": "get_current_rep", "args": {}, "id": "rep-call", "type": "tool_call"}],
            ),
            AIMessage(content="Rep found"),
        )
        rep = {"rep_id": "rep-1", "name": "Taylor", "email": "taylor@example.com"}

        with patch.object(module.data_service, "get_rep", return_value=rep) as get_rep:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Who am I?"}]},
                config={"metadata": {"user_id": "rep-1"}},
            )

        get_rep.assert_called_once_with("rep-1")
        tool_result = next(message for message in result["messages"] if isinstance(message, ToolMessage))
        self.assertIn('"found": true', tool_result.content)

    def test_send_prospect_email_receives_runtime_metadata(self):
        agent = make_agent(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "send_prospect_email",
                        "args": {
                            "prospect": {"name": "Jordan", "email": "jordan@example.com"},
                            "subject": "Hello",
                            "body": "Following up",
                        },
                        "id": "email-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Email sent"),
        )
        rep = {"rep_id": "rep-1", "name": "Taylor", "email": "taylor@example.com"}

        with patch.object(module.data_service, "get_rep", return_value=rep) as get_rep:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Email Jordan"}]},
                config={"metadata": {"user_id": "rep-1"}},
            )

        get_rep.assert_called_once_with("rep-1")
        tool_result = next(message for message in result["messages"] if isinstance(message, ToolMessage))
        self.assertIn('"status": "sent"', tool_result.content)
        self.assertIn('"from": "taylor@example.com"', tool_result.content)


class RunAgentConfigurationTests(unittest.TestCase):
    def test_run_agent_preserves_run_configuration(self):
        result_messages = [AIMessage(content="Done")]
        with patch.object(module.gtm_agent, "invoke", return_value={"messages": result_messages}) as invoke:
            result = module.run_agent(
                "Email prospect LEAD-1",
                user_id="rep-1",
                environment="staging",
                thread_id="thread-1",
            )

        config = invoke.call_args.kwargs["config"]
        self.assertEqual(config["run_name"], "GTM Assistant")
        self.assertEqual(config["metadata"], {
            "thread_id": "thread-1",
            "user_id": "rep-1",
            "environment": "staging",
            "request_intent": "send_email",
        })
        self.assertEqual(str(config["run_id"]), result["run_id"])
        self.assertEqual(result["messages"], result_messages)

"""
test_agent.py - Tests for the Email Draft Improver Reflection Agent.

Tests the graph structure, nodes, and routing logic
without calling the LLM (using mocks).
"""

import pytest
from unittest.mock import patch, MagicMock
from langgraph.graph import END

import agent


# ===== Test State Setup =====

def _make_state(**overrides) -> agent.EmailState:
    """Create a default EmailState with optional overrides."""
    defaults = {
        "messages": [],
        "topic": "Budget approval for Q3",
        "recipient": "CFO",
        "draft": "",
        "critique": "",
        "iteration": 0,
    }
    defaults.update(overrides)
    return defaults


# ===== Tests: should_continue routing =====

class TestRouting:
    def test_approved_ends(self):
        """When critique contains APPROVED, should return END."""
        state = _make_state(critique="APPROVED", iteration=1)
        assert agent.should_continue(state) == END

    def test_approved_case_insensitive(self):
        """APPROVED check should be case insensitive."""
        state = _make_state(critique="The email is great. Approved.", iteration=1)
        assert agent.should_continue(state) == END

    def test_max_iterations_ends(self):
        """When iteration >= MAX_ITERATIONS, should return END."""
        state = _make_state(critique="Needs more work", iteration=agent.MAX_ITERATIONS)
        assert agent.should_continue(state) == END

    def test_continues_when_not_approved(self):
        """When critique has feedback and under max, should continue."""
        state = _make_state(critique="Tone is too casual. Fix the greeting.", iteration=1)
        assert agent.should_continue(state) == "generator"

    def test_continues_at_iteration_1(self):
        """First iteration with feedback should continue."""
        state = _make_state(critique="Needs a stronger call to action.", iteration=1)
        assert agent.should_continue(state) == "generator"

    def test_continues_at_iteration_2(self):
        """Second iteration with feedback should still continue."""
        state = _make_state(critique="Structure needs work.", iteration=2)
        assert agent.should_continue(state) == "generator"

    def test_empty_critique_continues(self):
        """Empty critique with low iteration should continue."""
        state = _make_state(critique="", iteration=0)
        # Empty critique won't have APPROVED, and iteration < max
        assert agent.should_continue(state) == "generator"


# ===== Tests: generator_node =====

class TestGeneratorNode:
    @patch.object(agent, "llm")
    def test_first_draft_uses_topic(self, mock_llm):
        """First call (no critique) should generate from topic."""
        mock_response = MagicMock()
        mock_response.content = "Subject: Budget Approval\n\nDear CFO..."
        mock_llm.invoke.return_value = mock_response

        state = _make_state(critique="", iteration=0)
        result = agent.generator_node(state)

        # Check it generated a draft
        assert result["draft"] == "Subject: Budget Approval\n\nDear CFO..."
        assert result["iteration"] == 1

        # Check the prompt included topic and recipient
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[0].content
        assert "Budget approval for Q3" in prompt_text
        assert "CFO" in prompt_text

    @patch.object(agent, "llm")
    def test_rewrite_uses_critique(self, mock_llm):
        """Later calls (with critique) should rewrite based on feedback."""
        mock_response = MagicMock()
        mock_response.content = "Subject: Budget Approval Request\n\nDear CFO, improved..."
        mock_llm.invoke.return_value = mock_response

        state = _make_state(
            critique="Tone is too casual. Add specific numbers.",
            draft="Subject: Hey\n\nHey boss, need money...",
            iteration=1,
        )
        result = agent.generator_node(state)

        assert result["draft"] == "Subject: Budget Approval Request\n\nDear CFO, improved..."
        assert result["iteration"] == 2

        # Check the prompt included the critique
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[0].content
        assert "Tone is too casual" in prompt_text
        assert "Hey boss, need money" in prompt_text

    @patch.object(agent, "llm")
    def test_iteration_increments(self, mock_llm):
        """Each call should increment the iteration counter."""
        mock_response = MagicMock()
        mock_response.content = "Draft"
        mock_llm.invoke.return_value = mock_response

        state = _make_state(iteration=2, critique="Fix it")
        result = agent.generator_node(state)
        assert result["iteration"] == 3


# ===== Tests: reflector_node =====

class TestReflectorNode:
    @patch.object(agent, "llm")
    def test_returns_critique(self, mock_llm):
        """Reflector should return critique in state."""
        mock_response = MagicMock()
        mock_response.content = "TONE: 6/10 - too casual for a CFO"
        mock_llm.invoke.return_value = mock_response

        state = _make_state(draft="Hey boss, need some cash for the project.")
        result = agent.reflector_node(state)

        assert "critique" in result
        assert "too casual" in result["critique"]

    @patch.object(agent, "llm")
    def test_approved_response(self, mock_llm):
        """When draft is good, reflector should say APPROVED."""
        mock_response = MagicMock()
        mock_response.content = "APPROVED"
        mock_llm.invoke.return_value = mock_response

        state = _make_state(draft="Subject: Q3 Budget Approval\n\nDear CFO...")
        result = agent.reflector_node(state)

        assert "APPROVED" in result["critique"]

    @patch.object(agent, "llm")
    def test_prompt_includes_context(self, mock_llm):
        """Reflector prompt should include the draft, topic, and recipient."""
        mock_response = MagicMock()
        mock_response.content = "Needs work"
        mock_llm.invoke.return_value = mock_response

        state = _make_state(
            draft="Test draft content",
            topic="Project update",
            recipient="CEO",
        )
        agent.reflector_node(state)

        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[0].content
        assert "Test draft content" in prompt_text
        assert "Project update" in prompt_text
        assert "CEO" in prompt_text


# ===== Tests: Graph Structure =====

class TestGraphStructure:
    def test_graph_compiles(self):
        """The graph should compile without errors."""
        app = agent.build_agent()
        assert app is not None

    def test_graph_has_generator_node(self):
        """Graph should have a generator node."""
        app = agent.build_agent()
        # The compiled graph should contain our nodes
        assert "generator" in app.get_graph().nodes

    def test_graph_has_reflector_node(self):
        """Graph should have a reflector node."""
        app = agent.build_agent()
        assert "reflector" in app.get_graph().nodes


# ===== Tests: run_email_improver =====

class TestRunEmailImprover:
    @patch.object(agent, "llm")
    def test_full_run_with_approval(self, mock_llm):
        """Full run: generator writes, reflector approves → 1 iteration."""
        responses = [
            MagicMock(content="Subject: Q3 Budget\n\nDear CFO, I am writing to request..."),
            MagicMock(content="APPROVED"),
        ]
        mock_llm.invoke.side_effect = responses

        result = agent.run_email_improver("Q3 budget approval", "CFO")

        assert result["iteration"] == 1
        assert "APPROVED" in result["critique"]
        assert "Subject:" in result["draft"]

    @patch.object(agent, "llm")
    def test_full_run_with_revision(self, mock_llm):
        """Full run: first draft gets critique, second gets approved → 2 iterations."""
        responses = [
            MagicMock(content="Subject: Hey\n\nHey, need budget..."),          # gen 1
            MagicMock(content="TONE: 4/10 - way too casual for a CFO."),       # reflect 1
            MagicMock(content="Subject: Q3 Budget Request\n\nDear CFO..."),    # gen 2
            MagicMock(content="APPROVED"),                                      # reflect 2
        ]
        mock_llm.invoke.side_effect = responses

        result = agent.run_email_improver("Q3 budget approval", "CFO")

        assert result["iteration"] == 2
        assert "APPROVED" in result["critique"]
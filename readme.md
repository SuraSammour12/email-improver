# EmailCraft: Self-Improving Email Draft Agent

An agentic AI system built with **LangGraph** that writes professional emails, **critiques its own drafts**, and iteratively improves them using the **Reflection pattern**. The agent autonomously generates, evaluates, and refines emails until they meet professional standards.

Built as a portfolio project demonstrating **agent orchestration**, **self-reflection loops**, and **iterative improvement patterns**.

## Live Demo Results

The following examples show **actual outputs** from running the agent with real LLM inference (July 2026).

### Example 1: Budget Approval Request (2 iterations)

```
Email topic: Request budget approval for Q3 infrastructure upgrades
Recipient: CFO

  FINAL EMAIL (after 2 iteration(s))
  Status: APPROVED by reflector

Subject: Request for Budget Approval for Q3 Infrastructure Upgrades

Dear Mr. Johnson,

As we continue to drive business growth and expansion, it's essential
that our IT infrastructure keeps pace. Our IT team has identified key
infrastructure upgrades necessary to support the company's continued
growth and efficiency. These upgrades will enable us to enhance our
network security, improve data storage capacity, and increase overall
system performance, resulting in increased productivity and reduced
downtime.

The total budget required for these upgrades is $250,000, which will
be allocated across various departments. I have attached a detailed
breakdown of the proposed expenses for your review.

Please review and approve the budget by the end of this week, so we
can proceed with the upgrades as scheduled. If you have any questions
or concerns, please don't hesitate to reach out.

Best regards,
[Your Name]
```

The agent wrote the first draft, the reflector found issues with tone and specificity, and the generator rewrote it. The second version was approved.

### Example 2: Client Apology (2 iterations)

```
Email topic: Apologize for missing the project deadline
Recipient: client

  FINAL EMAIL (after 2 iteration(s))
  Status: APPROVED by reflector

Subject: Apology for Missing Project Deadline and Next Steps

Dear [Client],

I am writing to apologize for missing the deadline for your project,
which I understand is a significant inconvenience and may have caused
disruption to your plans. I regret any frustration or uncertainty this
may have caused and want to assure you that we are working diligently
to complete the project as soon as possible.

A call will allow us to discuss the revised timeline in more detail,
address any concerns you may have, and provide a clear understanding
of the project's status. Would you be available for a 30-minute call
on Wednesday or Thursday of this week?

I appreciate your understanding and patience in this matter and look
forward to speaking with you soon to rectify the situation.

Sincerely,
[Your Name]
```

The reflector critiqued the first draft for lacking a concrete call to action and specific next steps. The revised version added the meeting request and timeline.

### Example 3: Team Restructuring Announcement (1 iteration - approved on first draft)

```
Email topic: Announce a team restructuring effective next month
Recipient: engineering team

  FINAL EMAIL (after 1 iteration(s))
  Status: APPROVED by reflector

Subject: Upcoming Team Restructuring and What to Expect

Hello Engineering Team,

I am writing to announce that we will be undergoing a team
restructuring, effective next month. This change is aimed at improving
our workflow, enhancing collaboration, and driving innovation within
the team.

In the coming weeks, you can expect to receive more information about
the changes, including new team assignments, roles, and
responsibilities. In preparation, I encourage each of you to review
our current projects and identify areas where we can streamline our
processes.

To ensure a smooth transition, I would like to schedule a meeting with
each of you to discuss your specific role and how it will be impacted.
Please reply to this email by the end of the week to schedule a
meeting at your convenience.

Best regards,
[Your Name]
```

The reflector rated all five categories 8+ on the first draft and approved immediately - no revision needed.

## How Reflection Works

The agent doesn't just generate once - it runs a **self-improvement loop**:

```
         +-------------+        +--------------+
         |  Generator  |------->|  Reflector   |
         |  (writes/   |        |  (critiques  |
         |   rewrites) |        |   the draft) |
         +------^------+        +------+-------+
                |                      |
                |   Needs work         | All scores 8+
                +----------------------+       |
                                               v
                                              END
                                        (final email)
```

**The Reflector rates 5 categories (1-10):**

| Category | What it checks |
|----------|---------------|
| Tone | Appropriate for the recipient? |
| Clarity | Core message immediately clear? |
| Structure | Strong opening, logical body, clear closing? |
| Professionalism | Grammar, word choice, formatting? |
| Call to Action | Reader knows exactly what to do next? |

If **all scores are 8+** → APPROVED. Otherwise → specific feedback goes back to the Generator for a rewrite. Maximum 3 iterations as a safety net.

## Architecture

```
+-------+    +-------------+    +--------------+    +-----------------+
| User  |--->|  generator  |--->|  reflector   |--->| should_continue |
| Input |    |  (Node 1)   |    |  (Node 2)    |    | (Conditional    |
+-------+    +------^------+    +--------------+    |  Edge)          |
                    |                               +--------+--------+
                    |         "needs work"                   |
                    +----------------------------------------+
                                                             |
                                                    "APPROVED" or max
                                                             |
                                                             v
                                                            END
```

**Pattern:** Reflection (Generate → Critique → Improve → Repeat)

**Framework:** LangGraph with conditional routing and loop cycle

**LLM:** Groq (Llama 3.3 70B), free and fast inference

**Key difference from simple agents:** the graph has a **cycle** - the reflector's output feeds back into the generator. This is what makes it self-improving.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/emailcraft-agent.git
cd emailcraft-agent

pip install -r requirements.txt

cp .env.example .env
# Add your GROQ_API_KEY to .env (free at console.groq.com)

python agent.py
```

## Project Structure

```
emailcraft-agent/
├── agent.py            # LangGraph Reflection agent (graph, nodes, edges, loop)
├── test_agent.py       # 18 test cases (mocked LLM, no API calls needed)
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

## Tests

```bash
python -m pytest test_agent.py -v
```

```
test_agent.py::TestRouting::test_approved_ends PASSED
test_agent.py::TestRouting::test_approved_case_insensitive PASSED
test_agent.py::TestRouting::test_max_iterations_ends PASSED
test_agent.py::TestRouting::test_continues_when_not_approved PASSED
test_agent.py::TestRouting::test_continues_at_iteration_1 PASSED
test_agent.py::TestRouting::test_continues_at_iteration_2 PASSED
test_agent.py::TestRouting::test_empty_critique_continues PASSED
test_agent.py::TestGeneratorNode::test_first_draft_uses_topic PASSED
test_agent.py::TestGeneratorNode::test_rewrite_uses_critique PASSED
test_agent.py::TestGeneratorNode::test_iteration_increments PASSED
test_agent.py::TestReflectorNode::test_returns_critique PASSED
test_agent.py::TestReflectorNode::test_approved_response PASSED
test_agent.py::TestReflectorNode::test_prompt_includes_context PASSED
test_agent.py::TestGraphStructure::test_graph_compiles PASSED
test_agent.py::TestGraphStructure::test_graph_has_generator_node PASSED
test_agent.py::TestGraphStructure::test_graph_has_reflector_node PASSED
test_agent.py::TestRunEmailImprover::test_full_run_with_approval PASSED
test_agent.py::TestRunEmailImprover::test_full_run_with_revision PASSED

============================== 18 passed in 25.67s =============================
```

Tests use **mocked LLM calls** - no API key needed to run tests. They verify routing logic, node behavior, graph structure, and full agent runs with simulated approval and revision flows.

## Key Technical Decisions

1. **Reflection pattern**: The agent self-critiques using a structured rubric (5 categories, 1-10 scale) rather than vague "is this good?" prompts.
2. **Bounded iteration**: Maximum 3 loops prevents infinite reflection cycles while allowing meaningful improvement.
3. **Dual-behavior generator**: The same node handles both initial drafting and revision, keeping the graph simple.
4. **Explicit approval signal**: The reflector outputs "APPROVED" as a clear stop signal, parsed by the routing function.
5. **Mocked tests**: All 18 tests run without an API key by mocking LLM responses, making CI/CD reliable.

## Known Limitations

- Email quality depends on the underlying LLM's writing ability
- The reflector and generator use the same LLM (ideally, a stronger model would critique a weaker one)
- No email sending capability - generates drafts only
- Requires Groq API key (free at console.groq.com)

## Tech Stack

- **LangGraph**: Agent orchestration, state management, and graph cycles
- **LangChain**: LLM integration and message handling
- **Groq + Llama 3.3 70B**: Fast, free LLM inference
- **pytest**: Testing framework (18 test cases with mocked LLM)
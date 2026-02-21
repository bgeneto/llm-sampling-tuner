"""
Planner Mode Test Prompts
=========================
Categories:
  1. Architecture / System Design (novel problem decomposition)
  2. Debugging Strategy (plan to diagnose an unknown issue)
  3. Feature Planning (break down a feature into implementation steps)
  4. Refactoring Strategy (plan safe restructuring of existing code)
  5. Edge Case Analysis (enumerate and plan for failure modes)

Each prompt is designed to test the model's ability to:
  - Think structurally and hierarchically
  - Produce actionable, ordered steps
  - Avoid hallucinating tools/APIs/constraints
  - Stay focused without tangential rambling
"""

PLANNER_PROMPTS = [
    # ── 1. Architecture / System Design ──
    {
        "id": "plan_arch_01",
        "category": "architecture",
        "difficulty": "hard",
        "prompt": """You are a technical planner. Create a detailed implementation plan for:

A real-time collaborative code editor (like VS Code Live Share) that works in the browser.
It must support: concurrent editing by 5+ users, syntax highlighting, a shared terminal,
and conflict resolution. The backend should use WebSockets.

Produce a structured plan with:
1. Component breakdown
2. Data flow diagram (text-based)
3. Key technical decisions with tradeoffs
4. Implementation order with dependencies
5. Risk assessment

Do NOT write code. Only produce the plan.""",
        "eval_notes": "Should have CRDT or OT mention, clear component list, dependency ordering.",
    },

    {
        "id": "plan_arch_02",
        "category": "architecture",
        "difficulty": "medium",
        "prompt": """You are a technical planner. Design an architecture plan for:

A plugin system for a CLI tool (like ESLint or Prettier). Plugins should be:
- Discoverable (npm registry or local paths)
- Sandboxed (can't crash the host)
- Composable (plugins can depend on other plugins)
- Hot-reloadable during development

Produce: component list, lifecycle hooks, dependency resolution strategy,
error isolation approach, and a phased implementation plan.""",
        "eval_notes": "Should address sandboxing (VM2/isolate), dependency DAG, lifecycle hooks.",
    },

    # ── 2. Debugging Strategy ──
    {
        "id": "plan_debug_01",
        "category": "debugging",
        "difficulty": "hard",
        "prompt": """You are a senior debugging consultant. A production Node.js API is experiencing:
- 3x increase in p99 latency over 2 weeks (from 200ms to 600ms)
- Memory usage climbing ~50MB/day, never reclaimed
- No code deployments in that period
- PostgreSQL connection pool occasionally exhausts

Create a systematic diagnostic plan. Include:
1. Immediate triage steps
2. Data collection strategy (what metrics, logs, traces to gather)
3. Hypothesis tree (possible root causes ranked by likelihood)
4. Elimination strategy for each hypothesis
5. Rollback/mitigation plan while investigating

Do NOT guess the answer. Produce only the investigation plan.""",
        "eval_notes": "Should mention heap snapshots, connection pool monitoring, GC analysis, query plans.",
    },

    {
        "id": "plan_debug_02",
        "category": "debugging",
        "difficulty": "medium",
        "prompt": """You are a debugging planner. A React application has this bug:
- Users report that after navigating away from a page and back, form data is lost
- This only happens in production, not in development
- The app uses React Router v6 and Redux Toolkit
- The issue started after upgrading React from 18.2 to 18.3

Create a systematic debugging plan with hypotheses, test steps, and elimination criteria.""",
        "eval_notes": "Should consider StrictMode double-render, router remounting, Redux state persistence.",
    },

    # ── 3. Feature Planning ──
    {
        "id": "plan_feat_01",
        "category": "feature",
        "difficulty": "medium",
        "prompt": """Plan the implementation of an undo/redo system for a drawing application.

Context:
- Canvas-based drawing app (HTML5 Canvas)
- Supports: freehand draw, shapes, text, layers, fill
- Currently has NO undo system
- Must support: undo, redo, undo history branching (tree, not stack)

Produce a plan covering:
1. Data model for the history tree
2. Command pattern design
3. Memory management strategy (can't store infinite states)
4. UI integration points
5. Edge cases (mid-stroke undo, layer deletion undo)
6. Testing strategy""",
        "eval_notes": "Should describe command pattern, tree vs linear stack, snapshot vs delta approach.",
    },

    {
        "id": "plan_feat_02",
        "category": "feature",
        "difficulty": "hard",
        "prompt": """Plan a migration strategy for converting a large Express.js REST API (50+ endpoints)
to use a type-safe approach with runtime validation.

Current state: No TypeScript, no validation library, raw req.body usage.
Target state: Full TypeScript, Zod schemas for all I/O, generated OpenAPI docs.

Constraints:
- Must be done incrementally (can't stop feature work)
- Zero downtime required
- 3 developers available part-time

Produce a phased migration plan with risk analysis and rollback strategy at each phase.""",
        "eval_notes": "Should have incremental approach, dual-stack period, schema-first strategy.",
    },

    # ── 4. Refactoring Strategy ──
    {
        "id": "plan_refactor_01",
        "category": "refactoring",
        "difficulty": "medium",
        "prompt": """Plan a refactoring of a God Object anti-pattern.

The class `AppManager` (2,500 lines) handles:
- User authentication
- Database connections
- Email sending
- File uploads
- Cron job scheduling
- Logging configuration
- Cache management

It's used in 85+ files across the codebase. Plan how to safely decompose it
into focused modules without breaking anything. Include dependency analysis,
extraction order, and testing gates between phases.""",
        "eval_notes": "Should identify extraction order by coupling, suggest facade/adapter pattern for migration.",
    },

    # ── 5. Edge Case Analysis ──
    {
        "id": "plan_edge_01",
        "category": "edge_cases",
        "difficulty": "hard",
        "prompt": """You are designing a rate limiter for a multi-tenant SaaS API.

Enumerate and plan for ALL edge cases including:
- Clock skew across distributed nodes
- Token bucket vs sliding window tradeoffs
- Burst handling vs sustained rate
- Tenant isolation failures
- Race conditions in distributed counters
- Graceful degradation when Redis is down
- API key rotation during active rate limit windows

For each edge case: describe the failure mode, its impact, and your mitigation strategy.
Organize as a structured risk matrix.""",
        "eval_notes": "Should be comprehensive, structured as matrix, address distributed consensus.",
    },

    {
        "id": "plan_edge_02",
        "category": "edge_cases",
        "difficulty": "medium",
        "prompt": """Analyze all failure modes for a file upload pipeline:

User → Browser → Presigned S3 URL → S3 → Lambda trigger → Thumbnail generation → CDN

For each stage, enumerate:
1. What can go wrong
2. How you'd detect it
3. How you'd recover
4. What the user sees

Present as a structured failure mode table.""",
        "eval_notes": "Should cover: timeout, partial upload, Lambda cold start, CDN invalidation, presigned URL expiry.",
    },
]

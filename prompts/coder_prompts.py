"""
Coder Mode Test Prompts
=======================
Categories:
  1. Algorithm Implementation (novel problem, correct solution required)
  2. System Code (real-world patterns: APIs, DB, file I/O)
  3. Bug Fixing (given broken code, produce a fix)
  4. Code Generation from Spec (translate requirements to code)
  5. Refactoring (transform existing code while preserving behavior)

Each prompt tests:
  - Syntactic correctness (must parse/compile)
  - Semantic correctness (must solve the actual problem)
  - Code quality (clean, idiomatic, no unnecessary complexity)
  - Hallucination resistance (no made-up APIs or libraries)
  - Instruction following (does exactly what asked)
"""

CODER_PROMPTS = [
    # ── 1. Algorithm Implementation ──
    {
        "id": "code_algo_01",
        "category": "algorithm",
        "difficulty": "hard",
        "language": "python",
        "prompt": """Write a Python function that implements an LRU cache with O(1) get and put operations.

Requirements:
- `class LRUCache(capacity: int)`
- `get(key: int) -> int` returns -1 if not found
- `put(key: int, value: int)` evicts least recently used when at capacity
- Must be O(1) for both operations
- Do NOT use `functools.lru_cache` or `collections.OrderedDict`
- Implement using a doubly linked list + hash map

Include a `__repr__` that shows the cache contents in MRU→LRU order.
Include 5 test cases that verify correctness including eviction behavior.""",
        "eval_notes": "Must have DLL + dict, O(1) complexity, working eviction, valid test cases.",
        "has_verifiable_output": True,
        "reference_tests": """
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)
assert cache.get(2) == -1
cache.put(4, 4)
assert cache.get(1) == -1
assert cache.get(3) == 3
assert cache.get(4) == 4

cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
cache.put(1, 10)
cache.put(3, 3)
assert cache.get(1) == 10
assert cache.get(2) == -1
""",
    },
    {
        "id": "code_algo_02",
        "category": "algorithm",
        "difficulty": "hard",
        "language": "python",
        "prompt": """Implement Dijkstra's shortest path algorithm for a weighted directed graph.

Requirements:
- Input: adjacency list as `dict[str, list[tuple[str, float]]]` (node → [(neighbor, weight)])
- Function signature: `def dijkstra(graph, start, end) -> tuple[float, list[str]]`
- Returns (total_cost, path_as_list_of_nodes) or (float('inf'), []) if no path
- Must use a min-heap (heapq)
- Handle: disconnected nodes, self-loops, negative weight detection (raise ValueError)

Include 4 test cases: simple path, no path, single node, negative weight error.""",
        "eval_notes": "Must use heapq, reconstruct path, handle edge cases, raise on negative weights.",
        "has_verifiable_output": True,
        "reference_tests": """
graph = {
    "A": [("B", 1.0), ("C", 5.0)],
    "B": [("C", 1.0), ("D", 3.0)],
    "C": [("D", 1.0)],
    "D": [],
}
cost, path = dijkstra(graph, "A", "D")
assert cost == 3.0
assert path == ["A", "B", "C", "D"]

cost, path = dijkstra({"A": [("B", 1.0)], "B": [], "C": []}, "A", "C")
assert cost == float("inf")
assert path == []

cost, path = dijkstra({"A": []}, "A", "A")
assert cost == 0
assert path == ["A"]

try:
    dijkstra({"A": [("B", -1.0)], "B": []}, "A", "B")
    raise AssertionError("expected ValueError for negative weight")
except ValueError:
    pass
""",
    },
    {
        "id": "code_algo_03",
        "category": "algorithm",
        "difficulty": "medium",
        "language": "python",
        "prompt": """Write a function that merges K sorted lists into one sorted list.

Signature: `def merge_k_sorted(lists: list[list[int]]) -> list[int]`

Requirements:
- Must use a min-heap approach (not just flatten+sort)
- Handle empty lists in the input
- Handle completely empty input
- Time complexity must be O(N log K) where N = total elements, K = number of lists

Include 5 test cases including edge cases.""",
        "eval_notes": "Must use heap with list index tracking, handle empties, correct complexity.",
        "has_verifiable_output": True,
        "reference_tests": """
assert merge_k_sorted([[1, 4, 5], [1, 3, 4], [2, 6]]) == [1, 1, 2, 3, 4, 4, 5, 6]
assert merge_k_sorted([]) == []
assert merge_k_sorted([[], []]) == []
assert merge_k_sorted([[1], [], [-1, 0, 2]]) == [-1, 0, 1, 2]
assert merge_k_sorted([[1, 1], [1]]) == [1, 1, 1]
""",
    },
    # ── 2. System Code ──
    {
        "id": "code_sys_01",
        "category": "system",
        "difficulty": "medium",
        "language": "python",
        "prompt": """Write an async Python HTTP retry wrapper using aiohttp.

Requirements:
- `async def fetch_with_retry(url, max_retries=3, backoff_factor=0.5, timeout=10)`
- Retries on: 429, 500, 502, 503, 504 status codes
- Exponential backoff: wait = backoff_factor * (2 ** attempt)
- Respects Retry-After header if present on 429
- Raises after max_retries exhausted
- Returns the response JSON on success
- Logs each retry attempt with attempt number and wait time

Use only `aiohttp` and `asyncio` (standard library + aiohttp only).
Include type hints throughout.""",
        "eval_notes": "Must handle Retry-After, exponential backoff math, proper async/await, type hints.",
        "has_verifiable_output": False,
    },
    {
        "id": "code_sys_02",
        "category": "system",
        "difficulty": "hard",
        "language": "python",
        "prompt": """Implement a simple task queue with worker pool using only Python standard library.

Requirements:
- `class TaskQueue(num_workers: int)`
- `submit(fn, *args, **kwargs) -> Future` - submit a callable
- `shutdown(wait=True)` - graceful shutdown
- Workers pull from a thread-safe queue
- Future objects support `.result(timeout=None)` and `.done()`
- Handle worker crashes gracefully (log error, worker continues with next task)
- NOT using concurrent.futures (implement from scratch with threading + queue)

Include a demo showing 10 tasks submitted to 3 workers with mixed success/failure.""",
        "eval_notes": "Must use threading.Thread + queue.Queue, custom Future class, error isolation.",
        "has_verifiable_output": True,
        "reference_tests": """
def _explode():
    raise ValueError("boom")

task_queue = TaskQueue(2)
future_one = task_queue.submit(lambda x: x + 1, 1)
future_err = task_queue.submit(_explode)
future_two = task_queue.submit(lambda: "ok")

assert future_one.result(timeout=2) == 2
try:
    future_err.result(timeout=2)
    raise AssertionError("expected task failure to propagate")
except Exception:
    pass
assert future_two.result(timeout=2) == "ok"
assert future_two.done() is True
task_queue.shutdown(wait=True)
""",
    },
    # ── 3. Bug Fixing ──
    {
        "id": "code_fix_01",
        "category": "bugfix",
        "difficulty": "medium",
        "language": "python",
        "prompt": """Fix all bugs in this code. Explain each bug briefly.

```python
import threading

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance
        self.lock = threading.Lock()

    def deposit(self, amount):
        self.lock.acquire()
        new_balance = self.balance + amount
        self.balance = new_balance
        self.lock.release()
        return new_balance

    def withdraw(self, amount):
        self.lock.acquire()
        if self.balance >= amount:
            new_balance = self.balance - amount
            self.balance = new_balance
            self.lock.release()
            return new_balance
        return -1  # insufficient funds

    def transfer(self, other, amount):
        self.lock.acquire()
        other.lock.acquire()
        withdrawn = self.withdraw(amount)
        if withdrawn != -1:
            other.deposit(amount)
        other.lock.release()
        self.lock.release()

# Test
a = BankAccount(1000)
b = BankAccount(500)
a.transfer(b, 200)
print(f"A: {a.balance}, B: {b.balance}")
```

Provide the corrected version with all bugs fixed.""",
        "eval_notes": "Bugs: withdraw doesn't release lock on failure, transfer causes deadlock (nested locks + withdraw re-acquires), no context managers.",
        "has_verifiable_output": True,
        "reference_tests": """
account_a = BankAccount(100)
account_b = BankAccount(50)

assert account_a.withdraw(200) == -1
assert account_a.deposit(25) == 125

account_a.transfer(account_b, 25)
assert account_a.balance == 100
assert account_b.balance == 75

assert account_a.withdraw(100) == 0
""",
    },
    {
        "id": "code_fix_02",
        "category": "bugfix",
        "difficulty": "hard",
        "language": "python",
        "prompt": """This async code has multiple concurrency bugs. Find and fix ALL of them.
Explain each bug.

```python
import asyncio

class AsyncCache:
    def __init__(self):
        self.cache = {}
        self.pending = {}

    async def get_or_compute(self, key, compute_fn):
        if key in self.cache:
            return self.cache[key]

        if key in self.pending:
            return await self.pending[key]

        self.pending[key] = compute_fn(key)
        try:
            result = await self.pending[key]
            self.cache[key] = result
            return result
        finally:
            del self.pending[key]

    async def invalidate(self, key):
        if key in self.cache:
            del self.cache[key]

    async def get_many(self, keys, compute_fn):
        results = []
        for key in keys:
            result = await self.get_or_compute(key, compute_fn)
            results.append(result)
        return results

async def main():
    cache = AsyncCache()

    async def slow_compute(key):
        await asyncio.sleep(1)
        return f"value_{key}"

    # This should be fast for duplicate keys
    results = await cache.get_many(["a", "b", "a", "c", "b"], slow_compute)
    print(results)

asyncio.run(main())
```""",
        "eval_notes": "Bugs: race condition on pending (multiple awaits on same coroutine), get_many is sequential not concurrent, pending stores coroutine not Task/Future, invalidate during pending compute.",
        "has_verifiable_output": True,
        "reference_tests": """
import asyncio

async def _reference_test_async_cache():
    cache = AsyncCache()
    call_log = []

    async def dedup_compute(key):
        call_log.append(key)
        await asyncio.sleep(0.01)
        return f"value_{key}"

    results = await asyncio.gather(
        cache.get_or_compute("a", dedup_compute),
        cache.get_or_compute("a", dedup_compute),
    )
    assert results == ["value_a", "value_a"]
    assert call_log == ["a"]
    assert await cache.get_or_compute("a", dedup_compute) == "value_a"
    assert call_log == ["a"]

    parallel_started = []
    both_started = asyncio.Event()

    async def parallel_compute(key):
        parallel_started.append(key)
        if len(parallel_started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.2)
        return key.upper()

    results = await cache.get_many(["x", "y"], parallel_compute)
    assert results == ["X", "Y"]

    gate = asyncio.Event()

    async def slow_compute(key):
        await gate.wait()
        return f"done_{key}"

    pending_task = asyncio.create_task(cache.get_or_compute("z", slow_compute))
    await asyncio.sleep(0)
    await cache.invalidate("z")
    gate.set()
    assert await pending_task == "done_z"

asyncio.run(_reference_test_async_cache())
""",
    },
    # ── 4. Code from Spec ──
    {
        "id": "code_spec_01",
        "category": "from_spec",
        "difficulty": "medium",
        "language": "python",
        "prompt": """Implement a simple expression evaluator from this spec:

GRAMMAR:
  expr     → term (('+' | '-') term)*
  term     → factor (('*' | '/') factor)*
  factor   → NUMBER | '(' expr ')' | '-' factor
  NUMBER   → [0-9]+ ('.' [0-9]+)?

REQUIREMENTS:
- `def evaluate(expression: str) -> float`
- Recursive descent parser (no eval/exec/ast)
- Handle: operator precedence, parentheses, unary minus, decimals
- Raise `ValueError` with descriptive message on invalid input
- Examples: "3 + 4 * 2" → 11.0, "-(3 + 4) * 2" → -14.0

Include 8 test cases covering all grammar rules and error cases.""",
        "eval_notes": "Must implement recursive descent, handle precedence correctly, no eval() cheating.",
        "has_verifiable_output": True,
        "reference_tests": """
assert abs(evaluate("3 + 4 * 2") - 11.0) < 1e-9
assert abs(evaluate("-(3 + 4) * 2") - -14.0) < 1e-9
assert abs(evaluate("2.5 * 4") - 10.0) < 1e-9
assert abs(evaluate("((1 + 2) * 3) / 2") - 4.5) < 1e-9

for invalid in ("2 +", "1 / )", "(1 + 2"):
    try:
        evaluate(invalid)
        raise AssertionError(f"expected ValueError for {invalid!r}")
    except ValueError:
        pass
""",
    },
    {
        "id": "code_spec_02",
        "category": "from_spec",
        "difficulty": "hard",
        "language": "python",
        "prompt": """Implement a simple event emitter/pub-sub system with these exact features:

```
class EventEmitter:
    on(event: str, handler: Callable) -> Callable  # returns unsubscribe function
    once(event: str, handler: Callable) -> Callable  # auto-removes after first call
    emit(event: str, *args, **kwargs) -> int  # returns number of handlers called
    off(event: str, handler: Callable) -> bool  # returns True if handler was found
    listeners(event: str) -> list[Callable]  # returns copy of handler list
    wait(event: str, timeout: float = None) -> asyncio.Future  # async wait for event
```

Requirements:
- Thread-safe for on/off/emit (not async, use threading.Lock)
- `wait()` returns a Future that resolves with the args from the next emit
- `wait()` with timeout raises asyncio.TimeoutError
- `once` handlers fire exactly once then auto-remove
- `emit` during `emit` (re-entrant) must work correctly

Include comprehensive tests for all features including re-entrancy.""",
        "eval_notes": "Must handle re-entrancy (iterate copy), thread safety, async wait integration.",
        "has_verifiable_output": True,
        "reference_tests": """
import asyncio

emitter = EventEmitter()
received = []

def handler(value):
    received.append(("on", value))

unsubscribe = emitter.on("data", handler)
assert callable(unsubscribe)
listeners_snapshot = emitter.listeners("data")
listeners_snapshot.clear()
assert len(emitter.listeners("data")) == 1
assert emitter.emit("data", 3) == 1
assert received == [("on", 3)]
assert emitter.off("data", handler) is True
assert emitter.emit("data", 4) == 0

once_received = []
emitter.once("once", lambda value: once_received.append(value))
assert emitter.emit("once", 1) == 1
assert emitter.emit("once", 2) == 0
assert once_received == [1]

reentrant = []
def child():
    reentrant.append("child")
def parent():
    reentrant.append("parent")
    emitter.emit("child")
emitter.on("child", child)
emitter.on("parent", parent)
assert emitter.emit("parent") == 1
assert reentrant == ["parent", "child"]

async def _reference_test_event_emitter():
    waiter = emitter.wait("ready", timeout=0.2)

    async def emit_later():
        await asyncio.sleep(0.01)
        emitter.emit("ready", 1, 2)

    asyncio.create_task(emit_later())
    payload = await waiter
    if payload == ((1, 2), {}):
        normalized = (1, 2)
    elif isinstance(payload, tuple) and len(payload) == 1 and isinstance(payload[0], tuple):
        normalized = payload[0]
    elif isinstance(payload, list):
        normalized = tuple(payload)
    elif isinstance(payload, tuple):
        normalized = payload
    else:
        normalized = (payload,)
    assert normalized == (1, 2)

    timed_out = False
    try:
        await emitter.wait("never", timeout=0.01)
    except asyncio.TimeoutError:
        timed_out = True
    assert timed_out

asyncio.run(_reference_test_event_emitter())
""",
    },
    # ── 5. Refactoring ──
    {
        "id": "code_refactor_01",
        "category": "refactoring",
        "difficulty": "medium",
        "language": "python",
        "prompt": """Refactor this code to eliminate the code smells while preserving exact behavior:

```python
def process_order(order):
    if order['type'] == 'digital':
        if order['status'] == 'pending':
            price = order['base_price']
            if order.get('coupon'):
                if order['coupon']['type'] == 'percent':
                    price = price * (1 - order['coupon']['value'] / 100)
                elif order['coupon']['type'] == 'fixed':
                    price = price - order['coupon']['value']
                    if price < 0:
                        price = 0
            tax = price * 0.10
            total = price + tax
            order['total'] = total
            order['status'] = 'processed'
            order['delivery'] = 'email'
            send_email(order['customer_email'], f"Your digital order total: ${total:.2f}")
            return order
        elif order['status'] == 'processed':
            return order
        else:
            raise ValueError(f"Invalid status for digital order: {order['status']}")
    elif order['type'] == 'physical':
        if order['status'] == 'pending':
            price = order['base_price']
            if order.get('coupon'):
                if order['coupon']['type'] == 'percent':
                    price = price * (1 - order['coupon']['value'] / 100)
                elif order['coupon']['type'] == 'fixed':
                    price = price - order['coupon']['value']
                    if price < 0:
                        price = 0
            weight = order.get('weight', 0)
            if weight < 1:
                shipping = 5.99
            elif weight < 5:
                shipping = 9.99
            elif weight < 20:
                shipping = 14.99
            else:
                shipping = 24.99
            tax = price * 0.10
            total = price + tax + shipping
            order['total'] = total
            order['shipping'] = shipping
            order['status'] = 'processed'
            order['delivery'] = 'shipping'
            send_email(order['customer_email'], f"Your order total: ${total:.2f} (shipping: ${shipping:.2f})")
            return order
        elif order['status'] == 'processed':
            return order
        else:
            raise ValueError(f"Invalid status for physical order: {order['status']}")
    else:
        raise ValueError(f"Unknown order type: {order['type']}")
```

Refactor to remove duplication and deep nesting. Use strategy pattern or similar.
Preserve exact behavior for all code paths. Include a brief explanation of changes.""",
        "eval_notes": "Should extract: apply_coupon, calc_shipping, separate digital vs physical strategies.",
        "has_verifiable_output": False,
    },
]

"""Structural prompt-authority classification (S-35a0; issue #34, parent #31).

S-14/S-15 decide what an Agent may READ. This module decides what AUTHORITY the
containers of a model payload have: who is entitled to direct the agent. It
changes nothing the model sees - S-35a0 only computes the classification so a
later slice (S-35a1) can present it. It grants and denies nothing.

The five classes, from highest to lowest authority:

  instruction       the system message (operator prompt plus Core contracts)
  runtime           Core control fields whose CURRENT value Core computed
  caller_input      supplied by the caller of the run; Core does not vouch for it
  delegated_task    a work request written by a MODEL and handed to another session
  non_authoritative everything else (Knowledge, tool observations, earlier model
                    output, memory, observations...)

`non_authoritative` means "not entitled to direct the agent", not "malicious".

Rules (issue #31 §5.2). Inputs are structural and runtime-controlled ONLY; the
content of a value is never inspected.

  R1  Total and conservative: every key gets a class, unknown or ambiguous data is
      `non_authoritative`, and nothing here raises into the runtime.
  R2  A Core field NAME is not proof of Core ownership of its current value. A name
      is `runtime` only when Core overwrites it before every model call, or the
      current execution witnessed the write. A caller-supplied value that shares a
      Core field name is never `runtime` by name.
  R3  A mixed container takes the lowest applicable authority. Nested paths inherit
      their top-level key; a size-bounded replacement is the same container.
  R4  Orchestrator: caller keys are `caller_input`; keys written by any step
      (including the `final_output` merge) are `non_authoritative` even when they
      overwrite a caller key; an input-mapping target inherits the class of the
      root of its source path.
  R5  GAME: see the tables below. A delegated session (a `GameDelegationRun` row
      points at it - never `source_label`) is `delegated_task` for its goal and
      every key of its `initial_context`.
  R6  Direct runner callers have no session: everything is `caller_input` except
      `knowledge_context`.
  R7  Nothing is persisted. Classification is recomputed from structural facts, so
      it works for fresh runs, resumed runs and sessions that predate this module,
      and it does NOT claim exact historical provenance.

Known limitations: a host may place end-user or retrieved text in
`initial_context`; Core cannot tell and labels it `caller_input`. Source origin,
per-call `input_origins` and exposure are NOT tracked here (S-35c).
"""

INSTRUCTION = "instruction"
RUNTIME = "runtime"
CALLER_INPUT = "caller_input"
DELEGATED_TASK = "delegated_task"
NON_AUTHORITATIVE = "non_authoritative"

#: Highest authority first.
AUTHORITY_CLASSES = (INSTRUCTION, RUNTIME, CALLER_INPUT, DELEGATED_TASK, NON_AUTHORITATIVE)
_RANK = {name: index for index, name in enumerate(AUTHORITY_CLASSES)}

KNOWLEDGE_CONTEXT_KEY = "knowledge_context"

# --- GAME field tables (issue #31 §5.2) ---------------------------------------
# Table A: overwritten by Core before every model call. A caller value of the
# same name never reaches the model. Timing verified in `execution_runner`:
# `_restore_game_reserved_payload` restores GAME_RESERVED_PAYLOAD_KEYS on every
# payload, and `_update_game_context` writes the others before the first
# iteration and after each one.
GAME_RUNTIME_FIELDS = frozenset(
    {
        "iteration",
        "max_iterations",
        "available_actions",
        "game_policy",
        "game_response_contract",
        # Core constants, written before iteration 1 and after every iteration.
        "response_contract",
        "game_action_dispatch_enabled",
        "finish_reason",
    }
)
GAME_GOAL_FIELDS = frozenset({"goal", "goal_text"})
#: Core-assembled containers holding model-, tool-, Knowledge-, human- or
#: caller-originated content. `final_answer` is model-derived after iteration 1.
GAME_NON_AUTHORITATIVE_FIELDS = frozenset(
    {
        "memory",
        "scoped_memory",
        "observations",
        "previous_response",
        "final_answer",
        KNOWLEDGE_CONTEXT_KEY,
    }
)
GAME_TABLE_A_FIELDS = GAME_RUNTIME_FIELDS | GAME_GOAL_FIELDS | GAME_NON_AUTHORITATIVE_FIELDS

# Table B: written by Core only AFTER a condition or outcome. `runtime` only when
# the current execution witnessed the write (`core_written`); otherwise the value
# is the caller's if the caller supplied that name, else unknown.
GAME_CONDITIONAL_FIELDS = frozenset(
    {
        "execution_outcome",  # after a run segment ends
        "goal_outcome",  # after a run segment ends
        "waiting_reason",  # after a run segment ends
        "failed_iteration",  # exception path only
        "last_error",  # exception path only
    }
)
#: Formatted exception text can embed tool- or model-derived substrings (mixed),
#: so even a Core-written value is not `runtime`.
GAME_CONDITIONAL_MIXED_FIELDS = frozenset({"last_error"})
#: Written by `resume_goal_execution` from the caller's argument.
GAME_CALLER_SUPPLIED_FIELDS = frozenset({"continuation_payload"})


def lowest(*classes) -> str:
    """The lowest applicable authority. Empty or unknown input is non-authoritative."""
    result = None
    for name in classes:
        rank = _RANK.get(name, _RANK[NON_AUTHORITATIVE])
        if result is None or rank > result:
            result = rank
    if result is None:
        return NON_AUTHORITATIVE
    return AUTHORITY_CLASSES[result]


def authority_of_path(authority_map: dict, path) -> str:
    """Nested paths inherit the class of their top-level key (R3)."""
    root = str(path).split(".")[0]
    return (authority_map or {}).get(root, NON_AUTHORITATIVE)


def session_is_delegated(session) -> bool:
    """True iff a persisted `GameDelegationRun` points at this session (R5).

    Deliberately NOT `source_label`: that is a caller-supplied string.
    """
    pk = getattr(session, "pk", None)
    if pk is None:
        return False
    from ai_hub.models import GameDelegationRun

    return GameDelegationRun.objects.filter(delegated_session_id=pk).exists()


def _mapping_root(input_mapping, key):
    """Root key a payload key was copied from, or None when it was not mapped."""
    if not isinstance(input_mapping, dict) or not input_mapping:
        return key
    if key not in input_mapping:
        return None
    return str(input_mapping[key]).split(".")[0]


def classify_direct_payload(payload) -> dict:
    """R6: a direct runner caller has no session. Everything is caller input
    except `knowledge_context`, which Core assembles from Knowledge."""
    result = {}
    for key in _keys(payload):
        result[key] = NON_AUTHORITATIVE if key == KNOWLEDGE_CONTEXT_KEY else CALLER_INPUT
    return result


def classify_orchestrator_payload(
    payload,
    *,
    input_mapping=None,
    caller_keys=frozenset(),
    step_written_keys=frozenset(),
) -> dict:
    """R4. `caller_keys` is the key set of the session's `initial_context`;
    `step_written_keys` is the execution-local set of context keys any step
    output (or the final_output merge) has written. The writer wins: a caller key
    a step overwrote is `non_authoritative`."""
    caller_keys = _keyset(caller_keys)
    step_written_keys = _keyset(step_written_keys)
    result = {}
    for key in _keys(payload):
        if key == KNOWLEDGE_CONTEXT_KEY:
            result[key] = NON_AUTHORITATIVE
            continue
        root = _mapping_root(input_mapping, key)
        if root is None or root in step_written_keys:
            result[key] = NON_AUTHORITATIVE
        elif root in caller_keys:
            result[key] = CALLER_INPUT
        else:
            result[key] = NON_AUTHORITATIVE
    return result


def _game_source_class(root, *, caller_keys, delegated, core_written) -> str:
    """Authority of the value currently stored under context key `root`."""
    if root in GAME_GOAL_FIELDS:
        return DELEGATED_TASK if delegated else CALLER_INPUT
    if root in GAME_RUNTIME_FIELDS:
        return RUNTIME
    if root in GAME_NON_AUTHORITATIVE_FIELDS:
        return NON_AUTHORITATIVE
    if root in GAME_CONDITIONAL_FIELDS:
        if root in core_written:
            return NON_AUTHORITATIVE if root in GAME_CONDITIONAL_MIXED_FIELDS else RUNTIME
        if root in caller_keys:
            return DELEGATED_TASK if delegated else CALLER_INPUT
        return NON_AUTHORITATIVE
    if root in GAME_CALLER_SUPPLIED_FIELDS:
        return CALLER_INPUT
    if root in caller_keys:
        return DELEGATED_TASK if delegated else CALLER_INPUT
    return NON_AUTHORITATIVE


def classify_game_payload(
    payload,
    *,
    input_mapping=None,
    caller_keys=frozenset(),
    delegated=False,
    core_written=frozenset(),
) -> dict:
    """R2/R5. `payload` is the prepared payload sent to the model.

    `core_written` is the set of Table B names the CURRENT execution itself wrote
    before this model call. Table B names are written only after a run segment
    ends or on failure, so at every model call of the runner it is empty: a
    resumed run does not witness the previous segment's writes, and a value found
    under a Table B name is therefore never elevated to `runtime` by name.

    Reserved names (Table A) keep their own class even when an input mapping
    renamed or dropped them, because Core restores them on every payload.
    """
    caller_keys = _keyset(caller_keys)
    core_written = _keyset(core_written)
    result = {}
    for key in _keys(payload):
        if key in GAME_TABLE_A_FIELDS:
            result[key] = _game_source_class(
                key, caller_keys=caller_keys, delegated=delegated, core_written=core_written
            )
            continue
        root = _mapping_root(input_mapping, key)
        if root is None:
            result[key] = NON_AUTHORITATIVE
        else:
            result[key] = _game_source_class(
                root, caller_keys=caller_keys, delegated=delegated, core_written=core_written
            )
    return result


def _keys(payload):
    if not isinstance(payload, dict):
        return []
    return [str(key) for key in payload]


def _keyset(values):
    try:
        return frozenset(str(value) for value in (values or ()))
    except TypeError:
        return frozenset()

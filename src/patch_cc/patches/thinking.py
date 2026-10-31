"""Make thinking blocks visible in the normal UI instead of transcript-only.

Three gates hide a finished thinking block in the normal view, and all three
must go or the patch silently does nothing (which is exactly how it shipped
broken in 0.1.0):

1. **Grouping.** The activity-group builder swallows every thinking-first
   assistant message into the collapsed "✻ ... (thought for Ns)" group, so the
   message never reaches the renderer at all. This was the missed gate: the
   two rewrites below applied cleanly to a switch arm that thinking text never
   flows through.
2. **The null guard.** The ``case"thinking":`` arm returns null outside
   transcript/verbose mode.
3. **Presentation.** The renderer collapses the text to one italic line unless
   ``isTranscriptMode`` is set.

All three rewrites are therefore marked *required*: any one of them going
absent is a regression, not a build variant, and `doctor` reports it as such
instead of staying green on the other two.

Its fold (``inline_fold``) keeps a block open while its row is on screen or on
the last screen, and draws it as one line once newer content has pushed the
row out of both.

Rendering is only half of it. A fourth gate is not in the UI at all: the
account's server-side experiment bucket, which decides whether the API puts any
text in the thinking blocks it returns. `thinking-summaries` is that half, and
without it the other three render an empty string perfectly.

`max-effort` lives here too: effort is thinking depth, so the module that owns
how thinking is shown also owns how hard it is asked for.
"""

from __future__ import annotations

from collections.abc import Callable

from .. import js
from ..custom_models import EFFORT_LADDER
from ..js import Edit, Source
from .base import (
    FOLD_NEVER,
    FOLD_TIME,
    FOLDS,
    GROUP_MODELS,
    GROUP_OUTPUT,
    Options,
    Outcome,
    Patch,
    string_setting,
)

# ------------------------------------------------------------------ grouping

#: The property the grouping loop accumulates think-time into. It is what makes
#: one arm of that loop *the thinking arm*: of every branch deciding where an
#: entry goes, exactly one both accounts think-time and absorbs the entry into
#: the open group. Neither fact is a shape, so neither moved when 2.1.232
#: extracted the summary into two helper calls and split a comma-fused `if` --
#: a reshape anything modelling the arm body dies on.
_THINK_TIME = "thoughtForMs"

#: The group's summary line, written from the entry that is being absorbed.
_SUMMARY = "latestThinkingSummary"

#: How an entry joins a group: the list, and the call that adds to it.
_MESSAGES, _PUSH = "messages", "push"

#: Where the summary write is sent instead. The group object is built by one
#: factory and read by one reader, both by name, so an unread property is
#: inert -- and renaming the write is shape-free in a way deleting the
#: statement is not: upstream has spelled it as a comma-fused assignment inside
#: an `if` head *and* as a guarded statement of its own, and this edit is the
#: same single node either way.
_DROPPED = "__ccDroppedSummary"


def _absorbing(group: str, anchor: js.Node) -> Callable[[js.Node], bool]:
    """Predicate: this ``if`` accounts think-time *and* absorbs, in one branch.

    Bounded to the arm whose *consequence* holds the anchor. The climb starts at
    the think-time accounting and looks for the `if` that also absorbs into the
    same group; unbounded, a thinking arm whose absorb had drifted let it walk
    out to an *earlier* arm of the same `else` chain that happens to push into
    the same group local, and that arm's absorb was rewritten instead -- at 1/1,
    with the required step satisfied and thinking still swallowed. An arm the
    anchor sits in the `else` of has the anchor outside its consequence, so it no
    longer qualifies, and a genuinely drifted absorb now finds nothing (a loud,
    required-step failure) rather than the wrong node.
    """

    def want(node: js.Node) -> bool:
        if node.type != "if_statement":
            return False
        consequence = node.child_by_field_name("consequence")
        return (
            consequence is not None
            and consequence.start_byte <= anchor.start_byte < consequence.end_byte
            and _absorbed(consequence, group) is not None
        )

    return want


def _absorbed(arm: js.Node | None, group: str) -> js.Node | None:
    """This arm's ``<group>.messages.push(<entry>)`` -- into the group it accounts.

    Named by the object, not by the call: the group whose think-time this arm
    has just added to is the group it goes on to absorb the entry into, and
    those two facts about one local are what make this *the thinking arm*. A
    push into anything else is somebody else's -- ``.messages.push`` alone took
    the first one in the branch, so a decoy push ahead of the real one was
    rewritten instead of it, at `group-routing` 1/1 with thinking still
    swallowed.

    Of what the arm itself does: a push inside a callback declared here answers
    for the callback.

    Each hop is the read it is, never the chain spelled out: ``.messages`` and
    ``.push`` are fields of two member reads, and the group is a whole node
    compared against a whole node. Written as one string it also asserted the
    operators in between, so a defensive ``<group>.messages?.push`` -- one
    character upstream owes nobody an announcement about -- would read as this
    arm having no absorb at all, and `group-routing` is required.
    """
    return js.only(
        js.every(
            arm,
            lambda n: (
                n.type == "call_expression"
                and js.reads(fn := n.child_by_field_name("function"), _PUSH)
                and js.reads(list_ := js.receiver(fn), _MESSAGES)
                and js.text(js.receiver(list_)) == group
            ),
            scoped=True,
        ),
        f"ways of absorbing an entry into {group}",
    )


def _ungrouped(arm: js.Node) -> js.Node | None:
    """What this dispatch chain does with an entry that joins no group.

    The terminal ``else`` of the chain the arm belongs to -- flush the open
    group, push the entry into the visible list. Reusing it *verbatim* is what
    makes the rewrite independent of how this build spells either: the patch
    does not construct a flush call, it routes the thinking entry down the path
    the loop already has for entries that are not grouped.
    """
    tail = arm
    while (alternative := tail.child_by_field_name("alternative")) is not None:
        inner = alternative.named_children[0] if alternative.named_children else None
        if inner is not None and inner.type == "if_statement":
            tail = inner
            continue
        return alternative.named_children[-1] if alternative.named_children else None
    return None


def _reroute_grouping(source: Source, outcome: Outcome) -> Source:
    """Send finished thinking messages to the visible list, not the group."""
    step = outcome.step("group-routing")
    edits: list[Edit] = []
    seen: set[int] = set()

    for node in source.find(_THINK_TIME):
        accounted = js.up(node, "member_expression")
        if accounted is None or accounted.child_by_field_name("property") != node:
            continue
        # The open group, as this build spells it: what the arm adds think-time
        # to is what it absorbs the entry into.
        group = js.text(accounted.child_by_field_name("object"))
        arm = js.climb(node, _absorbing(group, node))
        if arm is None or arm.id in seen:
            continue
        absorb = _absorbed(arm.child_by_field_name("consequence"), group)
        ungrouped = _ungrouped(arm)
        if absorb is None or ungrouped is None:
            continue
        seen.add(arm.id)
        step.candidates += 1
        edits.append(Edit.replace(absorb, js.text(ungrouped).rstrip(";")))
        edits += [
            Edit.replace(name, _DROPPED)
            for name in js.every(
                arm,
                lambda n: n.type == "property_identifier" and js.text(n) == _SUMMARY,
            )
        ]
        step.applied += 1

    return source.apply(edits)


#: What a message is timed by, as the grouping arm reads it.
_TIMESTAMP = "timestamp"


def _accounting(source: Source, outcome: Outcome, *, folds: bool) -> list[Edit]:
    """Stop the group timing thinking it no longer holds.

    The grouping arm accounts each block's think-time into the open group,
    ``<group>.thoughtForMs += <time>``, just before the block -- routed out of
    the group -- would have joined it. Left standing, the tool group before a
    block read "Thought for 58s" for thinking drawn on its own row beside it.
    So the sum goes (**group-time**) and the time is still worked out, for
    ``folds`` to note under the block (**fold-note**, :func:`_noted`).

    Found on its own, not through the arm `group-routing` identifies: the sum
    is the bundle's one ``+=`` into ``thoughtForMs`` (:func:`js.only`), and the
    block it times is the message its own block reads ``timestamp`` off, which
    is the same object the arm goes on to route: one binding, however often it
    is read, and one that means the same at the sum.
    """
    step = outcome.step("group-time")
    sum_ = js.only(
        [
            read.parent
            for node in source.find(_THINK_TIME)
            if (read := js.up(node, "member_expression")) is not None
            and read.child_by_field_name("property") == node
            and read.parent is not None
            and read.parent.type == "augmented_assignment_expression"
            and read.parent.child_by_field_name("left") == read
        ],
        "sums into the group's think-time",
    )
    time = None if sum_ is None else sum_.child_by_field_name("right")
    if sum_ is None or time is None:
        return []
    step.candidates += 1
    shown = js.text(time)
    if folds:
        note = outcome.step("fold-note")
        note.candidates += 1
        timed = js.only(
            list(
                {
                    frozenset(value.id for value in source.values(message)): message
                    for n in js.every(
                        js.up(sum_, "statement_block"),
                        lambda n: js.reads(n, _TIMESTAMP),
                        scoped=True,
                    )
                    if (message := js.receiver(n)) is not None
                }.values()
            ),
            "messages the sum times",
        )
        if timed is not None and source.portable(timed, sum_):
            shown = _noted(js.text(timed), shown)
            note.applied += 1
    step.applied += 1
    return [Edit.replace(sum_, f"({shown})")]


# ------------------------------------------------------------------ the arm

_THINKING_LABEL = '"thinking"'
_TRANSCRIPT = "isTranscriptMode"
_HIDE_IN_TRANSCRIPT = "hideInTranscript"


def _answers_null(branch: js.Node | None) -> js.Node | None:
    """The ``return null`` a branch answers with, if that is what it does.

    Both minifier statement forms (``if(x)return null;`` and the 2.1.216
    ``if(x){return null}``) are the same tree, so there is no longer a form to
    miss -- which is exactly how this patch silently broke once.

    The branch's *own* statements, never the subtree: an `if` nested inside this
    one is its own gate with its own answer, and reading it as this branch's
    made the two overlap -- one return in a batch twice, which is the corruption
    `Source.apply` refuses. Whatever else the branch does is upstream's and is
    left standing, because what is neutralised here is the answer, not the
    conditional around it: deleting the whole `if` took a statement upstream had
    put before the return with it, at 1/1 and green.
    """
    if branch is None:
        return None
    statements = js.children(branch) if branch.type == "statement_block" else [branch]
    return next(
        (
            node
            for node in statements
            if node.type == "return_statement"
            and [child.type for child in node.named_children] == ["null"]
        ),
        None,
    )


def _hidden(arm: js.Node, transcript: set[str]) -> list[js.Node]:
    """Every early return in this arm that hides the block outside transcript mode.

    Keyed on the flag the arm itself hands the renderer as ``isTranscriptMode``,
    so what is neutralised is the gate this patch is about and not any other
    guard the arm may grow: a defensive ``if(!param)return null`` reads exactly
    like this one to a matcher that asks only what the branch answers, and
    answering that one is a render this build never meant to draw.
    """
    found = []
    # `scoped`: an early return this arm makes, not one a callback nested inside
    # it makes -- a `return null` inside a handler answers for the handler, and
    # neutralising it is a render this arm never owned.
    for gate in js.every(arm, js.of_type("if_statement"), scoped=True):
        if gate.child_by_field_name("alternative") is not None:
            continue
        condition = gate.child_by_field_name("condition")
        if not any(
            js.text(node) in transcript
            for node in js.every(condition, js.of_type("identifier"))
        ):
            continue
        answer = _answers_null(gate.child_by_field_name("consequence"))
        if answer is not None:
            found.append(answer)
    return found


#: What the arm hands its renderer the block as, and what the renderer takes
#: it apart from.
_PARAM = "param"
_THINKING = "thinking"

#: Where the thinking rows that have left the screen are remembered: by the
#: block's ``signature``, which every finished thinking block carries and no
#: live one does, so a live row -- each update is a row of its own, removed as
#: the next arrives -- is never taken for a row that left. Strings, because a
#: block is re-read from the transcript as often as it is drawn, and the
#: running session is all it has to outlive.
_FOLD = "globalThis.__ccThinkingFold"

#: The note itself, made by whichever writer runs first: the signatures whose
#: rows left, each block's think-time, and one fold gate per renderer
#: (:func:`_gated`). The app context the gates read frames from is added to it
#: where upstream names that context (:func:`_fold_frames`).
_FOLD_STATE = "{left:new Set,ms:new Map,gates:new WeakMap}"

#: The prop the gate hands the renderer its ref under, and the local the
#: renderer binds it to: a name no minifier writes.
_REF = "__ccFoldRef"

#: The row's own box: the one the renderer hands its margin to.
_ROW = "marginTop"

#: The collapsed form is upstream's: dim, italic, every line run together.
#: Folded, it is cut to one line, the way upstream draws its own one-line
#: dim text.
_ITALIC = "italic"
_ONE_LINE = '"truncate-end"'


#: The transcript list's own question of a row: may a click expand it? Folded
#: tool calls answer yes, and the click toggles the row into its verbose view,
#: which the thinking renderer draws whole.
_CLICKABLE = "isItemClickable"


def _clickable(test: str) -> str:
    """Upstream's clickable test, with a folded thinking block answering yes too.

    A thinking block is its own message, keyed for expansion by its own uuid,
    so the click expands that block and nothing beside it; a second click folds
    it again. Only a block that has left the screen answers: an open one has
    nothing to expand, and a live one carries no signature.
    """
    return (
        f'((clickable)=>(message)=>{{let block=message?.type==="assistant"'
        f"?message.message?.content?.[0]:void 0;"
        f'return block?.type==="thinking"&&!!{_FOLD}?.left.has(block.signature)'
        f"||clickable(message)}})({test})"
    )


def _fold_click(source: Source, outcome: Outcome) -> list[Edit]:
    """Let a click expand a folded block, the way it expands a folded tool call.

    The transcript list is handed its clickable test under upstream's own name
    (:data:`_CLICKABLE`), and that value is wrapped (:func:`_clickable`): the
    click, the expanded set and the verbose redraw are all upstream's, so the
    patch rides the path folded tool calls take rather than building one. Of
    the bag *passed*, never the list's own pattern taking it apart.
    """
    step = outcome.step("fold-click")
    handed = js.only(
        [
            value
            for node in source.find(_CLICKABLE)
            if (pair := js.named(node)) is not None
            and pair.type == "pair"
            and (value := pair.child_by_field_name("value")) is not None
        ],
        f"values handed as {_CLICKABLE}",
    )
    if handed is None:
        return []
    step.candidates += 1
    step.applied += 1
    return [Edit.replace(handed, _clickable(js.text(handed)))]


def _thinking_of(
    source: Source, scope: js.Node, block: js.Node
) -> list[tuple[js.Node, js.Node]]:
    """Each name this scope binds the ``thinking`` of ``block`` to, with the
    value it is taken off.

    ``block`` is the name a pattern binds the block to, and a pattern takes
    ``thinking`` off it when the value it takes apart can be that binding
    (:meth:`js.Source.values`), however the minifier spells either.
    """
    return [
        (js.binding(js.props(pattern)[_THINKING]), value)
        for pattern in js.every(scope, js.of_type("object_pattern"), scoped=True)
        if _THINKING in js.props(pattern)
        and pattern.parent is not None
        and pattern.parent.type == "variable_declarator"
        and (value := pattern.parent.child_by_field_name("value")) is not None
        and block in source.values(value)
    ]


def _carries(
    source: Source, value: js.Node | None, thought: list[js.Node], seen: set[int]
) -> bool:
    """Can this expression hand on the block's ``thinking``?

    It can when one of the values it can stand for (:meth:`js.Source.values`,
    names followed to what is written to them) is a binding of it, or a call
    handed one: a helper that cleans the text hands back the text, and nothing
    here models which calls do.
    """
    for candidate in source.values(value):
        if candidate in thought:
            return True
        if candidate.type == "call_expression" and candidate.id not in seen:
            seen.add(candidate.id)
            if any(_carries(source, a, thought, seen) for a in js.arguments(candidate)):
                return True
    return False


def _mentions(source: Source, node: js.Node | None, thought: list[js.Node]) -> bool:
    """Does this expression read the block's ``thinking``, by any name?"""
    seen: set[int] = set()
    return (
        js.first(
            node,
            lambda n: n.type == "identifier" and _carries(source, n, thought, seen),
        )
        is not None
    )


#: Where an element's row sits: its top summed up to the content of the
#: nearest box that scrolls (``overflowY: "scroll"``), with that box, its
#: content and the row's height -- upstream's own visibility walk, the one its
#: dialogs and its visible-rows hook use. Nothing where there is no layout or
#: no scrolling box (the classic renderer has none).
_WHERE = (
    "(node)=>{let y=0,at=node;for(;;){let parent=at?.parentNode,box=parent?.parentNode;"
    "if(!at?.yogaNode||!parent?.yogaNode)return;y+=at.yogaNode.getComputedTop();"
    'if(box?.yogaNode&&(box.style?.overflowY??box.style?.overflow)==="scroll")'
    "return{box,content:parent,y,rows:node.yogaNode.getComputedHeight()};at=parent}}"
)

#: Has newer content pushed the row out of the last screen -- the bottom
#: screenful of the content (its height, less the box's), which is what the
#: screen shows at the bottom? A row on it stays open however far up you
#: scroll: what folds a row is content arriving after it, never scrolling away
#: from the end.
_PUSHED = (
    "({box,content,y,rows})=>rows>0&&"
    "y+rows<=content.yogaNode.getComputedHeight()-box.yogaNode.getComputedHeight()"
)

#: Is a row wholly off the screen, given the offset the box scrolls by? The
#: row's top within the box against the box's height.
_OFF = (
    "({box,content,y,rows},top)=>{let shown=content.yogaNode.getComputedTop()+y-top;"
    "return shown+rows<=0||shown>=box.yogaNode.getComputedHeight()}"
)

#: Has a mounted row left for good, as a frame shows it? ``1`` is pushed out
#: of the last screen and wholly off the screen at the offset the box last drew
#: (``scrollTopRendered``). ``0`` is not. ``-1`` is not yet known: no layout,
#: or a scroll still easing in or jumping to an element
#: (``pendingScrollDelta``, ``scrollAnchor``) -- the two states in which
#: upstream's transcript does not hold its view steady, so a fold then would
#: move what is on screen.
_PLACE = (
    "(node)=>{let at=where(node),box=at?.box,top=box?.scrollTopRendered;"
    'if(!at||typeof top!=="number"||box.pendingScrollDelta||box.scrollAnchor)return-1;'
    "return pushed(at)&&off(at,top)?1:0}"
)

#: Did a row the transcript has just unloaded leave for good? Asked once the
#: update that unloaded it has been laid out, of where the row sat before it.
#: The scroll box must still be in the tree (its parents reach Ink's root): a
#: box taken down with everything in it is a different screen, ctrl+o's or a
#: dialog's, and says nothing about the row. Then the row must be pushed out
#: of the last screen and off the screen the box now scrolls to -- the last
#: screen itself while it sticks to the bottom, and its own offset and any
#: scroll still easing in otherwise. Not while it jumps to an element, whose
#: offset is not known yet.
_GONE = (
    "(at)=>{let{box,content}=at,root=box;while(root.parentNode)root=root.parentNode;"
    'if(root.nodeName!=="ink-root"||!content.yogaNode||box.scrollAnchor||!pushed(at))'
    "return!1;if(box.stickyScroll??!!box.attributes?.stickyScroll)return!0;"
    "return off(at,(box.scrollTop??0)+(box.pendingScrollDelta??0))}"
)


def _gated(component: str, factory: str) -> str:
    """The renderer, behind a gate that folds its block once the row has left
    the screen for good.

    The gate is a class component, so it needs nothing from React but the
    class contract React has kept since its first release: a prototype marked
    ``isReactComponent``, the ``updater`` React gives every instance, and the
    ``contextType`` it reads a context through. The context is upstream's app
    context (:func:`_fold_frames`), and from it the gate takes
    ``subscribeFrames``, which upstream calls after every frame it draws -- a
    scroll included -- so the check runs when the screen changes and never
    otherwise. One gate class per renderer, made the first time it draws.

    The gate draws the renderer as the arm asked, with transcript presentation
    while the block is open, and hands it a ref (:data:`_REF`) for its row. A
    finished block whose row a settled frame shows has left (:data:`_PLACE`)
    folds in two commits. The first draws nothing new: every commit has Ink
    measure the screen and upstream's transcript re-anchor its view on the row
    now at its top -- which a row wholly off the screen cannot be. The second,
    from ``componentDidUpdate``, notes the block as left and draws it folded,
    and the transcript, finding its anchor moved up by the rows the fold took,
    scrolls by as much before the frame is drawn: a block folding above the
    screen does not move what is on it. A live block has no signature yet and
    is not watched.

    A frame can miss a row. Output that lands in one update -- a long reply
    arriving whole -- pushes a block past the transcript's mounted window in
    the same update that unloads it, so no frame ever draws it off the screen.
    So a gate being unmounted notes where its row sat, and once that update
    has been laid out asks whether it left for good (:data:`_GONE`). Folding
    an unloaded row moves nothing on screen, so this needs no settled frame.
    """
    return (
        f"((type,jsx)=>{{let fold={_FOLD}??={_FOLD_STATE},gate=fold.gates.get(type);"
        f"if(!gate){{let where={_WHERE},pushed={_PUSHED},off={_OFF},"
        f"place={_PLACE},gone={_GONE};"
        f"gate=function(){{this.foldNode=null;this.foldRef=(node)=>{{this.foldNode=node}}}};"
        f"Object.assign(gate.prototype,{{isReactComponent:{{}},"
        f"componentDidMount(){{this.foldWatch()}},"
        f"componentDidUpdate(){{let s=this.foldArmed;if(s){{this.foldArmed=void 0;"
        f"fold.left.add(s);this.updater.enqueueForceUpdate(this,null);return}}"
        f"this.foldWatch()}},"
        f"componentWillUnmount(){{this.foldStop?.();this.foldStop=void 0;"
        f"let s=this.foldArmed??this.props.param?.signature;"
        f'if(typeof s!=="string"||!s||fold.left.has(s))return;'
        f"if(this.foldArmed){{fold.left.add(s);return}}"
        f"let at=where(this.foldNode);"
        f"if(at)queueMicrotask(()=>{{try{{if(gone(at))fold.left.add(s)}}catch{{}}}})}},"
        f"foldWatch(){{let s=this.props.param?.signature;"
        f'if(this.foldStop||typeof s!=="string"||!s||fold.left.has(s))return;'
        f"this.foldStop=this.context?.subscribeFrames?.(()=>{{try{{"
        f"let s=this.props.param?.signature;"
        f'if(typeof s!=="string"||!s||!fold.left.has(s)&&place(this.foldNode)!==1)return;'
        f"this.foldStop?.();this.foldStop=void 0;if(!fold.left.has(s))this.foldArmed=s;"
        f"this.updater.enqueueForceUpdate(this,null)}}catch{{}}}})}},"
        f"render(){{let p=this.props;return jsx(type,{{...p,isTranscriptMode:"
        f"p.isTranscriptMode||!fold.left.has(p.param?.signature),{_REF}:this.foldRef}})}}}});"
        f"fold.gates.set(type,gate)}}"
        f"gate.contextType??=fold.app;return gate}})({component},{factory})"
    )


#: Upstream's app context, by the name it gives it, and the frame subscription
#: the gate reads off it.
_APP = '"InternalAppContext"'
_FRAMES = "subscribeFrames"


def _fold_frames(source: Source, outcome: Outcome) -> list[Edit]:
    """Leave upstream's app context where the fold gates can read it.

    The context is the object upstream names ``"InternalAppContext"`` (its
    ``displayName``, the bundle's one), and it is the one whose default value
    carries ``subscribeFrames``; both are asked. It lives in the renderer's
    module of Ink, not the thinking renderer's, so it is noted beside the fold
    (:data:`_FOLD`) where it is named, as the module loads.
    """
    step = outcome.step("fold-frames")
    named_ = js.only(
        [
            assignment
            for node in source.find(_APP)
            if node.type == "string"
            and (assignment := node.parent) is not None
            and assignment.type == "assignment_expression"
            and assignment.child_by_field_name("right") == node
            and js.reads(assignment.child_by_field_name("left"), "displayName")
        ],
        f"contexts named {_APP}",
    )
    if named_ is None:
        return []
    step.candidates += 1
    context = js.receiver(named_.child_by_field_name("left"))
    if context is None or context.type != "identifier":
        return []
    made = list(source.values(context))
    statement = named_.parent
    if (
        len(made) != 1
        or made[0].type != "call_expression"
        or _FRAMES not in js.call_props(made[0])
        or statement is None
        or statement.type not in ("expression_statement", "sequence_expression")
    ):
        return []
    step.applied += 1
    return [Edit.after(named_, f",({_FOLD}??={_FOLD_STATE}).app={js.text(context)}")]


def _noted(entry: str, time: str) -> str:
    """The think-time, handed on unchanged after two things are noted under the
    entry's block.

    Its time, for the folded row to say (:func:`_thought`): upstream's own
    figure, cap and all, which the group never shows once its thinking is
    routed out. And, for a block older than this session -- a resumed one --
    that its row has already left the screen, so it opens folded, as a tool
    group does, and only thinking that arrives while you watch starts open. The
    session's age is the process's own (``performance.timeOrigin``), against
    the message's own timestamp. A live entry has no signature yet and is not
    noted.
    """
    return (
        f"((entry,time)=>{{let s=entry?.message?.content?.[0]?.signature;"
        f'if(typeof s==="string"&&s){{let fold={_FOLD}??={_FOLD_STATE};'
        f"fold.ms.set(s,time);"
        f"if(Date.parse(entry.timestamp)<performance.timeOrigin)fold.left.add(s)}}"
        f"return time}})({entry},{time})"
    )


def _thought(block: str) -> str:
    """What a folded block says: how long it was thought for, in whole seconds
    as upstream's spinner words it, or just that it was, with no time noted."""
    return (
        f'((time)=>time>0?"Thought for "+Math.max(1,Math.round(time/1000))+"s"'
        f':"Thought")({_FOLD}?.ms?.get({block}?.signature))'
    )


def _renderers(source: Source) -> list[js.Node]:
    """Every thinking renderer: a function taking a block (``param``) and the
    ``isTranscriptMode`` flag, which then reads ``thinking`` off that block.

    Found on its own, not through the arm `renderer-props` identifies: ten
    functions take both props on 2.1.314, and one of them takes a thinking
    block apart.
    """
    found: dict[int, js.Node] = {}
    for node in source.find(_TRANSCRIPT):
        pattern = js.up(node, "object_pattern")
        if pattern is None or _PARAM not in js.props(pattern):
            continue
        renderer = js.climb(pattern, js.of_type(*js.FUNCTIONS))
        block = js.binding(js.props(pattern)[_PARAM])
        if renderer is not None and _thinking_of(source, renderer, block):
            found[renderer.id] = renderer
    return list(found.values())


def _folding(
    source: Source, renderer: js.Node, outcome: Outcome, *, timed: bool
) -> list[Edit]:
    """A thinking renderer (:func:`_renderers`), taught to hand over and to
    fold its row. Two edits:

    * **fold-ref** -- the renderer binds the ref its gate hands it
      (:data:`_REF`) beside the block it takes apart (``param``), and its row's
      box (:data:`_ROW`) takes it, so the gate can measure the row. A box that
      already has a ref is upstream's to keep, and is reported as found and
      left.
    * **fold-line** -- the collapsed text, the italic bag whose children read
      the block's ``thinking`` (followed through what its names are bound
      to, :func:`_mentions`), keeps to one line: upstream's own collapsed thinking,
      the start of the block, or with ``timed`` how long it was thought for
      instead (:func:`_thought`) -- "∴ Thought for 12s". Either reads apart at
      a glance from the tool groups around it.
    """
    ref, line = outcome.step("fold-ref"), outcome.step("fold-line")
    edits: list[Edit] = []
    param = js.only(
        [
            (js.binding(js.props(pattern)[_PARAM]), pattern)
            for pattern in js.every(renderer, js.of_type("object_pattern"), scoped=True)
            if _PARAM in js.props(pattern)
            and pattern.parent is not None
            and pattern.parent.type == "variable_declarator"
        ],
        f"bindings of {_PARAM}",
    )
    if param is None:
        return edits
    block, taken = param
    declarator = taken.parent
    if declarator is None:
        return edits
    taken_off = _thinking_of(source, renderer, block)
    thought = [binding for binding, _ in taken_off]
    bags = js.every(renderer, js.of_type("object"), scoped=True)

    row = js.only([b for b in bags if _ROW in js.props(b)], "row boxes")
    if row is not None:
        ref.candidates += 1
        if (
            "ref" not in js.props(row)
            and _REF not in js.props(taken)
            and js.visible(declarator, row)
        ):
            # First in the pattern: a rest element, if upstream adds one, has
            # to stay last.
            edits.append(Edit.before(js.children(taken)[0], f"{_REF},"))
            edits.append(Edit.after(js.children(row)[-1], f",ref:{_REF}"))
            ref.applied += 1

    collapsed = js.only(
        [
            b
            for b in bags
            if _ITALIC in js.props(b)
            and _mentions(source, js.props(b).get("children"), thought)
        ],
        "collapsed thinking texts",
    )
    if collapsed is not None:
        line.candidates += 1
        shown = js.props(collapsed)
        if timed:
            # The block as the renderer reads it there, by a name that means
            # the block where the time is drawn.
            read = next(
                (
                    value
                    for _, value in taken_off
                    if value.type == "identifier"
                    and source.portable(value, shown["children"])
                ),
                None,
            )
            if read is None:
                return edits
            edits.append(Edit.replace(shown["children"], _thought(js.text(read))))
        wrap = shown.get("wrap")
        if wrap is None:
            edits.append(Edit.before(js.children(collapsed)[0], f"wrap:{_ONE_LINE},"))
        elif js.text(wrap) != _ONE_LINE:
            edits.append(Edit.replace(wrap, _ONE_LINE))
        line.applied += 1
    return edits


def _drawing(bag: js.Node) -> tuple[js.Node, js.Node] | None:
    """The factory and component a props bag is drawn with, ``f(C, bag)``.

    Every render here is a component and a bag (:func:`js.call_props`); this
    is the same call read the other way, from the bag out to what receives it.
    """
    call = None if bag.parent is None else bag.parent.parent
    if call is None or call.type != "call_expression":
        return None
    factory = call.child_by_field_name("function")
    handed = js.arguments(call)
    if factory is None or len(handed) < 2 or handed[1] != bag:
        return None
    return factory, handed[0]


def _thinking_inline(source: Source, options: Options, outcome: Outcome) -> Source:
    """Route thinking past the collapse machinery and render it expanded.

    One rewrite in the activity-group builder (see :func:`_reroute_grouping`),
    then two inside every arm labelled ``"thinking"`` that renders with an
    ``isTranscriptMode`` prop: neutralise the early null-return, and force the
    renderer into transcript presentation (full markdown instead of the
    one-line collapsed form).

    The flag the arm hands its renderer is what ties the two together: it is
    the value forced to ``!0`` here and the name the early return is keyed on
    there, so neither rewrite is looking for a gate the other cannot see.

    Folding once off screen (``inline_fold``, to the block's thought time or
    to its first line) draws it instead through a gate (:func:`_gated`) that
    forces transcript presentation only while the block has not left the
    screen -- the real transcript view still draws it whole -- and watches each
    frame for the row leaving it, reading frames off upstream's app context
    (:func:`_fold_frames`). It also notes each block's time and age where the
    grouping accounts it (:func:`_noted`), teaches the renderer to hand over its
    row and to fold to one line (:func:`_folding`), and lets a click expand a
    folded block (:func:`_fold_click`): five more steps, declared under the same
    condition they run.
    """
    folds = options.inline_fold != FOLD_NEVER
    timed = options.inline_fold == FOLD_TIME
    outcome.declare(
        required=(
            "group-routing",
            "group-time",
            "null-guard",
            "renderer-props",
            *(
                ("fold-note", "fold-frames", "fold-ref", "fold-line", "fold-click")
                if folds
                else ()
            ),
        )
    )
    guard = outcome.step("null-guard")
    props = outcome.step("renderer-props")
    source = _reroute_grouping(source, outcome)
    edits = _accounting(source, outcome, folds=folds)
    if folds:
        edits += _fold_frames(source, outcome)
        edits += _fold_click(source, outcome)
        for renderer in _renderers(source):
            edits += _folding(source, renderer, outcome, timed=timed)

    for node in source.find(_THINKING_LABEL):
        arm = js.up(node, "switch_case")
        if arm is None or arm.child_by_field_name("value") != node:
            continue

        # Presentation is a set of properties, never a call shape: whichever
        # renderer this build reaches, and whatever else it is handed, the
        # bundle's own prop names are what say how a thinking block is drawn.
        # Of the object *passed* -- the same names in a destructuring pattern
        # bind locals, and forcing a value over one writes rubble rather than a
        # prop, which the gate would report as this patch broken.
        drawn = [
            obj
            for obj in js.every(arm, js.of_type("object"))
            if _TRANSCRIPT in js.props(obj)
        ]
        # What this arm calls transcript mode, in its own locals: the value it
        # hands over under that name. Every identifier inside it, for the reason
        # `subagent-prompt` reads through the expression rather than requiring a
        # bare one.
        transcript = {
            js.text(name)
            for obj in drawn
            for name in js.every(js.props(obj)[_TRANSCRIPT], js.of_type("identifier"))
        }
        for answer in _hidden(arm, transcript):
            guard.candidates += 1
            guard.applied += 1
            edits.append(Edit.replace(answer, b";"))

        for obj in drawn:
            carried = js.props(obj)
            # Folding, the block's own bag is drawn through the gate, which
            # hands the renderer transcript presentation while the block is
            # open; the flag the arm passes stays the arm's, so the real
            # transcript view still draws it whole.
            gated = folds and _PARAM in carried
            drawing = _drawing(obj) if gated else None
            forced = {_HIDE_IN_TRANSCRIPT: b"!1"}
            if not gated:
                forced[_TRANSCRIPT] = b"!0"
            landed = False
            for name, wanted in forced.items():
                value = carried.get(name)
                if value is None or value.text == wanted:
                    continue
                edits.append(Edit.replace(value, wanted))
                landed = True
            if drawing is not None:
                factory, component = drawing
                edits.append(
                    Edit.replace(
                        component, _gated(js.text(component), js.text(factory))
                    )
                )
            props.candidates += 1
            props.applied += (drawing is not None) if gated else landed

    return source.apply(edits)


# ------------------------------------------------------- the experiment bucket

#: The bucket's own read. Claude Code caches a GrowthBook assignment and echoes
#: it to the API so the server applies the same bucket; a slot that withholds
#: thinking summaries is served thinking blocks carrying an empty string, which
#: every other patch here then renders faithfully.
#:
#: Every read yields nothing, and that is the whole rewrite -- there is no
#: getter to identify. Emptying *the* getter meant proving which function was
#: it, because replacing a brace-free body wherever the property appeared would
#: delete whatever an unrelated reader did; one `.atis` per bundle is today's
#: happenstance, not an invariant. A read that reports nothing is correct
#: however many there are, so the branch that had to tell them apart is gone.
#:
#: The property, not the optional chain in front of it: `?.` is the caller's
#: choice and `.atis` is the name. Spelled as text it also took the front of a
#: longer name -- ``Gx()?.atisVersion`` read as this bucket and became
#: ``(Gx(),void 0)``, discarding a value nothing here has any claim on.
_BUCKET_READ = "atis"

#: The header the bucket rides in -- the feature's own name, and durable in a
#: way any read site is not. Candidates are counted off *this* so a reshaped
#: read reads as "found it, rewrote nothing" instead of the zero that would be
#: indistinguishable from a build where the mechanism is gone entirely.
_BUCKET_HEADER = '"x-cc-atis"'


def _thinking_summaries(source: Source, _options: Options, outcome: Outcome) -> Source:
    """Stop reporting the account's experiment bucket to the API."""
    outcome.candidates += source.count(_BUCKET_HEADER)
    edits: list[Edit] = []

    for node in source.find(_BUCKET_READ):
        read = js.up(node, "member_expression")
        if read is None or read.child_by_field_name("property") != node:
            continue
        # A write is a place, not a value: `(cache(),void 0)=x` is rubble, and
        # the gate would report it as this patch broken -- losing the read that
        # is the whole patch along with it. `spinner-tips` skips writes for the
        # same reason and out of the same home; upstream writes neither today,
        # which is exactly when a hazard is cheap to answer for.
        if js.written(read):
            continue
        cached = read.child_by_field_name("object")
        if cached is None:
            continue
        # The cache read is kept and its value discarded, so only the server's
        # view of the assignment changes: local feature values still come from
        # the on-disk cache, which is the promise this patch makes.
        edits.append(Edit.replace(read, f"({js.text(cached)},void 0)"))
        outcome.applied += 1

    return source.apply(edits)


# ------------------------------------------------------------- max effort
#
# `max` is a first-class level in session -- the level list the binary ships is
# ["low","medium","high","xhigh","max"], and the CLI and env var both accept it
# -- but persistence runs through two whitelists that stop at xhigh. Both are
# widened; either alone changes nothing (docs/PLAYBOOK.md), which is also why
# both steps are required.

#: The effort ladder from its one home, split at the rung this patch is about:
#: ``max`` is the top level (`_MAX`), and the binary's persistence whitelist
#: ships admitting everything below it (`_LEVELS`), which is how the whitelist is
#: told from the other functions that compare effort levels. Derived, so an
#: upstream level lands here without a second edit; the split assumes persistence
#: lags the session list by exactly the top rung, which it does today, and a
#: build that broke the assumption would report broken (loud), never silently
#: wrong.
_LEVELS = EFFORT_LADDER[:-1]
_MAX = EFFORT_LADDER[-1]
#: The persisted-settings schema entry. Its zod chain ends in `.catch(void 0)`,
#: so a clean binary reading a settings file that carries "max" (baked by us,
#: then reverted by a Claude update) treats it as unset rather than failing the
#: whole settings parse -- the property that makes this patch safe to lose.
_EFFORT_PROP = "effortLevel"


def _admits(condition: js.Node) -> set[str]:
    """The levels this condition tests against, as their values.

    The same membership question `schema` asks of the settings array, asked of an
    expression: what a guard admits is the set of levels it names, and a set is
    not a sequence here either. Asked as text -- ``'"low"' in js.text(condition)``
    -- it was a claim about quoting and spacing, made in the one module that
    exists to stop making claims about either, and it sat one build away from
    reading a single-quoted rewrite as the whole whitelist having vanished.

    Of the condition itself (``scoped=True``): a level named inside a callback
    written into the test is that callback's, not a level this guard admits.
    """
    return {
        js.text(level)[1:-1]
        for level in js.every(condition, js.of_type("string"), scoped=True)
    }


def _whitelist(node: js.Node) -> js.Node | None:
    """The *condition* admitting a level, reached from any level it names.

    Identity is what the guard *does*: it compares one value against every
    level and returns that same value. The gate is one tiny function and the
    whole choke point -- the `/effort` save path writes settings only when it
    returns a value, the startup resolver reads settings back through it, and
    the model picker persists through it too -- so write and read widen
    together in one edit.

    The condition, not the `if`, because the condition is what the rewrite
    extends and what the caller has to hold on to; handing back the `if` only
    to reach through it again for the same child is a second chance to reach
    for a different one.
    """
    guard = js.up(node, "if_statement")
    if guard is None:
        return None
    condition = guard.child_by_field_name("condition")
    consequence = guard.child_by_field_name("consequence")
    if condition is None or consequence is None:
        return None
    if not set(_LEVELS) <= _admits(condition):
        return None
    # The value admitted is the value returned -- which is what tells this
    # whitelist from any other function that merely compares effort levels.
    subject = js.first(condition, js.of_type("identifier"))
    if subject is None or js.first(consequence, js.returns(js.text(subject))) is None:
        return None
    return condition


def _max_effort(source: Source, _options: Options, outcome: Outcome) -> Source:
    """Let ``/effort max`` save as the default for new sessions."""
    outcome.declare(required=("gate", "schema"))
    gate = outcome.step("gate")
    schema = outcome.step("schema")
    edits: list[Edit] = []
    seen: set[int] = set()

    for node in source.find(f'"{_LEVELS[-1]}"'):
        if node.type != "string":
            continue
        condition = _whitelist(node)
        if condition is None or condition.id in seen:
            continue
        seen.add(condition.id)
        gate.candidates += 1
        gate.applied += 1
        subject = js.first(condition, js.of_type("identifier"))
        # An upstream that has adopted the level itself is the goal already
        # achieved, not a miss -- the same judged-on-achievement rule
        # `subagent-models` follows. Asked of the same set the guard was
        # identified by, so "already admits max" and "admits every other level"
        # can never be answered two different ways.
        if _MAX in _admits(condition):
            continue
        edits.append(
            Edit.replace(
                condition, f'({js.text(condition)}||{js.text(subject)}==="{_MAX}")'
            )
        )

    for node in source.find(_EFFORT_PROP):
        if node.type != "property_identifier":
            continue
        pair = js.named(node)
        if pair is None:
            continue
        # The array is confirmed by what it *contains*, which is what tells it
        # from the twelve other `effortLevel` properties in the bundle -- and
        # from anything else composed inside the same value, since the
        # expression this build reaches its schema factory through is minifier
        # noise that is skipped rather than described, so "the first array in
        # there" is a guess about the noise. Never by the sequence it lists them
        # in, or by there being nothing else in it: a set is not a sequence here
        # either (`custom-models/validator` says so in as many words), so an
        # upstream reshuffle or one more level is a spelling of the same array,
        # and demanding the exact list made both read as the whole
        # persisted-settings schema having vanished.
        levels = next(
            (
                n
                for n in source.walk(pair.child_by_field_name("value"))
                if n.type == "array" and set(_LEVELS) <= set(js.strings(n))
            ),
            None,
        )
        if levels is None:
            continue
        schema.candidates += 1
        schema.applied += 1
        if levels.id in seen or _MAX in js.strings(levels):
            continue
        seen.add(levels.id)
        edits.append(Edit.after(js.elements(levels)[-1], f',"{_MAX}"'))

    return source.apply(edits)


PATCHES = [
    Patch(
        id="thinking-summaries",
        title="Fix blank thinking blocks",
        group=GROUP_OUTPUT,
        fn=_thinking_summaries,
        anchors=(_BUCKET_HEADER, _BUCKET_READ),
    ),
    Patch(
        id="thinking-inline",
        title="Always show thinking",
        group=GROUP_OUTPUT,
        fn=_thinking_inline,
        anchors=(_THINKING_LABEL, _TRANSCRIPT, _SUMMARY, _THINK_TIME),
        setting=string_setting(
            "inline_fold", "inline_fold", FOLD_NEVER, choices=tuple(FOLDS)
        ),
    ),
    Patch(
        id="max-effort",
        title="Persist max effort",
        group=GROUP_MODELS,
        fn=_max_effort,
        anchors=('"xhigh"', f"{_EFFORT_PROP}:", "Persisted effort level"),
    ),
]

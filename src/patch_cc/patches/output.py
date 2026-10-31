"""Tool-call and diff rendering patches, and the icon done tool groups draw."""

from __future__ import annotations

from .. import js
from ..js import Edit, Source
from .base import GROUP_OUTPUT, Options, Outcome, Patch

# --------------------------------------------------------------- tool calls

_COLLAPSED = '"collapsed_read_search"'
_VERBOSE = "verbose"


def _arm(node: js.Node) -> js.Node | None:
    """The ``switch`` arm this label *labels*, rather than one it occurs in.

    The label is an authored name and the arm is a node; ``case"x":`` spelled as
    text is neither -- it is the two of them with the minifier's spacing in
    between, which is the one thing about a build that has no reason to hold.
    """
    arm = js.up(node, "switch_case")
    return arm if arm is not None and arm.child_by_field_name("value") == node else None


def _tool_call_verbose(source: Source, _options: Options, outcome: Outcome) -> Source:
    """Render collapsed read/search rows as if verbose mode were on.

    The arm is the dispatch label; the row is whatever inside it is *handed* a
    ``verbose`` prop. Which render function receives it, whether the arm is an
    expression or a block, and what else it is handed are all the minifier's
    to spell -- a matcher per arm form, carrying neighbouring prop names to
    prove it has found the right call, dates itself to one build. The label
    proves that.

    Handed, not taken apart. A prop being passed is an object literal; the same
    name in a destructuring pattern is a local being bound, and writing ``!0``
    over the name it binds is not a value flip but rubble. The gate would catch
    it, which makes it this patch reported broken on a build that merely reads
    the prop somewhere else -- a shape upstream is free to ship, and one this
    rewrite has nothing to say about.
    """
    edits = []
    for node in source.find(_COLLAPSED):
        arm = _arm(node)
        if arm is None:
            continue
        # `scoped`: the row is what *this arm* hands a `verbose` prop, not what a
        # callback nested inside it hands one -- a decoy object in such a callback
        # would otherwise be flipped too.
        for obj in js.every(arm, js.of_type("object"), scoped=True):
            verbose = js.props(obj).get(_VERBOSE)
            if verbose is None:
                continue
            outcome.candidates += 1
            if not js.is_true(verbose):
                edits.append(Edit.replace(verbose, b"!0"))
                outcome.applied += 1
    return source.apply(edits)


# --------------------------------------------------------------- create diff

_CREATE = '"create"'
_UPDATE = '"update"'

#: What the created-file render is handed. Membership, never the whole set: the
#: arm draws several other things -- a "/plan to preview" placeholder, two
#: "Wrote N lines" summaries -- and these three pick it out of them alone, on
#: every build in the corpus. Demanding *exactly* these was one added prop from
#: reporting the anchor gone, which is precisely what upstream has already done
#: twice to the sibling render below.
_CREATE_PROPS = ("filePath", "content", "verbose")

#: What the updated-file render is handed. Membership for the same reason:
#: upstream keeps adding to this one (``previewHint`` and ``collapsed`` since
#: this patch was written), and none of that is a fact about which render it is.
_UPDATE_PROPS = ("filePath", "structuredPatch")

#: The word the arm's own summary pluralises the count with -- what tells the
#: line count from any other call the arm makes on the created file's text.
_LINE = '"line"'


def _sibling_arm(arm: js.Node, label: str) -> js.Node | None:
    """The arm for ``label`` in the same ``switch`` as ``arm``.

    Membership of one switch, never a direction: which of two arms upstream
    lists first is not a fact about either, and walking forward only read a
    build that lists `update` above `create` as the anchor being gone (0/0, on
    a switch plainly carrying both).
    """
    return next(
        (
            sibling
            for sibling in js.children(arm.parent)
            if sibling.type == "switch_case"
            and (value := sibling.child_by_field_name("value")) is not None
            and js.text(value) == label
        ),
        None,
    )


def _line_count(source: Source, arm: js.Node, created: js.Node) -> str | None:
    """How this build counts the lines of a created file.

    The arm already computes it for its own "Wrote N lines" summary, so the
    helper is borrowed rather than reimplemented, and what we render says the
    same number the arm beside it does.

    Which call that is, is settled by what the arm *says* about the number --
    it pluralises it as a ``"line"``, upstream's own word -- and not by being
    the first call in the arm taking the content. Any call does that: put a
    bare ``String(t);`` at the arm's head and it was baked in as the file's
    length, ``newLines:String(t)``, counted and green.

    There is deliberately no inline fallback. One stood here, never ran on any
    build in the corpus, and disagreed with the borrow it stood in for on the
    commonest input there is: upstream's helper drops the empty trailing field
    of a file that ends in a newline (and calls an empty file one line), while
    ``split`` counts both. A second answer nothing exercises is not resilience,
    it is a number that changes meaning on the build that finally reaches it.

    The arm counts the same file twice (once per summary it may draw), so what
    has to be single is the *answer*, not the call: two different counts would
    leave nothing to borrow.
    """
    content = js.text(js.call_props(created)["content"])
    counted = [
        value
        for summary in js.every(arm, js.of_type("call_expression"))
        if _LINE in [js.text(a) for a in js.arguments(summary)[1:]]
        for value in source.values(js.arguments(summary)[0])
        if value.type == "call_expression"
        and [js.text(a) for a in js.arguments(value)] == [content]
    ]
    borrowed = {js.text(value) for value in counted}
    if len(borrowed) != 1 or not all(source.portable(v, created) for v in counted):
        return None
    return borrowed.pop()


def _create_diff_colors(source: Source, _options: Options, outcome: Outcome) -> Source:
    """Render newly-created files through the diff component so lines get ``+``.

    A creation is a diff against nothing, so the row is re-pointed at the
    renderer the neighbouring ``update`` arm already uses, handed a patch whose
    every line is an addition. Both renders are found by what they are handed,
    so nothing here depends on which JSX runtime the build emits or on the
    order upstream lists the props in.
    """
    edits = []
    for node in source.find(_CREATE):
        create_arm = _arm(node)
        if create_arm is None:
            continue
        update_arm = _sibling_arm(create_arm, _UPDATE)
        if update_arm is None:
            continue

        # `js.only`, not `js.first`: the created render is rewritten and the
        # update render is read for the renderer/component/style it lends, and a
        # rewrite that takes the first of two matches is how a decoy wins without
        # a counter moving (docs/PLAYBOOK.md). Cardinality is 1 on every build in
        # the corpus, so this costs nothing today and turns a second match into
        # one broken patch naming its own confusion.
        created = js.only(
            js.every(
                create_arm,
                lambda n: (
                    n.type == "call_expression"
                    and all(name in js.call_props(n) for name in _CREATE_PROPS)
                ),
            ),
            "created-file renders in this arm",
        )
        updated = js.only(
            js.every(
                update_arm,
                lambda n: (
                    n.type == "call_expression"
                    and all(name in js.call_props(n) for name in _UPDATE_PROPS)
                ),
            ),
            "updated-file renders in this arm",
        )
        if created is None or updated is None:
            continue

        outcome.candidates += 1
        made = js.call_props(created)
        path, content, verbose = (
            js.text(made["filePath"]),
            js.text(made["content"]),
            js.text(made[_VERBOSE]),
        )
        lent = (
            updated.child_by_field_name("function"),
            js.arguments(updated)[0],
            js.call_props(updated)["style"],
        )
        lines = _line_count(source, create_arm, created)
        if lines is None or not all(source.portable(node, created) for node in lent):
            continue
        renderer, component, style = map(js.text, lent)

        edits.append(
            Edit.replace(
                created,
                f"{renderer}({component},{{"
                f"filePath:{path},structuredPatch:[{{"
                f"oldStart:1,oldLines:0,newStart:1,newLines:{lines},"
                f'lines:{content}===""?[]:{content}.split(`\\n`)'
                f'.map((__cc_line)=>"+"+__cc_line)}}],'
                f"firstLine:{content}.split(`\\n`)[0]??null,"
                f'fileContent:"",style:{style},verbose:{verbose},'
                f"previewHint:void 0}})",
            )
        )
        outcome.applied += 1

    return source.apply(edits)


# --------------------------------------------------------------- tool icon

#: The status dot a running tool group draws in its header's leading slot --
#: upstream's own prop name, and the branch that says which slot this is.
_UNRESOLVED = "isUnresolved"

#: The slot: a box given two columns, where every other row draws its glyph
#: (``∴`` thinking, ``✻`` the turn's duration, ``※`` a recap). A done tool
#: group's is left empty.
_SLOT = "minWidth"

#: The header's own text element, keyed by upstream: what the glyph is drawn
#: with, so no text component is spelled.
_ELLIPSIS_KEY = '"ellipsis"'

#: ⚙, written as the escape Bun writes non-ASCII as.
_GEAR = '"\\u2699"'


def _branch_bag(branch: js.Node | None) -> js.Node | None:
    """The props bag a branch's call hands over, if the branch is a call."""
    if branch is None or branch.type != "call_expression":
        return None
    return next((a for a in js.arguments(branch) if a.type == "object"), None)


def _tool_icon(source: Source, _options: Options, outcome: Outcome) -> Source:
    """Draw ⚙ in a done tool group's empty leading slot.

    The slot is found by the ternary that fills it: one branch the running
    status dot (the call handed ``isUnresolved``), the other a box handed only
    its two columns. A slot that already holds something is upstream's to keep
    -- its own tool row draws a glyph there -- so only the empty one is filled,
    and in upstream's own shape for a glyph in that slot: a dim text in the box,
    built with the slot's own element call and the header's own text component
    (the element keyed ``"ellipsis"``), so neither is named here, and only
    where both mean the same in the slot (:meth:`js.Source.portable`). Every
    such slot is filled: each is proven by both branches.
    """
    edits: list[Edit] = []
    for node in source.find(_UNRESOLVED):
        pair = js.named(node)
        if pair is None or pair.type != "pair" or pair.parent is None:
            continue
        handed = pair.parent.parent
        dot = (
            handed.parent if handed is not None and handed.type == "arguments" else None
        )
        choice = dot.parent if dot is not None else None
        if choice is None or choice.type != "ternary_expression":
            continue
        branches = [
            choice.child_by_field_name("consequence"),
            choice.child_by_field_name("alternative"),
        ]
        slot = next((b for b in branches if b != dot), None)
        bag = _branch_bag(slot)
        if bag is None or slot is None or set(js.props(bag)) != {_SLOT}:
            continue
        outcome.candidates += 1
        component = js.climb(choice, js.of_type(*js.FUNCTIONS))
        text = js.only(
            [
                args[0]
                for call in js.every(component, js.of_type("call_expression"))
                if (args := js.arguments(call))
                and js.text(args[-1]) == _ELLIPSIS_KEY
                and args[0].type == "identifier"
            ],
            "elements keyed as the header's ellipsis",
        )
        element = slot.child_by_field_name("function")
        if text is None or not all(source.portable(n, bag) for n in (element, text)):
            continue
        glyph = (
            f"{js.text(element)}({js.text(text)},"
            f'{{"aria-hidden":!0,dimColor:!0,children:{_GEAR}}})'
        )
        edits.append(Edit.before(js.children(bag)[0], f"children:{glyph},"))
        outcome.applied += 1
    return source.apply(edits)


PATCHES = [
    Patch(
        id="tool-calls",
        title="Detailed tool calls",
        group=GROUP_OUTPUT,
        fn=_tool_call_verbose,
        anchors=(_COLLAPSED,),
    ),
    Patch(
        id="create-diff",
        title="Colour new files as diffs",
        group=GROUP_OUTPUT,
        fn=_create_diff_colors,
        anchors=(_CREATE, _UPDATE, "structuredPatch:"),
    ),
    Patch(
        id="tool-icon",
        title="Tool call icon",
        group=GROUP_OUTPUT,
        fn=_tool_icon,
        default=False,
        anchors=(_UNRESOLVED, _ELLIPSIS_KEY),
    ),
]

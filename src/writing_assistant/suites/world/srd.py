"""A deterministic SRD 5.1 subset over `ep.state["world"]`, and its tool table.

The world is plain JSON; every function here takes it and changes it in place, so a fork
(a deep copy of `state`) carries the dice too: `roll` draws from `seed` and `roll_count`,
which makes a twin forked mid-play roll what its parent would have rolled next.

The subset is what makes action legality and visibility checkable, not the rulebook: d20
attacks and ability checks with advantage/disadvantage from conditions, HP and the drop to 0,
the SRD conditions that stop actions or movement, inventory, items on the ground, location
exits, secrets with a reveal condition, and recorded facts. A refused move returns
`{"ok": False, "error": ..., "rule": ...}` (the workspace tools' failure shape, so a model
sees one shape) and counts in `world["rules_rejections"]`.

World skeleton (`state["world"]`, written by `setup()` from the task row):

    seed: int, roll_count: int, rules_rejections: int
    party: [character id]                       # the player characters
    locations: {id: {name, exits: [id], items: [item id]}}
    characters: {id: {name, location, hp, max_hp, ac, attack_bonus,
                      abilities: {str, dex, con, int, wis, cha} (modifiers),
                      conditions: [name], inventory: [item id], motive?}}
    items: {id: {name, kind: "weapon" | "potion" | "gear", damage?, damage_bonus?, heal?}}
    secrets: {id: {text, owner, known_by: [character id], reveal: {location?: id}}}
    facts: [{text, visible_to: [perceiver]}]

Perceivers in `visible_to` are character ids and `"dm"`.
"""

from __future__ import annotations

import random
import re
from typing import Any

from writing_assistant.episode import Episode

DM = "dm"
CONDITIONS = (
    "blinded", "charmed", "deafened", "frightened", "grappled", "incapacitated", "invisible",
    "paralyzed", "petrified", "poisoned", "prone", "restrained", "stunned", "unconscious",
)
NO_ACTIONS = {"incapacitated", "paralyzed", "petrified", "stunned", "unconscious"}
NO_MOVEMENT = NO_ACTIONS | {"grappled", "restrained"}
ATTACK_DISADVANTAGE = {"blinded", "frightened", "poisoned", "prone", "restrained"}
ATTACKED_WITH_ADVANTAGE = {"blinded", "paralyzed", "restrained", "stunned", "unconscious"}
CHECK_DISADVANTAGE = {"frightened", "poisoned"}
ABILITIES = ("str", "dex", "con", "int", "wis", "cha")
ACTIONS = ("move", "attack", "take", "give", "use", "check", "condition")

_DICE = re.compile(r"^\s*(\d{0,2})d(\d{1,3})\s*(?:([+-])\s*(\d{1,3}))?\s*$")


def _reject(world: dict, rule: str, error: str) -> dict:
    world["rules_rejections"] += 1
    return {"ok": False, "rule": rule, "error": error}


def roll(world: dict, expr: str) -> dict:
    """Roll `NdM±K` (N 1-20, M 2-100) with the world's RNG: one draw stream per call."""
    m = _DICE.match(expr)
    n, sides = (int(m[1] or 1), int(m[2])) if m else (0, 0)
    if not m or not 1 <= n <= 20 or not 2 <= sides <= 100:
        return _reject(world, "dice", f"bad dice expression {expr!r}: use NdM+K with N 1-20 and M 2-100, e.g. '2d6+1'")
    rng = random.Random(f"{world['seed']}:{world['roll_count']}")
    world["roll_count"] += 1
    rolls = [rng.randint(1, sides) for _ in range(n)]
    bonus = int(m[4] or 0) * (-1 if m[3] == "-" else 1)
    return {"ok": True, "expr": expr, "rolls": rolls, "total": sum(rolls) + bonus}


def _d20(world: dict, advantage: bool, disadvantage: bool) -> tuple[int, list[int]]:
    """SRD: advantage and disadvantage cancel; otherwise roll two and keep one."""
    if advantage == disadvantage:
        r = roll(world, "1d20")["rolls"]
        return r[0], r
    r = roll(world, "2d20")["rolls"]
    return (max(r) if advantage else min(r)), r


def present(world: dict, location: str) -> list[str]:
    return sorted(c for c, ch in world["characters"].items() if ch["location"] == location)


def sees(world: dict, *locations: str) -> dict[str, str]:
    """Everyone at the locations, plus the DM, perceives the event in full."""
    seen = {DM: "full"}
    for loc in locations:
        seen |= {c: "full" for c in present(world, loc)}
    return seen


def _perceivers(world: dict, ids: list[str]) -> str | None:
    unknown = [i for i in ids if i != DM and i not in world["characters"]]
    return f"unknown perceivers {unknown}; use character ids {sorted(world['characters'])} or 'dm'" if unknown else None


def resolve(world: dict, action: dict) -> dict:
    """Validate and apply one action. On success: `{ok: True, text, visible_to, ...}` where
    `text` is the event's summary and `visible_to` who perceives it."""
    kind = action["action"]
    chars, items = world["characters"], world["items"]
    actor_id = action.get("actor")
    actor = chars.get(actor_id)
    if actor is None:
        return _reject(world, "unknown_id", f"unknown actor {actor_id!r}; characters: {sorted(chars)}")
    here = actor["location"]
    conds = set(actor["conditions"])

    if kind == "condition":
        name, remove = action.get("condition"), action.get("remove", False)
        if name not in CONDITIONS:
            return _reject(world, "unknown_id", f"unknown condition {name!r}; SRD conditions: {list(CONDITIONS)}")
        if remove:
            if name not in conds:
                return _reject(world, "legality", f"{actor_id} is not {name}")
            actor["conditions"].remove(name)
        elif name not in conds:
            actor["conditions"].append(name)
        verb = "is no longer" if remove else "is now"
        return {"ok": True, "text": f"{actor['name']} {verb} {name}.", "visible_to": sees(world, here)}

    blocked = conds & (NO_MOVEMENT if kind == "move" else NO_ACTIONS)
    if blocked:
        return _reject(world, "legality", f"{actor_id} cannot {kind} while {sorted(blocked)[0]}")

    if kind == "move":
        to = action.get("to")
        if to not in world["locations"]:
            return _reject(world, "unknown_id", f"unknown location {to!r}; locations: {sorted(world['locations'])}")
        if to not in world["locations"][here]["exits"]:
            return _reject(world, "legality", f"no way from {here} to {to}; exits: {world['locations'][here]['exits']}")
        seen = sees(world, here, to)
        actor["location"] = to
        seen |= {actor_id: "full"}
        return {"ok": True, "text": f"{actor['name']} goes from {here} to {to}.", "visible_to": seen}

    if kind == "take":
        item = action.get("item")
        if item not in world["locations"][here]["items"]:
            return _reject(world, "legality", f"no {item!r} at {here}; items here: {world['locations'][here]['items']}")
        world["locations"][here]["items"].remove(item)
        actor["inventory"].append(item)
        return {"ok": True, "text": f"{actor['name']} takes the {items[item]['name']}.", "visible_to": sees(world, here)}

    if kind == "check":
        ability, dc = action.get("ability"), action.get("dc")
        if ability not in ABILITIES or not isinstance(dc, int) or not 1 <= dc <= 30:
            return _reject(world, "legality", f"a check needs ability in {list(ABILITIES)} and dc 1-30")
        d20, rolls = _d20(world, False, bool(conds & CHECK_DISADVANTAGE))
        total = d20 + actor["abilities"][ability]
        result = "succeeds" if total >= dc else "fails"
        return {"ok": True, "text": f"{actor['name']} {result} a {ability} check ({total} vs DC {dc}).",
                "visible_to": sees(world, here), "rolls": rolls, "total": total, "success": total >= dc}

    item_id = action.get("item")
    if item_id not in actor["inventory"]:
        return _reject(world, "legality", f"{actor_id} does not carry {item_id!r}; inventory: {actor['inventory']}")
    item = items[item_id]
    target_id = action.get("target", actor_id if kind == "use" else None)
    target = chars.get(target_id)
    if target is None:
        return _reject(world, "unknown_id", f"unknown target {target_id!r}; characters: {sorted(chars)}")
    if target["location"] != here:
        return _reject(world, "legality", f"{target_id} is not at {here}")

    if kind == "give":
        actor["inventory"].remove(item_id)
        target["inventory"].append(item_id)
        return {"ok": True, "text": f"{actor['name']} gives the {item['name']} to {target['name']}.",
                "visible_to": sees(world, here)}

    if kind == "use":
        if item["kind"] != "potion":
            return _reject(world, "legality", f"{item_id} is not usable; only potions are")
        actor["inventory"].remove(item_id)
        healed = roll(world, item["heal"])["total"]
        target["hp"] = min(target["max_hp"], target["hp"] + healed)
        if target["hp"] > 0 and "unconscious" in target["conditions"]:
            target["conditions"].remove("unconscious")
        return {"ok": True, "text": f"{target['name']} drinks the {item['name']} and heals {healed} (HP {target['hp']}/{target['max_hp']}).",
                "visible_to": sees(world, here), "healed": healed}

    if kind == "attack":
        if item["kind"] != "weapon":
            return _reject(world, "legality", f"{item_id} is not a weapon")
        if target_id == actor_id:
            return _reject(world, "legality", "a character cannot attack itself")
        adv = bool(set(target["conditions"]) & ATTACKED_WITH_ADVANTAGE)
        d20, rolls = _d20(world, adv, bool(conds & ATTACK_DISADVANTAGE))
        hit = d20 == 20 or (d20 != 1 and d20 + actor["attack_bonus"] >= target["ac"])
        if not hit:
            return {"ok": True, "text": f"{actor['name']} attacks {target['name']} with the {item['name']} and misses.",
                    "visible_to": sees(world, here), "rolls": rolls, "hit": False}
        n, _, sides = item["damage"].partition("d")
        dice = f"{int(n or 1) * (2 if d20 == 20 else 1)}d{sides}"
        damage = max(0, roll(world, dice)["total"] + item.get("damage_bonus", 0))
        target["hp"] = max(0, target["hp"] - damage)
        text = f"{actor['name']} hits {target['name']} with the {item['name']} for {damage} (HP {target['hp']}/{target['max_hp']})."
        if target["hp"] == 0:
            target["conditions"] += [c for c in ("unconscious", "prone") if c not in target["conditions"]]
            text += f" {target['name']} falls unconscious."
        return {"ok": True, "text": text, "visible_to": sees(world, here), "rolls": rolls, "hit": True, "damage": damage}

    return _reject(world, "unknown_id", f"unknown action {kind!r}; actions: {list(ACTIONS)}")


def propose_event(world: dict, text: str, location: str, visible_to: list[str]) -> dict:
    """A narrative event at a location. Only the DM and characters there can perceive it."""
    if location not in world["locations"]:
        return _reject(world, "unknown_id", f"unknown location {location!r}; locations: {sorted(world['locations'])}")
    problem = _perceivers(world, visible_to)
    if problem:
        return _reject(world, "unknown_id", problem)
    absent = [c for c in visible_to if c != DM and world["characters"][c]["location"] != location]
    if absent:
        return _reject(world, "visibility", f"{absent} are not at {location} and cannot perceive this; present: {present(world, location)}")
    return {"ok": True, "text": text, "visible_to": {DM: "full"} | {c: "full" for c in visible_to}}


def record_fact(world: dict, text: str, visible_to: list[str]) -> dict:
    """Commit a world fact, known to the listed perceivers (the DM always knows it)."""
    problem = _perceivers(world, visible_to)
    if problem:
        return _reject(world, "unknown_id", problem)
    seen = {DM: "full"} | {c: "full" for c in visible_to}
    world["facts"].append({"text": text, "visible_to": sorted(seen)})
    return {"ok": True, "text": text, "visible_to": seen}


def reveal(world: dict, secret_id: str, to: list[str]) -> dict:
    """Reveal a secret to characters, once its reveal condition holds for each of them."""
    secret = world["secrets"].get(secret_id)
    if secret is None:
        return _reject(world, "unknown_id", f"unknown secret {secret_id!r}; secrets: {sorted(world['secrets'])}")
    problem = _perceivers(world, to)
    if problem:
        return _reject(world, "unknown_id", problem)
    where = secret["reveal"].get("location")
    early = [c for c in to if c != DM and where and world["characters"][c]["location"] != where]
    if early:
        return _reject(world, "visibility", f"secret {secret_id} can only be revealed at {where}; {early} are elsewhere")
    secret["known_by"] = sorted(set(secret["known_by"]) | {c for c in to if c != DM})
    return {"ok": True, "text": secret["text"], "visible_to": {DM: "full"} | {c: "full" for c in to}}


# -- the tool table ------------------------------------------------------------------------

PERCEIVERS = {
    "type": "array",
    "items": {"type": "string", "minLength": 1},
    "description": "Who perceives this: character ids from the world brief, or 'dm'. Unknown ids are refused.",
}


def _apply(ep: Episode, actor: str, out: dict, **data: Any) -> dict:
    """Log a successful engine result as a `world` event carrying its `visible_to`."""
    if out["ok"]:
        visible_to = out.pop("visible_to")
        ep.log(actor, "world", {**data, **out}, visible_to)
        out["visible_to"] = sorted(visible_to)
    return out


def _roll_tool(ep: Episode, actor: str, args: dict) -> dict:
    out = roll(ep.state["world"], args["expr"])
    hidden = args.get("hidden", False)
    if out["ok"]:
        out["text"] = f"Roll {out['expr']}: {out['rolls']} = {out['total']}" + (f" ({args['reason']})" if args.get("reason") else "")
        out["visible_to"] = {DM: "full"} if hidden else {"*": "full"}
    return _apply(ep, actor, out, tool="roll", hidden=hidden)


def _resolve_tool(ep: Episode, actor: str, args: dict) -> dict:
    return _apply(ep, actor, resolve(ep.state["world"], args), tool="resolve", action=args)


def _event_tool(ep: Episode, actor: str, args: dict) -> dict:
    out = propose_event(ep.state["world"], args["text"], args["location"], args["visible_to"])
    return _apply(ep, actor, out, tool="propose_event", location=args["location"])


def _fact_tool(ep: Episode, actor: str, args: dict) -> dict:
    return _apply(ep, actor, record_fact(ep.state["world"], args["text"], args["visible_to"]), tool="record_fact")


def _reveal_tool(ep: Episode, actor: str, args: dict) -> dict:
    out = reveal(ep.state["world"], args["secret_id"], args["to"])
    return _apply(ep, actor, out, tool="reveal", secret_id=args["secret_id"])


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}


TOOLS = {
    "srd_roll": (
        _schema(
            "srd_roll",
            "Roll dice with the world's seeded dice. 'expr' is NdM or NdM+K / NdM-K with N 1-20 and M 2-100 "
            "(e.g. '1d20+3'); anything else is refused. A hidden roll is seen by the DM only; otherwise everyone sees it.",
            {
                "expr": {"type": "string", "minLength": 2, "description": "Dice expression, e.g. '2d6+1'."},
                "hidden": {"type": "boolean", "description": "True for a DM-only roll. Default false."},
                "reason": {"type": "string", "description": "What the roll is for."},
            },
            ["expr"],
        ),
        _roll_tool,
    ),
    "srd_resolve": (
        _schema(
            "srd_resolve",
            "Resolve one character action under the rules; the engine rolls the dice and updates HP, conditions, "
            "inventory and location, and everyone at the location perceives the result. Per action: "
            "'move' needs 'to', a location listed in the actor's current location's exits; "
            "'take' needs 'item', lying at the actor's location; "
            "'give' needs 'item' the actor carries and 'target' at the same location; "
            "'use' needs 'item', a potion the actor carries, and optional 'target' at the same location (default the actor); "
            "'attack' needs 'item', a weapon the actor carries, and 'target', another character at the same location "
            "(d20 + attack bonus vs AC; 20 always hits and doubles damage dice, 1 always misses; at 0 HP the target falls unconscious); "
            "'check' needs 'ability' and 'dc' 1-30 (d20 + ability modifier); "
            "'condition' needs 'condition', an SRD condition, and 'remove' true to end one the actor has. "
            "An actor that is incapacitated, paralyzed, petrified, stunned or unconscious cannot act except by 'condition'; "
            "grappled or restrained also cannot move. A refused action counts as a rules rejection.",
            {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "actor": {"type": "string", "minLength": 1, "description": "Character id taking the action."},
                "target": {"type": "string", "minLength": 1, "description": "Character id acted on."},
                "item": {"type": "string", "minLength": 1, "description": "Item id."},
                "to": {"type": "string", "minLength": 1, "description": "Destination location id (move)."},
                "ability": {"type": "string", "enum": list(ABILITIES)},
                "dc": {"type": "integer", "minimum": 1, "maximum": 30},
                "condition": {"type": "string", "enum": list(CONDITIONS)},
                "remove": {"type": "boolean", "description": "End the condition instead of applying it."},
            },
            ["action", "actor"],
        ),
        _resolve_tool,
    ),
    "srd_propose_event": (
        _schema(
            "srd_propose_event",
            "Put a narrative event into the world at one location. Every id in 'visible_to' must be 'dm' or a "
            "character currently at that location; anyone elsewhere is refused. The DM always perceives it.",
            {
                "text": {"type": "string", "minLength": 1, "description": "What happens, as the perceivers experience it."},
                "location": {"type": "string", "minLength": 1, "description": "Location id where it happens."},
                "visible_to": PERCEIVERS,
            },
            ["text", "location", "visible_to"],
        ),
        _event_tool,
    ),
    "srd_record_fact": (
        _schema(
            "srd_record_fact",
            "Commit a fact about the world, known to the listed perceivers and the DM. Ids must be 'dm' or character ids.",
            {"text": {"type": "string", "minLength": 1, "description": "The fact."}, "visible_to": PERCEIVERS},
            ["text", "visible_to"],
        ),
        _fact_tool,
    ),
    "srd_reveal": (
        _schema(
            "srd_reveal",
            "Reveal a secret from the world brief to characters. A secret with a reveal location can only be "
            "revealed to characters who are at that location; otherwise it is refused.",
            {
                "secret_id": {"type": "string", "minLength": 1, "description": "Secret id from the world brief."},
                "to": PERCEIVERS,
            },
            ["secret_id", "to"],
        ),
        _reveal_tool,
    ),
}

"""Assemble authored JSON slots and compile a self-contained training wave."""
import json
import re
from pathlib import Path

from writing_agent.catalog import fingerprint, save_json, validate_catalog
from writing_agent.suite import compile_scenarios

WORK = Path(__file__).resolve().parent
OUT = Path('data/processed/gen-wave1-train')
seeds = json.loads((WORK / 'seeds.json').read_text())
turns = json.loads((WORK / 'turns.json').read_text())
slots = json.loads((WORK / 'slots.json').read_text())
vocab = json.loads(Path('data/training/variation-catalog-v1.json').read_text())
READ = ['list_dir', 'read_file', 'search']
WRITE = READ + ['write_file', 'patch_file']
records, catalog, audit = [], [], []


def check(key, kind, artifact, metric='Q1', **params):
    return dict(id=key, metric=metric, method='deterministic', kind=kind,
                artifact=artifact, required=True, **params)


for n, seed in enumerate(seeds, 1):
    family = 'F1' if n <= 14 else 'F2' if n <= 24 else 'F3' if n <= 30 else 'F4' if n <= 38 else 'F5'
    ident = f'wave1-train-{n:03d}'
    slug = re.sub('[^a-z0-9]+', '-', seed['title'].lower()).strip('-')
    sid = 'original-wave1-train-' + slug
    raw_text = seed['passage']
    if 'protected' in seed:
        raw_text += '\n\n' + seed['protected']
    if 'fact' in seed:
        raw_text += '\n\nSigned current instruction: ' + seed['fact']
        raw_text += '\nWithdrawn earlier instruction: ' + seed['old_fact']
    source_path = OUT / 'sources' / (sid + '.txt')
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(raw_text)
    catalog.append(dict(id=sid, role='train', provenance='synthetic', work_id=sid,
                        author='Codex session author', author_id='codex-wave1-train-author',
                        sha256=fingerprint(raw_text), raw_sha256=fingerprint(source_path.read_bytes()),
                        raw_file=str(source_path), character_span=[0, len(raw_text)],
                        span_basis='UTF-8 decoded, universal newlines', url=None,
                        terms={'license': 'Repository-authored synthetic research material',
                               'evidence': str(source_path)}, upstream_split='newly_authored',
                        parents=[], transformations=[], text=raw_text))
    entry = turns.get(str(n), [])
    deep = bool(entry and isinstance(entry[0], str))
    followups = entry if deep else [item[1] for item in entry]
    kinds = slots['deep_kinds'] if deep else [item[0] for item in entry]
    pressure = 'REQUIRES_ASK' if 'ambiguity' in seed else 'MUST_NOT_ASK' if n in slots['must_not_ask'] else 'PERMITS_DEFAULT'
    specificity = 'explicit' if n % 2 else 'loose'
    channel = 'reply' if family in {'F1', 'F5'} else 'file'
    tools = [] if family == 'F1' and n <= 8 else READ if family == 'F5' else WRITE
    lo, hi = slots['word_ranges'][(n - 1) % 5]
    initial, prose, checks = {}, [], []
    brief = [f"{seed['title']} — a {seed['genre']} project.", seed['goal']]
    if family == 'F1' and not tools:
        brief.append('Passage to continue:\n' + seed['passage'])
    elif family == 'F1':
        initial['source/passage.md'] = seed['passage']
        brief.append('Read the supplied passage in source/passage.md and continue from its last moment. File tools are available for reference, but the scene belongs in your reply. Do not modify the workspace.')
    elif family == 'F2':
        initial['notes/source.md'] = seed['passage']
        initial['drafts/scene.md'] = seed['passage'].split('. ')[0] + '.\n\n' + seed['protected'] + '\n'
        brief.append('Read notes/source.md and the unfinished draft in drafts/scene.md. Expand and revise that rough start using the source facts.')
    elif family in {'F3', 'F4'}:
        initial['notes/source.md'] = seed['passage']
        brief.append('Use the original story material in notes/source.md.')
    else:
        code = f'STORY-{n:03d}'
        active_path = f'kb/record-{(n * 7) % 29:02d}.md'
        initial[active_path] = slots['retrieval_active'].format(code=code, **seed)
        initial['kb/withdrawn.md'] = slots['retrieval_old'].format(code=code, **seed)
        for j, topic in enumerate(slots['distractor_topics'], 1):
            initial[f'kb/facility-{j:02d}.md'] = slots['distractor'].format(topic=topic, number=j)
        links = '\n'.join(f'- [Record {i}]({Path(path).name})' for i, path in enumerate(sorted(initial), 1))
        initial['kb/index.md'] = slots['retrieval_index'].format(links=links)
        brief.append('Scene setup:\n' + seed['passage'])
        brief.append(slots['f5_intro'].format(code=code))
    brief.extend([f"Aim for a {seed['style']} voice.", slots[specificity], slots['continuity']])
    if family in {'F1', 'F5'}:
        brief.append(slots['reply_delivery'].format(lo=lo, hi=hi))
        prose = [dict(id='scene', kind='reply', selection='whole')]
    elif family == 'F2':
        brief.append(slots['file_delivery'].format(lo=lo, hi=hi))
        prose = [dict(id='scene', kind='file', path='drafts/scene.md', selection='whole')]
    elif family == 'F3':
        brief.append(slots['alternatives_delivery'].format(lo=lo, hi=hi))
        prose = [dict(id=f'direction-{c}', kind='file', path=f'plans/direction-{c}.md', selection='whole') for c in 'abc']
    else:
        brief.append(slots['kb_delivery'])
        prose = [dict(id=p, kind='file', path=f'kb/{p}.md', selection='whole') for p in ['index', 'state', 'questions']]
    if pressure == 'REQUIRES_ASK':
        brief.append(seed['ambiguity'])
    elif pressure == 'MUST_NOT_ASK':
        brief.append(slots['must_not'])
    else:
        brief.append(slots['permits_' + channel])
    if followups:
        brief.append(slots['deep'])
    for selector in prose:
        aid = selector['id']
        bounds = (120, 220) if aid == 'state' else (80, 150) if family == 'F4' else (lo, hi)
        checks.append(check('delivery-' + aid, 'nonempty', aid, metric='Q3'))
        checks.append(check('words-' + aid, 'word_range', aid, min=bounds[0], max=bounds[1]))
        checks.append(check('continuity-' + aid, 'excludes', aid, metric='Q13', text=seed['forbidden']))
    if tools:
        allowed = [p['path'] for p in prose if p['kind'] == 'file']
        checks.append(check('workspace-scope', 'allowed_changes', prose[0]['id'], paths=allowed))
    if family == 'F2':
        checks.append(check('protected-line', 'protected', 'scene', metric='Q13', path='drafts/scene.md', text=seed['protected']))
    if family == 'F4':
        checks.extend([check('wiki-words', 'kb_word_budget', 'index', max=520),
                       check('wiki-links', 'wiki_links', 'index', metric='Q10')])
    if family == 'F5':
        checks.extend([check('current-instruction', 'contains', 'scene', metric='Q12', text=seed['fact']),
                       check('instruction-exposed', 'evidence_exposed', 'scene', metric='Q12', text=seed['fact']),
                       check('superseded-instruction', 'excludes', 'scene', metric='Q13', text=seed['old_fact'])])
    descriptions = {
        'Q1': 'Follow the visible brief and subsequent author steers. Judge every turn when assessing whether a consequential branch was surfaced, a reversible default was stated, or an unnecessary question was avoided. Do not invent additional requirements.',
        'Q13': f"Preserve the source facts and later accepted changes. In particular, do not assert as established fact: {seed['forbidden']} A literal prohibition check alone does not establish semantic continuity.",
    }
    descriptions['Q2' if family in {'F1', 'F2', 'F5'} else 'Q5' if family == 'F3' else 'Q7'] = seed['goal'] + ' Evaluate specificity, coherence, and usefulness; do not demand a hidden preferred plot.'
    if family == 'F3':
        descriptions['Q6'] = 'The three developed directions differ in character choices and causal consequences, rather than cosmetic names or tone.'
    if family == 'F4':
        descriptions['Q8'] = 'The bible retains facts useful for the requested scene, separates claims and uncertainty, and avoids invented answers or filler to meet its budget.'
    if family == 'F5':
        descriptions['Q12'] = 'The assistant searches for the requested key, reads the active record in context, and integrates its current instruction in the scene without relying on a withdrawn draft.'
    labels = dict(checks=checks, entrypoints=['kb/index.md'] if family in {'F4', 'F5'} else [],
                  knowledge={}, rubrics={k: dict(description=v, range=[1, 5], anchors=slots['rubric_anchors']) for k, v in descriptions.items()})
    visible = dict(brief='\n\n'.join(brief), initial_files=initial, followups=followups, tools=tools,
                   budgets=dict(max_steps=48 if deep else 24, max_tool_calls=64 if deep else 24,
                                max_read_tokens=1200 if family == 'F5' else 12000, max_total_bytes=262144), prose=prose)
    record = dict(id=ident, schema_version=1, family=family, role='train', condition=f'{family.lower()}_{channel}_{"deep" if deep else "short" if followups else "single"}',
                  genre=seed['genre'], style=seed['style'], instruction_specificity=specificity, review_status='generated',
                  provenance='synthetic', source_groups=[], source_ids=[sid],
                  response_pressure=pressure, steer_kinds=list(kinds),
                  realized_variation=dict(trope=vocab['tropes'][seed['trope']], situation=vocab['situations'][seed['situation']]),
                  visible=visible, labels=labels)
    records.append(record)
    audit.append(dict(id=ident, pressure=pressure, ambiguity=seed.get('ambiguity'),
                      required_check_reachability='Delivery paths and all word limits are explicit; positive content comes from named readable source material; prohibitions preserve stated uncertainty; final followups retain all delivery requirements.',
                      selectors=[p.get('path', 'final reply') for p in prose],
                      literal_continuity_limit=seed['forbidden']))

groups = validate_catalog(catalog)
for record in records:
    record['source_groups'] = sorted({groups[sid] for sid in record['source_ids']})
save_json(OUT / 'scenarios.json', records)
save_json(WORK / 'authoring-audit.json', audit)
manifest = compile_scenarios(records, catalog, OUT)
print(f"Compiled {len(manifest['scenarios'])} scenarios and {len(catalog)} original sources into {OUT}")

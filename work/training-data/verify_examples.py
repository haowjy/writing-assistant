"""Offline artifact verification using repository tools and mechanical_score.

This is a deterministic replay of authored deliverables, not a model evaluation.
Tool trace entries contain real Workspace dispatch results, never fabricated reads.
"""
import json
import re
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from writing_agent.artifacts import extract_prose, markdown_graph
from writing_agent.scoring import mechanical_score
from writing_agent.workspace import Workspace, dispatch

ROOT = Path('/home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation')
scenarios = {s['id']: s for s in json.load(open('data/scenarios/development.json'))}
payload = json.loads((ROOT / 'worked-examples.json').read_text())
examples = payload['examples']
assert set(payload) == {'schema_version', 'examples'} and payload['schema_version'] == 1
assert len(examples) == 20 and len({e['case_id'] for e in examples}) == 20
expected = {'case_id', 'family', 'channel', 'brief', 'response', 'files', 'satisfies', 'why_it_is_good'}
report = []
trace_results = []
openings = []
required_total = 0
unmet = []
queries = {'F5-03': ['captain', 'creditor'], 'F5-05': ['seal', 'drawing'], 'F5-07': ['well']}
for e in examples:
    cid = e['case_id']
    s = scenarios[cid]
    assert set(e) == expected, cid
    assert e['family'] == s['family'] and e['brief'] == s['visible']['brief']
    assert s['role'] == 'development'
    assert set(s['visible']) == {'brief', 'initial_files', 'followups', 'tools', 'budgets', 'prose'}
    assert all(isinstance(f, str) for f in s['visible']['followups'])
    assert set(s['visible']['tools']) <= {'list_dir','read_file','search','write_file','patch_file'}
    assert e['response'].strip() and e['why_it_is_good'].strip()
    for path, text in e['files'].items():
        assert not Path(path).is_absolute() and '..' not in Path(path).parts and text.strip()
    selectors = {p['id'] for p in s['visible']['prose']}
    assert all(c['artifact'] in selectors for c in s['labels']['checks'] if 'artifact' in c)
    req = {c['id'] for c in s['labels']['checks'] if c.get('required')}
    assert req == {c['check_id'] for c in e['satisfies']}
    assert len(req) == len(e['satisfies'])
    required_total += len(req)
    for c in e['satisfies']:
        assert isinstance(c['met'], bool) and c['evidence'].strip()
        assert '“' in c['evidence'] and '”' in c['evidence'], (cid, c)
        if not c['met']:
            assert 'UNMET' in c['evidence']
            unmet.append(f"{cid}:{c['check_id']}")
    if s['instruction_specificity'] == 'loose':
        assert re.search(r'assum|I treated|I chose', e['response']), cid
    md = ROOT / 'worked-examples' / f'{cid}.md'
    assert md.is_file() and md.stat().st_size
    assert e['response'] in md.read_text()
    for content in e['files'].values():
        assert content.rstrip('\n') in md.read_text()
    with TemporaryDirectory(prefix=f'replay-{cid}-', dir=ROOT) as temp:
        ws = Workspace(Path(temp), max_total_bytes=s['visible']['budgets']['max_total_bytes'])
        for path, text in s['visible']['initial_files'].items():
            ws.write_file(path, text)
        before = ws.snapshot()
        trace = []
        def call(name, **arguments):
            assert name in s['visible']['tools'], (cid, name)
            observation = dispatch(ws, name, arguments)
            assert observation['ok'] and observation['valid'], (cid, observation)
            trace.append({'type': 'tool', 'call': {'id': f'{cid}-{len(trace)+1}',
                          'function': {'name': name, 'arguments': arguments}},
                          'observation': observation})
        if s['family'] == 'F5' and s['visible']['initial_files']:
            call('read_file', path='kb/index.md')
            for query in queries[cid]:
                call('search', query=query, path='kb')
            for path in s['visible']['initial_files']:
                if path != 'kb/index.md':
                    call('read_file', path=path)
        elif s['visible']['initial_files']:
            for path in s['visible']['initial_files']:
                call('read_file', path=path)
        for path, content in e['files'].items():
            call('write_file', path=path, content=content)
        after = ws.snapshot()
        changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
        assert changed == sorted(e['files']), (cid, changed)
        read_tokens = sum(len(json.dumps(t['observation']['result'], ensure_ascii=False).split())
                          for t in trace if t['call']['function']['name'] in {'read_file', 'search', 'list_dir'})
        assert len(trace) <= s['visible']['budgets']['max_tool_calls']
        assert read_tokens <= s['visible']['budgets']['max_read_tokens']
        # One replay tool call per step plus final reply: conservative upper bound.
        assert len(trace) + 1 <= s['visible']['budgets']['max_steps']
        result = {'status': 'completed', 'output': e['response'], 'before': before, 'after': after,
                  'trace': trace, 'turns': [{'output': e['response'], 'snapshot': after}],
                  'provenance': 'synthetic', 'model': {'kind': 'scripted', 'id': 'authored-demonstration-replay'},
                  'attempted_tool_calls': len(trace), 'read_tokens': read_tokens,
                  'read_tokenizer': 'whitespace-v1'}
        card = mechanical_score(s, result)
        artifacts = extract_prose(result, s['visible']['prose'])
        assert all(a['status'] == 'ok' for a in artifacts), cid
        mechanical = [c for c in card['checks'] if c['method'] in {'deterministic', 'reference_match'}]
        assert all(c['status'] == 'ok' and c['passed'] for c in mechanical), (cid, mechanical)
        words = len(artifacts[0]['text'].split()) if artifacts else None
        kb = {p: t for p,t in after.items() if p.startswith('kb/')}
        graph = markdown_graph(kb, ['kb/index.md']) if s['family'] == 'F4' else None
        if graph:
            assert graph['valid_fraction'] == graph['reachability'] == 1
            assert not graph['orphans'] and not graph['missing_entrypoints']
        primary = artifacts[0]['text'].strip() if artifacts else (
            next(iter(e['files'].values())).strip() if e['files'] else e['response'])
        # Distinct first clauses are also read manually; normalization ignores formatting.
        normalized = re.sub(r'^[\s#*\d.)“]+', '', primary)
        clause = re.split(r'[.!?;\n]|,(?:\s|$)', normalized, maxsplit=1)[0].casefold().strip('” ')
        response_start = re.sub(r'^[\s#*\d.)“]+', '', e['response'])
        response_clause = re.split(r'[.!?;\n]|,(?:\s|$)', response_start, maxsplit=1)[0].casefold().strip('” ')
        openings.append({'case_id': cid, 'primary_first_clause': clause,
                         'response_first_clause': response_clause,
                         'primary_first_line': primary.splitlines()[0]})
        report.append({'case_id': cid, 'channel': e['channel'], 'words': words,
                       'kb_words': sum(len(t.split()) for t in kb.values()) if graph else None,
                       'deterministic': [{'id': c['id'], 'passed': c['passed'], 'status': c['status']}
                                         for c in mechanical],
                       'semantic_pending': len([c for c in card['checks'] if c['status'] == 'pending']),
                       'unmet_author_checks': [c['check_id'] for c in e['satisfies'] if not c['met']],
                       'changed_paths': changed, 'tool_calls': len(trace), 'read_tokens': read_tokens,
                       'wiki_graph': graph, 'scorecard': card})
        trace_results.append({'case_id': cid, 'scope': 'first_brief_only', 'result': result})
        outcomes = ', '.join(c['id'] + '=PASS' for c in mechanical) or 'none declared'
        size = f'prose_words={words}' if words is not None else f"kb_words={report[-1]['kb_words']}" if graph else 'planning response'
        print(f'{cid}: {outcomes}; {size}; semantic_pending={report[-1]["semantic_pending"]}')
assert len({o['primary_first_clause'] for o in openings}) == 20, openings
assert len({o['response_first_clause'] for o in openings}) == 20, openings
families = dict(sorted(Counter(e['family'] for e in examples).items()))
channels = dict(sorted(Counter(e['channel'] for e in examples).items()))
assert set(families) == {'F1','F2','F3','F4','F5'} and set(channels) == {'file','reply'}
assert len([e for e in examples if e['family'] in {'F1','F2','F5'}]) >= 10
checks = [c for r in report for c in r['deterministic']]
summary = {'examples': 20, 'families': families, 'channels': channels,
           'loose_examples': sum(scenarios[e['case_id']]['instruction_specificity'] == 'loose' for e in examples),
           'scene_examples': 14, 'required_checks': required_total, 'author_unmet': unmet,
           'deterministic_passed': len(checks), 'deterministic_failed': 0,
           'unique_primary_openings': 20, 'unique_response_openings': 20,
           'search_calls': sum(t['call']['function']['name']=='search' for r in trace_results for t in r['result']['trace'])}
(ROOT / 'verification-results.json').write_text(json.dumps({'summary': summary, 'cases': report, 'openings': openings}, ensure_ascii=False, indent=2)+'\n')
(ROOT / 'tool-replays.json').write_text(json.dumps({'schema_version': 1, 'origin': 'Offline replay of assistant-authored targets; no candidate model was run.', 'replays': trace_results}, ensure_ascii=False, indent=2)+'\n')
print('SUMMARY ' + json.dumps(summary, sort_keys=True))

#!/usr/bin/env python3
"""Reproduce the handoff checker's negative tests; no hardware/network execution.
Temporary copies stay under this package and are deleted on exit.
This is not the production system's 32-scenario acceptance suite.
"""
from pathlib import Path
import importlib.util, tempfile, shutil, json
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('handoff_checker',root/'tools/validate_handoff.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def find(b,rid):return next(x for x in b['records'] if x['record_id']==rid)
def mutate_json(dst,func):
 p=dst/'examples/demo_bundle.json'; b=json.loads(p.read_text());func(b);p.write_text(json.dumps(b,indent=2)+'\n')

def mutation(case,dst):
 if case=='unknown_record_field':mutate_json(dst,lambda b:find(b,'cfg-demo').update({'unexpected_field':True}))
 elif case=='tested_source_mismatch':mutate_json(dst,lambda b:find(b,'run-demo-a')['payload']['source']['tested_commit'].update({'hex':'f'*40}))
 elif case=='unknown_spill_filled_with_zero':mutate_json(dst,lambda b:find(b,'run-demo-a')['payload']['analysis_metrics'][0].update({'value':0}))
 elif case=='incorrect_latency_summary':mutate_json(dst,lambda b:find(b,'run-demo-a')['payload']['timing'].update({'median_us':1}))
 elif case=='fixture_claiming_trusted_worker':mutate_json(dst,lambda b:find(b,'run-demo-a')['payload'].update({'provenance':'trusted_worker'}))
 elif case=='artifact_tampering':
  p=dst/'examples/artifacts/run-demo-a-samples.json';p.write_text(p.read_text()+' ')
 elif case=='wrong_config_hash':mutate_json(dst,lambda b:find(b,'cfg-demo')['payload'].update({'config_hash':'sha256:'+'0'*64}))
 elif case=='cross_pr_membership_error':mutate_json(dst,lambda b:find(b,'snapshot-demo-102')['payload'].update({'commit_refs':['commit-demo-a','commit-demo-c']}))
 elif case=='duplicate_record_id':mutate_json(dst,lambda b:b['records'].append(find(b,'cfg-demo')))
 elif case=='artifact_path_traversal':mutate_json(dst,lambda b:find(b,'run-demo-a')['payload']['artifacts'][0].update({'uri':'../../outside.json'}))
 elif case=='invalid_boolean_dimension':mutate_json(dst,lambda b:find(b,'cfg-demo')['payload']['problem'].update({'n':True}))
 elif case=='duplicate_json_key':
  p=dst/'examples/demo_bundle.json'; s=p.read_text();s=s.replace('"bundle_version":','"bundle_version": "0.2.0", "bundle_version":',1);p.write_text(s)
 else: raise ValueError(case)
cases=['unknown_record_field','tested_source_mismatch','unknown_spill_filled_with_zero','incorrect_latency_summary','fixture_claiming_trusted_worker','artifact_tampering','wrong_config_hash','cross_pr_membership_error','duplicate_record_id','artifact_path_traversal','invalid_boolean_dimension','duplicate_json_key']
res=[]
with tempfile.TemporaryDirectory(prefix='.qa-negative-',dir=root) as t:
 for i,c in enumerate(cases):
  dst=Path(t)/str(i);shutil.copytree(root,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.qa-negative-*'))
  mutation(c,dst)
  try:m.validate(dst)
  except Exception as e:res.append({'case':c,'expected':'reject','actual':'rejected','exception_type':type(e).__name__})
  else:res.append({'case':c,'expected':'reject','actual':'INCORRECTLY_ACCEPTED'})
report={'scope':'handoff checker only, not production-system acceptance tests','status':'passed' if all(x['actual']=='rejected' for x in res) else 'failed','negative_cases_executed':len(res),'results':res}
(root/'HANDOFF_NEGATIVE_TESTS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert report['status']=='passed'

"""Recorded tripod closure must preserve all hand commands and require support."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from shape_s8v4 import ShapeS8V4
root=Path(__file__).resolve().parent
rows=[json.loads(l) for l in (root/'demo/control.jsonl').open()]
rows=[r for r in rows if r['phase'] in ('close','preload')]
def fixture(supported):
 x=object.__new__(ShapeS8V4)
 x.controller=SimpleNamespace(deactivate=lambda:None)
 x.recorded_closure=rows;x.closure_state={};x.support_history=[supported]*5
 x.sim=SimpleNamespace(lower=np.full(18,-7),upper=np.full(18,7),target=np.zeros(18))
 return x
x=fixture(True);result=list(x._recorded_tripod_closure())
assert len(result)==len(rows)
for before,after in zip(rows,result):
 assert before['phase']==after['phase']
 np.testing.assert_array_equal(before['sent_target_rad'][6:],after['sent_target_rad'][6:])
assert np.ptp(np.array([r['sent_target_rad'][12:14] for r in result]),axis=0).max()>.01
try:list(fixture(False)._recorded_tripod_closure())
except ValueError as e:assert 'three-finger' in str(e)
else:raise AssertionError('Missing support must block lift')
print('PASS: exact recorded thumb/index/middle closure at one row per tick; missing support blocks lift')

Astra + frozen tactile PPO: first online integration trial

Run from any terminal:
cd /home/sai/Documents/ChatGPT/Shiqi/tactile-genesis
bash experiments/astra_ppo_cuboid_125g_mu05/run.sh

The launcher starts a NEW simulator on http://127.0.0.1:8775, then calls Astra.
No Astra request was made during preparation. The script uses existing ChatGPT
login under /home/sai/.codex, model gpt-6-astra, medium reasoning.

Object: cuboid 45 x 60 x 45 mm,125g, both robot/object friction0.5.
Demonstration: actual cuboid successful teacher replay, fresh v3 tactile,
7 chronological keyframes /21 images. Source hashes checked at startup.
PPO: frozen seed1 tactile checkpoint selected on original validation,
not seed2 picked from hard test outcomes. Checkpoint and trajectory in config.json.

Controls: physics300Hz, observations60Hz,12-frame history, PPO15Hz.
Astra controls high-level skills, hand shape and initial close gain.
PPO takes gain ownership at preload; original closure, support gate and
RuleSupportController joint-target corrections remain active.
Arm and non-pinch finger gains and effort limits are unchanged.
Hold is180 samples at60Hz =3s; original40Hz behavior remains120 samples.

Experimental limitations:
- Online executor does not know full future joint trajectory. PPO next-target
  difference is explicitly zero (causal held target). Phase-local progress uses
  reference phase start + elapsed time, clipped to reference boundary.
  This is NOT identical to the offline demonstrated input distribution.
- Cuboid reference uses its actual images, hand targets and align pose, but
  retains prior nominal visible-surface anchor at the same workspace location.
  This is approximate registration, not newly measured cuboid RGB-D calibration.
  Astra must assess live alignment and can abort.
- Local CPU checks and HTTP skill startup passed; an Astra-controlled grasp
  has not been run or claimed successful.

Outputs per run:
outputs/online_astra_ppo/<label>/astra.log, server.log, preflight.log
outputs/online_astra_ppo/<label>/bridge/online_ppo.jsonl (observations/actions/gains)
outputs/s8v4_agent/<label>/ (API inputs, replies, instructions, final score)
outputs/s8v4_genesis/<label>/ (commands, physical evaluations, frames)

Read-only check without Astra:
PYTHONPATH=ur7e_xhand_sim:. .venv/bin/python experiments/astra_ppo_cuboid_125g_mu05/check.py

Close or Ctrl+C launcher: stops its own simulator child. No real hardware.

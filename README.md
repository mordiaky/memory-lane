# MemoryLane

A caregiver-facing life-story and reminiscence companion tool for families of people living with dementia.

---

> ### ⚠️ Important — this is not a medical device
>
> MemoryLane is a **wellness and organization tool**. It does **not** diagnose, treat, cure, or prevent any medical condition, including Alzheimer's disease or any other form of dementia. It is not a substitute for professional medical advice, diagnosis, or therapy. Always consult a qualified clinician for medical decisions about a loved one's care.
>
> The recommendations surfaced by this tool (anchor memories, visit plans, at-risk lists) are heuristic suggestions intended to help a caregiver plan a reminiscence session. Every suggestion should be reviewed by the caregiver before acting on it.

---

## What it does

MemoryLane helps families:

1. **Build a life-story archive** for a loved one — each memory is stored with a title, description, approximate era, an emotional tone, and a three-point emotional arc (how it *felt* from start to peak to end, not just a single tag).
2. **Plan a visit** — when you go to see them, MemoryLane suggests which memories are likeliest to land well today, and which are fading and worth revisiting before they slip further.
3. **Find a comfort anchor** — if the patient is agitated or distressed, MemoryLane can recommend the most reliably positive, strongly-connected memory to bring up.
4. **Track what lives and what fades** — as you log reactions during visits, the system quietly tracks which memories are still vivid and which are slipping, so the family can reinforce the at-risk ones while they still land.
5. **Respect hard-won boundaries** — mark memories that cause distress once, and MemoryLane will never recommend them again.

## How it works under the hood

MemoryLane is built on [Living Memory Dynamics (LMD)](https://github.com/mordiaky/LMD), a research library that models memories as living entities with metabolic energy, emotional trajectories, and resonance coupling between memories. LMD is the *engine*; MemoryLane is the product wrapped around it:

- Each memory becomes an LMD `LivingMemory` with a `ValenceTrajectory` built from the three valence scores.
- Reactions logged during visits nudge that memory's metabolic energy up or down.
- Energy translates to a human-readable **status** (vivid → active → dormant → fading → ghost).
- Coupling between memories means surfacing one emotionally anchored memory can lift neighbors too — that's what powers the anchor recommendation.

None of this requires a language model or a cloud API. Everything runs locally on your machine.

## Install

Requires Python 3.10 or newer.

```bash
git clone https://github.com/mordiaky/memory-lane.git
cd memory-lane
pip install -e ".[dev]"
```

## Quick tour via the CLI

```bash
# Create a patient record
memory-lane add-patient "Eleanor Thomas" --birth-year 1942

# Take the id it prints and add some memories
memory-lane add-memory <PATIENT_ID> "Wedding day" "A sunny July afternoon in 1965." \
    --tone joyful --valence-peak 0.95 --approximate-year 1965

memory-lane add-memory <PATIENT_ID> "Garden roses" "Late summer rose garden." \
    --tone joyful --valence-peak 0.9

memory-lane add-memory <PATIENT_ID> "Hospital year" "A long illness." \
    --tone difficult

# See what's stored
memory-lane list-memories <PATIENT_ID>

# Start a visit session
memory-lane start-session <PATIENT_ID> --caregiver-name Daughter
# Log reactions against memories (copy memory ids from list-memories)
memory-lane log-reaction <SESSION_ID> <MEMORY_ID> recognized_positive \
    --notes "She named all the rose varieties."

memory-lane end-session <SESSION_ID> --summary "Good visit."

# Ask for a comfort anchor if she becomes upset
memory-lane anchor <PATIENT_ID>

# Plan the next visit
memory-lane visit-plan <PATIENT_ID>

# See which memories are at risk
memory-lane fading <PATIENT_ID>

# Attach a photo or audio clip to a memory
memory-lane attach-media <MEMORY_ID> /path/to/wedding.jpg

# Get an end-of-visit summary
memory-lane visit-report <SESSION_ID>

# Browse memories grouped by life era
memory-lane eras <PATIENT_ID>

# Bulk-import existing memories from a spreadsheet
memory-lane import-csv <PATIENT_ID> family-memories.csv

# Export the full archive (family keeps control of their data)
memory-lane export <PATIENT_ID> --format json --output archive.json
memory-lane export <PATIENT_ID> --format csv --output memories.csv
```

### CSV format for import and export

Both `import-csv` and `export --format csv` use the same column set:

```
title, description, tone, approximate_year, era_label,
valence_start, valence_peak, valence_end, tags
```

Only `title` and `description` are required. `tone` is one of
`joyful`, `bittersweet`, `difficult`, `neutral`. Unknown tones
default to `neutral` with a warning.

## Run the web app

```bash
uvicorn memory_lane.api:app --reload
```

Then open:

- **http://127.0.0.1:8000/** — the caregiver web UI (patient list, life-story archive, visit sessions, reports).
- **http://127.0.0.1:8000/docs** — interactive JSON API docs.
- **http://127.0.0.1:8000/api/…** — the JSON API itself.

The web UI and JSON API share the same service layer, so anything you can do through the API you can also do through the browser (and vice versa).

### The visit flow, in the UI

1. From the patient dashboard, click **Start visit** (optionally naming the caregiver).
2. The session page shows suggested memories, ordered by how much they need reinforcement.
3. For each one, click a reaction button: **Recognized · positive** / **Neutral** / **Caused distress** / **Did not recognize**. The card updates in place (no page reload) and the memory's status is recalculated.
4. When you're done, fill in an optional summary and click **End visit & view report**.
5. The report page shows highlights, concerns, and follow-up suggestions for next time.

## Data model

| Table | Purpose |
|---|---|
| `patients` | One row per loved one being supported. |
| `memories` | Life-story entries: title, description, emotional tone + three-point valence arc, era, optional media pointers, energy/status, distress flag. |
| `sessions` | One row per caregiver visit. |
| `reactions` | One row per observed reaction to a memory surfaced during a visit. The primary signal that drives energy updates. |

## What's intentionally **not** here

This is phase 1. Things that are *explicitly out of scope* for this release:

- **No cloud / no sync.** All data stays in a local SQLite file (`~/.memory-lane/data.db` by default).
- **No real-time distress detection.** The patient's reactions are logged by the caregiver; MemoryLane never observes the patient directly.
- **No clinical recommendations.** Suggestions are heuristic and aimed at helping a caregiver plan a visit, not at guiding clinical care.
- **No diagnosis.** Nothing here infers a patient's cognitive state. Status labels describe a memory's recognition pattern, not the patient.

## Phase 2 and beyond

If early feedback suggests MemoryLane is genuinely useful, planned future work includes:

- A web UI so the family doesn't need to use the CLI.
- Shared accounts so multiple relatives can contribute memories.
- Optional HIPAA-compatible hosted storage.
- IRB-governed pilots with partner memory clinics to generate outcome data.

None of those require turning MemoryLane into a medical device; they extend it as a caregiver-assist tool.

## License

[MIT](LICENSE). See the LICENSE file for full terms.

## Acknowledgements

MemoryLane is built on [Living Memory Dynamics (LMD)](https://github.com/mordiaky/LMD), invented by Joshua R. Thomas.

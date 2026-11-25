# Agent integration

Start with the two JSON fixtures under `data/`. Implement the protocols in `careerpilot/schemas.py`, supply the new object to `Orchestrator(...)` in the notebook, then run the workflow tests. Each agent owns its prompts, fixtures and module tests. Use one branch per module and agree on contract changes together before merging.

| Owner | Entry points | Contract at the handoff |
| --- | --- | --- |
| Member 1 · Orchestrator | `classify`, `start`, `select_job`, `clarify`, `approve`, `revise`, `reject` | Validate decisions against the current session, reserve budget before a call, keep approved feedback IDs and emit trace events. |
| Member 2 · Discovery and matching | `discover`, `match` | Return read-only job snapshots and one eligibility judgement per supplied job. Sort match results by relevance. Reference confirmed evidence IDs. Explain the unresolved requirement when eligibility is unknown. |
| Member 3 · Application preparation | `prepare` | Receive the selected job, evidence, previous draft and review. Return a package for the same job, advance its version on edits, and link claims to evidence. Use a null answer for an unresolved form field. |
| Member 4 · Verification and learning | `review` | Return `pass`, `revise`, `human_required` or `blocked`. Include actionable repair instructions or a clarification question. Resolve approved feedback IDs through the feedback store when integrating few-shot reuse. |

Each specialist method accepts a keyword-only `runtime`. Call `runtime.reserve_model_call()` immediately before each model call. The runtime exposes approved feedback IDs as a read-only tuple. The classifier receives the goal text and a flag indicating whether a job was supplied. The job description remains a separate field for specialist agents.

## Route and checkpoint rules

| Request | First work after parsing | Where it stops |
| --- | --- | --- |
| Search | Discover and match fixture jobs | Ranked results with eligibility and evidence references |
| Match supplied jobs | Match the supplied records | Assessments only |
| Prepare after a search | Discover and match, then ask the candidate to select | Unknown facts pause the draft. A reviewed draft pauses for approval. |
| Prepare one supplied job | Match that job directly | Same clarification and approval gates |
| Revise | Use a previous package and its job | Verify the new version, then request fresh approval |

A caller can persist a checkpoint with `save_session(state, path)` and restore it with `load_session(path)`. Recreate the compatible workflow and invoke the action for that checkpoint. A saved session contains source text, drafts and candidate answers. It belongs in a private local directory. The trace holds operational labels and usage data and can be inspected separately. The delivered prototype uses one owner per session.

An explicit search reads fresh results even if the caller supplies an earlier job. A successful search can return no results. Discovery ranks the candidate pool before the orchestrator applies the requested shortlist size. Match-only requests assess each supplied job.

## Bring another dataset

Pass a local catalogue to `ReplayDiscovery(load_jobs(path))` and pass the selected profile to `start`. Every job carries its own requirements. Set `requirements_complete` when the complete set of mandatory conditions has been recorded. A fully described job can have zero mandatory conditions. With a partial description the matcher can identify an explicit conflict, while a positive eligibility decision remains pending.

| Field | Meaning |
| --- | --- |
| `role_aliases` | Other titles for this job in the catalogue. The matcher uses the title and these aliases during discovery. |
| `fact_key` | A shared key for a requirement and a candidate fact, such as a licence, language or length of experience. Record and source IDs retain their own independent meanings. |
| `operator`, `expected` | Compare typed values for equality or numeric thresholds (`eq`, `gte`, `lte`). Job minimums measured in years can set `category="experience_years"` for comparison with the candidate's search ceiling. |
| `confirmed`, `boolean_value`, `fact_value` | The source can confirm a positive or negative answer. Use one typed value per record. A source that gives no answer leaves the requirement unresolved. |
| `ats_question` | Optional question supplied by the job. Each answer links to the evidence for that particular question. Sensitive record text stays out of the resume and cover letter. |

The offline classifier supports common English search, match, preparation and revision requests. The local matcher evaluates titles, aliases, location, preferences, the recognized experience ceiling and normalized requirements. When parsing identifies a hard condition that the matcher cannot evaluate, it appears in the gaps and eligibility stays unresolved until the condition is checked. The application package and trace keep the same shape across catalogues.

## Demo and evaluation boundaries

Replay uses invented listings and a known evidence set. Its scores and text are examples for integration. The tests establish routing, budgets, recovery and approval behavior. Member 2 supplies live search and retrieval, Member 3 supplies tailored material generation, and Member 4 supplies independent verification, memory and held-out evaluation. For the 29 November integration run, replace one adapter at a time and retain these tests as regression cases.

For the 12-minute video, Member 1 can use the diagram for the first minute and run the two notebook routes for the next two minutes. Show which calls disappeared for a supplied job, the exact question that stopped the draft and the version recorded on approval.

# Brief Agent V2 study starter

This version starts with MLflow tracing configured and uses GPT-5.6 Terra.
It researches live Hacker News and GitHub sources on normal runs.

The organizer supplies the same private starter bundle to everyone. No email
list or personal assignment file is required. The public tag is unpublished.
Use the supplied bundle and its exact tag when cloning into a new directory.

## Set up

From your cloned starter:

```bash
bash setup-v2.sh
```

Use your existing dogfood login or complete the SDK OAuth login yourself.
Setup creates your own experiment and trace tables, then imports the five real
baseline runs. This provisioning can take several minutes, with product waits
recorded separately. The five runs represent simulated production weeks.
Setup makes zero agent or model calls and preserves progress after interruption.
Run the same command again to resume. Python 3.12 and locked dependencies from
the Databricks internal package proxy are required.

Setup prints the experiment URL, five brief paths, live rerun command, and
`.study/manifest.json`. The manifest contains no credentials. Use the MLflow
time range October 1, 2026, 22:50-23:10 UTC to see every baseline. Expand it
through the current time when comparing new runs.

## Run normally

```bash
bash run-v2.sh
```

This runs the live agent with the tested MLflow batching settings. Briefs appear
under `output/`; traces append to your experiment. A repeat setup preserves your
code edits. To start a fresh attempt, clone the same immutable bundle into a new
directory and run setup there. Keep the old code, traces, briefs, and transcript.
Do not delete `.study` to reset an attempt.

The optional `--assignment /path/to/assignment.json` route remains available
for previously prepared attempts. It is not needed for self-service setup.

# Brief Agent study starter

This version starts with MLflow tracing configured and uses GPT-5.6 Terra.
It researches live Hacker News and GitHub sources on normal runs.

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

Setup opens your experiment in your browser and prints an existing brief under
`AGENT OUTPUT EXAMPLE:`. It also prints the five brief paths and the command to
run the agent again. These instructions are saved in `.study/ready.md`.
For a terminal without a browser, use `bash setup-v2.sh --no-open`.
If older traces are hidden, widen the experiment's time filter.

## Run normally

```bash
bash run-v2.sh
```

Edit the code under `plain/`, then run that command after each change.
Briefs appear under `output/`; traces append to your experiment.
A repeat setup preserves your code edits. To start a fresh attempt, clone the same release into a new
directory and run setup there. Keep the old code, traces, briefs, and transcript.
Do not delete `.study` to reset an attempt.

The optional `--assignment /path/to/assignment.json` route remains available
for previously prepared attempts. It is not needed for self-service setup.

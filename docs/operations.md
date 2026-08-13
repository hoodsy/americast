# Operations

Where the data lives, what runs where, and what it costs. Everything in
this file is reproducible from the AWS CLI; nothing was clicked in a
console except the account plan and the GitHub secrets.

## The two kinds of work

The project has two workloads with opposite shapes, and they live in
different places for that reason.

| | The daily loop | Backfills |
|---|---|---|
| What | one HRRR run, forecast, grade | thousands of historical runs |
| How long | ~11 minutes | 12 hours to 4 days |
| Where | **GitHub Actions** | **EC2 spot worker** |
| Why there | free scheduling, logs, retries; nothing to leave running | too long for Actions' limits; wants a box that can be stopped |

Putting the daily loop on EC2 would mean a machine awake at 09:00 UTC
every day, which is the opposite of stop-when-idle. Putting a four-day
backfill on Actions is not possible at all.

## Storage

Everything resolves through `americast/storage.py` against one
environment variable:

```sh
AMERICAST_DATA_ROOT=data                      # local, the default
AMERICAST_DATA_ROOT=s3://americast-data/americast   # CI and the worker
```

Bucket `americast-data`, **us-west-2**, versioning on.

```
americast/
  caiso/        labels, curtailment
  registry/     plants_ciso.parquet
  hrrr/         one parquet per run, plus manifest.csv
  train/        table.parquet
  model/        p10/p50/p90 boosters + meta.json
  eia860/ eia923/
  live/         forecasts.parquet, scores.parquet
  public/       <- anonymously readable, CORS enabled
    regions.json
    caiso/forecast.json
    caiso/scoreboard.json
  code/         americast.tar.gz, what the worker unpacks
```

**Only `public/` is world-readable.** The bucket policy grants anonymous
`GetObject` on that prefix and nothing else — verified 200 inside, 403
outside. An object is public because of where it was written, not
because someone set an ACL.

## The worker box

`r5.large` spot in **us-east-1**, tagged `Project=americast`.

Four decisions, each with a reason worth keeping:

- **us-east-1, not us-west-2.** `noaa-hrrr-bdp-pds` lives there, and a
  full backfill pulls ~630 GB. Same-region transfer is free; cross-region
  would cost ~$13 per backfill. The bucket stays in us-west-2 because the
  results going back are tiny.
- **`r5.large`, not `t3.xlarge`.** Same 16 GB for 24% less. The job is
  I/O-bound — 2.3 s of CPU per 23 s of wall — so 2 vCPU is ample and
  memory is the only real constraint at ~1.6 GB per worker.
- **Spot, interruption behaviour `stop`.** `pending()` skips completed
  runs, so an interruption costs one run. 70% off.
- **Runs write straight to S3.** The box needs 30 GB, not a terabyte,
  and a spot reclaim loses nothing.

**No credentials on the box.** An IAM instance role grants S3 access to
`americast/*` and permission to stop itself. Access is via SSM — no SSH
keys, no open ports, and the security group has zero ingress rules.

### Idle shutdown watches the process, not the CPU

`/usr/local/bin/americast-idle-check` runs every 5 minutes and stops the
instance after 30 minutes with no `americast.ingest` process.

It must not be CPU-based. These workers sit near zero CPU while waiting
on S3, so a low-CPU CloudWatch alarm would stop the box *mid-backfill*.

## Runbook

Everything below uses the `americast` profile — static keys, no expiry.
`aws login` sessions last about an hour and the C++ SDK behind pyarrow
cannot read them at all, so that profile is the one to use.

```sh
export AWS_PROFILE=americast
IID=$(aws ec2 describe-instances --region us-east-1 \
  --filters Name=tag:Project,Values=americast Name=instance-state-name,Values=running,stopped \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
```

**Start the box**

```sh
aws ec2 start-instances --region us-east-1 --instance-ids $IID
```

**Run something on it**

```sh
aws ssm send-command --region us-east-1 --instance-ids $IID \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["<your command>"]' \
  --query 'Command.CommandId' --output text
# then
aws ssm get-command-invocation --region us-east-1 \
  --command-id <id> --instance-id $IID --query StandardOutputContent --output text
```

**Start a backfill** (detached, so the SSM call returns)

```sh
export AMERICAST_DATA_ROOT=s3://americast-data/americast AWS_REGION=us-west-2
cd /opt/americast
nohup setsid uv run python -m americast.ingest.hrrr \
  --hours 6 --start 2025-03-01 --end 2026-08-12 --workers 8 \
  > /var/log/americast-backfill.log 2>&1 < /dev/null &
```

`--hours 0,12,18` is pass 3. Eight workers fits 16 GB at ~1.6 GB each.

**Check progress**

```sh
aws s3 ls s3://americast-data/americast/hrrr/ | grep -c parquet
```

**Ship code changes to the box**

```sh
git archive --format=tar.gz -o /tmp/americast.tar.gz HEAD
aws s3 cp /tmp/americast.tar.gz s3://americast-data/americast/code/americast.tar.gz
# then on the box: aws s3 cp .../americast.tar.gz /tmp/ && tar -xzf /tmp/... -C /opt/americast && uv sync
```

**Set the environment durably.** SSM commands run in a non-login shell,
so `/etc/profile.d/` is not sourced. The variables are in
`/etc/environment`, and any `send-command` should export them anyway.

## What it costs

| | |
|---|---|
| `r5.large` spot | **$0.0506/hr** |
| Finish the 18-month refetch (~510 runs, 8 workers) | ~12 h, **$0.62** |
| Pass 3 (3,951 runs) | ~95 h, **$4.79** |
| S3 storage, ~4 GB | **$0.10/month** |
| GitHub Actions daily loop | free tier |

A **$25/month budget alarm** emails at 50%. That is a wide margin over
the ~$5 expected, so it fires on something unexpected rather than on
normal use.

The costs that would dominate if these decisions were reversed: running
the box 24/7 (~$36/month), or fetching cross-region (~$13 per backfill).

## Identities

| Identity | What it is for | Credentials |
|---|---|---|
| account root | creating AWS resources | `aws login`, ~1 hour, browser |
| `americast-ci` | GitHub Actions, and managing the worker from a laptop | static keys, no expiry |
| `americast-worker` role | the box itself | instance role, no keys |

`americast-ci` holds two policies: S3 on `americast/*`, and start/stop
plus SSM scoped to instances tagged `Project=americast`. It cannot
create resources, which is deliberate — that needs root.

Two access keys exist on `americast-ci`. GitHub secrets hold the first,
the local profile holds the second. Both are valid; consolidate once a
workflow run has proven which one CI is using.

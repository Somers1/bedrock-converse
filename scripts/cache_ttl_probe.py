import argparse
import time
import uuid
import boto3
from botocore.exceptions import ClientError

FILLER = ("You are a diagnostic assistant used to measure Bedrock prompt cache behavior. "
          "The following text exists only to push the cached prefix above the model minimum. ") * 220

CONFIGS = [
    ("au.anthropic.claude-sonnet-4-6", "1h"),
    ("au.anthropic.claude-sonnet-4-6", "5m"),
    ("au.anthropic.claude-sonnet-4-5-20250929-v1:0", "1h"),
    ("au.anthropic.claude-haiku-4-5-20251001-v1:0", "1h"),
]


class Probe:
    def __init__(self, client, model_id, ttl):
        self.client = client
        self.model_id = model_id
        self.ttl = ttl
        self.marker = uuid.uuid4().hex
        self.calls = {}
        self.error = None

    @property
    def label(self):
        return f"{self.model_id}  ttl={self.ttl}"

    @property
    def system(self):
        point = {"type": "default"} if self.ttl is None else {"type": "default", "ttl": self.ttl}
        return [{"text": f"PROBE-{self.marker}\n{FILLER}"}, {"cachePoint": point}]

    def call(self, phase):
        try:
            resp = self.client.converse(
                modelId=self.model_id,
                system=self.system,
                messages=[{"role": "user", "content": [{"text": "Reply with the single word OK."}]}],
                inferenceConfig={"maxTokens": 5, "temperature": 0},
            )
            usage = resp["usage"]
            self.calls[phase] = {
                "input": usage.get("inputTokens", 0),
                "read": usage.get("cacheReadInputTokens", 0),
                "write": usage.get("cacheWriteInputTokens", 0),
                "raw": usage,
            }
        except ClientError as e:
            self.error = e.response["Error"]["Code"] + ": " + e.response["Error"]["Message"]
            self.calls[phase] = None

    @property
    def verdict(self):
        if self.error:
            return f"ERROR ({self.error})"
        cold, warm, gapped = self.calls.get("A"), self.calls.get("B"), self.calls.get("C")
        if not (cold and warm and gapped):
            return "incomplete"
        if warm["read"] == 0:
            return "caching NOT working (no read even immediately) — format/model rejected the cachePoint"
        if gapped["read"] > 0:
            return f"SURVIVED the gap → {self.ttl} is HONORED (read {gapped['read']} tokens after gap)"
        return f"DIED within the gap → {self.ttl} silently behaves as ~5m (re-wrote {gapped['write']} tokens)"


def row(phase, c):
    if c is None:
        return f"    {phase}: (no data / error)"
    return f"    {phase}: input={c['input']:>6}  read={c['read']:>6}  write={c['write']:>6}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="ap-southeast-2")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--gap", type=int, default=360, help="seconds between warm and post-gap call (>300 to cross 5m)")
    args = ap.parse_args()
    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    client = session.client("bedrock-runtime", region_name=args.region)
    probes = [Probe(client, model_id, ttl) for model_id, ttl in CONFIGS]
    print(f"region={args.region} gap={args.gap}s  models={len(probes)}\n")
    print("Phase A — cold write (fresh unique prefix per config)")
    for p in probes:
        p.call("A")
        print(f"  {p.label}\n{row('A', p.calls['A'])}" + (f"  !! {p.error}" if p.error else ""))
    print("\nPhase B — immediate re-send (<5m): proves caching works at all for this model/format")
    for p in probes:
        if p.error:
            continue
        p.call("B")
        print(f"  {p.label}\n{row('B', p.calls['B'])}")
    print(f"\nWaiting {args.gap}s to cross the 5-minute boundary...")
    time.sleep(args.gap)
    print("\nPhase C — post-gap re-send: read>0 means the TTL survived the gap")
    for p in probes:
        if p.error:
            continue
        p.call("C")
        print(f"  {p.label}\n{row('C', p.calls['C'])}")
    print("\n================ VERDICT ================")
    for p in probes:
        print(f"  {p.label}\n    {p.verdict}")


if __name__ == "__main__":
    main()

"""Scenario calculator: evaluation instances are NOT the number of Add requests."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--strategy",default="adaptive")
    parser.add_argument("--profile",default="single_image_512")
    parser.add_argument("--add-requests",type=int,required=True)
    parser.add_argument("--search-requests",type=int,required=True)
    parser.add_argument("--total-upload-bytes",type=int,default=0)
    parser.add_argument("--upload-mbps",type=float,default=10)
    parser.add_argument("--answer-eval-seconds",type=float)
    parser.add_argument("--queue-seconds",type=float)
    parser.add_argument("--deadline",default="2026-11-04T23:59:00+08:00")
    args=parser.parse_args()
    if min(args.add_requests,args.search_requests,args.total_upload_bytes)<0 or args.upload_mbps<=0:
        parser.error("Counts must be nonnegative and bandwidth positive")
    rows=json.loads(args.report.read_text())["performance"]
    row=next(row for row in rows if row["strategy"]==args.strategy and row["profile"]==args.profile)
    api_add=args.add_requests/row["requests_per_second"]
    api_search=args.search_requests*row["search_p95_ms"]/1000
    upload=args.total_upload_bytes*8/(args.upload_mbps*1_000_000)
    results=[]
    remaining=(datetime.fromisoformat(args.deadline)-datetime.now(timezone.utc)).total_seconds()
    for name,multiplier in (("optimistic",1.),("central",1.5),("conservative",3.)):
        # Sum is a conservative sequential API/network scenario, not measured AML scheduling.
        api=(api_add+api_search+upload)*multiplier
        total=api+args.answer_eval_seconds+args.queue_seconds if args.answer_eval_seconds is not None and args.queue_seconds is not None else None
        results.append({"scenario":name,"slowdown_assumption":multiplier,"add_seconds":api_add*multiplier,
                        "search_seconds":api_search*multiplier,"network_upload_seconds":upload*multiplier,
                        "api_and_upload_seconds":api,"answer_eval_seconds":args.answer_eval_seconds,
                        "platform_queue_seconds":args.queue_seconds,"total_seconds":total,
                        "estimated_finish_before_deadline":total<remaining if total is not None else None})
    print(json.dumps({"kind":"scenario_estimate_not_official_schedule","deadline":args.deadline,
        "assumptions":vars(args)|{"report":str(args.report)},"scenarios":results,
        "limitations":"Local disk/CPU/HTTP rates do not predict cloud/network/AML capacity. Substitute real request counts. Official 5000 instances is not an Add count."},indent=2))


if __name__=="__main__": main()

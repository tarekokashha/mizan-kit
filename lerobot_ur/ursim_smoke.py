"""
URSim smoke test: prove the 500 Hz RTDE stream and measure its jitter.

    docker run --rm -d --name ursim -p 5900:5900 -p 6080:6080 -p 29999:29999 \
        -p 30001-30004:30001-30004 universalrobots/ursim_e-series
    python -m lerobot_ur.ursim_smoke --ip 127.0.0.1 --seconds 5

Output: received sample count, achieved rate, p50/p99/p99.9 inter-sample
jitter in microseconds, and the F/T vector. This is the first number in the
KEYSTONE README and the CI job's pass/fail gate (rate >= 480 Hz, p99.9 jitter
< 2500 us on a laptop; tighten once you have PREEMPT_RT).
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--hz", type=float, default=500.0)
    args = ap.parse_args()

    import dashboard_client
    import rtde_receive

    dash = dashboard_client.DashboardClient(args.ip)
    dash.connect()
    dash.powerOn()
    dash.brakeRelease()
    time.sleep(1.0)
    rr = rtde_receive.RTDEReceiveInterface(args.ip, frequency=args.hz)
    stamps = []
    t_end = time.perf_counter() + args.seconds
    last_ts = None
    while time.perf_counter() < t_end:
        ts = rr.getTimestamp()  # controller time, advances by 1/hz per sample
        if ts != last_ts:
            stamps.append(time.perf_counter())
            last_ts = ts
        else:
            time.sleep(0.0002)
    q = rr.getActualQ()
    ft = rr.getActualTCPForce()
    rr.disconnect()
    dash.disconnect()

    if len(stamps) < 10:
        print("FAIL: no RTDE samples received; is URSim running and powered on?")
        return 1
    d = np.diff(np.asarray(stamps)) * 1e6
    rate = len(stamps) / args.seconds
    jit = np.abs(d - 1e6 / args.hz)
    print(f"samples={len(stamps)} rate={rate:.1f} Hz")
    print(f"jitter us: p50={np.percentile(jit,50):.0f} p99={np.percentile(jit,99):.0f} p99.9={np.percentile(jit,99.9):.0f}")
    print(f"q={np.round(q,3).tolist()}")
    print(f"tcp_force={np.round(ft,2).tolist()}")
    ok = rate >= 0.96 * args.hz and np.percentile(jit, 99.9) < 2500
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse

from voice_twin.enrollment.enrollment import EnrollmentPipeline


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--speaker",required=True)
    p.add_argument("--audio",required=True)
    p.add_argument("--consent-verified",action="store_true")
    args=p.parse_args()
    pipeline=EnrollmentPipeline()
    sample=pipeline.prepare(args.audio)
    pipeline.consent(args.speaker,args.audio,args.consent_verified)
    print({"speaker":args.speaker,"duration_s":sample.quality.duration_s,"quality":sample.quality.score})


if __name__=="__main__":
    main()

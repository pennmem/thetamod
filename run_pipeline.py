#! /usr/bin/env python

from argparse import ArgumentParser
import logging
import os
from thetamod.pipeline import TMIPipeline


logger = logging.getLogger("cml")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    import dask

    parser = ArgumentParser()
    parser.add_argument("--clear-cache", "-c", action="store_true",
                        help="clear the cache prior to running")
    parser.add_argument("--subject", "-s")
    parser.add_argument("--experiment", "-x")
    parser.add_argument("--session", "-e")
    parser.add_argument("--rootdir", "-r", default=os.environ.get("RHINO_ROOT",
                                                                  "/"))
    args = parser.parse_args()

    rootdir = args
    if args.clear_cache:
        from cml_pipelines import memory
        memory.clear(warn=False)

    pipeline = TMIPipeline(args.subject, args.experiment, args.session,
                           args.rootdir)
    result = pipeline.run(get=dask.get)
    print(result)

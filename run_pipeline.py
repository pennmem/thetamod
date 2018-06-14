from argparse import ArgumentParser
import logging

from thetamod.pipeline import TMIPipeline

logger = logging.getLogger("cml")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    import dask

    parser = ArgumentParser()
    parser.add_argument("--clear-cache", "-c", action="store_true",
                        help="clear the cache prior to running")
    args = parser.parse_args()

    if args.clear_cache:
        from cml_pipelines import memory
        memory.clear(warn=False)

    subject = "R1260D"
    experiment = "catFR3"
    session = 0
    rootdir = "/Volumes/rhino_root"

    pipeline = TMIPipeline(subject, experiment, session, rootdir)
    result = pipeline.run(get=dask.get)
    print(result)

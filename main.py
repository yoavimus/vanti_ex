import logging
import logging.config
import logging.handlers
from Runners import SVM_runner
import time
import sys


def main():

    """
    main function
    :param run_type: int, type of run to preform. default 1 = market data run.
    :return:
    """
    logger = None
    try:
        data_file_path = sys.argv[1]
        time1 = time.time()

        # configure logging
        logging.handlers = logging.handlers
        logging.config.fileConfig('log_config.conf')
        logger = logging.getLogger("root")

        # initialize runner:
        testrun = SVM_runner.SVM_Runner(logger,data_file_path)
        testrun.lin_svm()

        time2 = time.time()
        logger.info("run time %s" % (time2-time1))

    except Exception as error:
        if logger is not None:
            logger.exception(error)
        else:
            print(error)


if __name__ == "__main__":
    main()
    # print('finished running main')

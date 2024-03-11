import time
import datetime

__all__ = ['current_time', 'get_run_name', 'PrintTime', 'Timer']


def current_time():
    return datetime.datetime.now().strftime("%d-%m-%Y: %H:%M:%S")


def get_run_name(name=None):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if name is None:
        name = 'run'
    output = f'{name}_{time_str}'
    return output


class PrintTime:
    def __init__(self, end=None, print_every=10, start=0, flush=True):
        self.start = start
        self.end = end
        self.flush = flush
        if end is not None:
            total = end - start
            if total // 10 < print_every:
                print_every = max(total // 10, 1)
        self.print_every = print_every
        self.prev_time = None
        self.count = 0

    def print(self, msg):
        if self.prev_time is None:
            self.prev_time = time.time()

        self.count += 1
        if self.count % self.print_every == 0:
            msg_info = f'{current_time()} | [{self.start + self.count:>2d}'

            if self.end:
                msg_info += f'/{self.end} '

            elapsed = time.time() - self.prev_time
            time_str = datetime.timedelta(seconds=int(elapsed))
            msg_info += f'] | {msg} | elapsed: {time_str}'

            rate = elapsed / self.count
            if rate != 0:
                if rate >= 1:
                    msg_info += f' | rate: {rate:.1f}s/it'
                else:
                    msg_info += f' | rate: {1 / rate:.1f}it/s'

                if self.end:
                    eta = rate * (self.end - self.start - self.count)
                    eta_str = datetime.timedelta(seconds=int(eta))
                    msg_info += f' | eta: {eta_str}'

            print(msg_info, flush=self.flush)
            return msg_info


class Timer:
    def __init__(self, msg: str = None, logger=None, debug: bool = True):
        """

        Args:
            msg: message to print with the time stats
            logger: logger
            debug: prints only if debug is set to True.

        Notes:
            Example usage:
                timer = Timer(msg='operation-1 description')
                # Do operation-1
                timer('operation-2 description')  # print operation-1 time stats and reset
                # Do operation-2
                timer() # print operation-1 time stats and reset
        """
        self.msg = msg
        self.start = time.time()
        self.logger = logger
        self.debug = debug

    def __call__(self, msg=None):
        """ Print existing stats and reset time.

        Args:
            msg: message to reset existing message.

        Notes:
            If msg was set previously (not None), then prints elapsed time and resets.
            Else, only resets time and sets current msg.
        """
        if self.msg is not None:
            self.time()
        self.msg = msg
        self.start = time.time()

    def time(self):
        if self.debug:
            elapsed = time.time() - self.start
            msg = f'{current_time()} | {self.msg} | {datetime.timedelta(seconds=elapsed)}'
            if self.logger:
                self.logger.log(msg)
            else:
                print(msg)

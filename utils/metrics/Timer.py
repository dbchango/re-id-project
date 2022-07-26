from timeit import default_timer as timer


class Timer:
    """
    This class will be used like a timer for time-consuming measurement
    """
    def __int__(self):
        self.time = None
        self.start_time = None
        self.end_time = None

    def start(self):
        """
        Put this function before the measured process starts
        :return:
        """
        self.start_time = timer()

    def end(self):
        """
        Put this function after the measured process finishes
        :return:
        """
        self.end_time = timer()

    def calculate_time(self):
        """
        This functions will return the measured time
        :return: measured time
        """
        self.time = self.end_time - self.start_time
        return self.time

import os
from processing import write_csv
import logging as lg


def label_iamges(path="", output_path="", file_name="labels"):
    """
    It will crate a .csv file with the images labels
    :param path: route of origin data
    :param output_path: route where labels file will be saved
    :return:
    """
    try:
        data = []

        for id in os.listdir(path):
            data.append(str(id))

        write_csv(os.path.join(output_path, f"{file_name}.csv"), data, [])
        lg.info(f"{file_name}.csv was correctly created on {output_path}")
    except Exception as e:
        lg.error("Exception occurred", exc_info=True)

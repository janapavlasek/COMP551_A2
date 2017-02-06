#!/usr/bin/env python
import re
import csv
import os


class CleanData(object):
    """Cleans the given data into a usable form."""
    def __init__(self):
        self.data = []

    def clean_data(self, input_file, output_file=None):
        """Cleans the data.

        Args:
            input_file: The path to the file to clean.
            output_file: If provided, file to save the output to.
        """
        with open(input_file, 'rb') as f:
            reader = csv.reader(f)
            self.data = list(reader)

            for element in self.data:
                # Remove the tags.
                element[1] = re.sub(r'<.+?>', "", element[1])
                element[1] = element[1].replace("\n", "")
                element[1] = element[1].strip(" ")

        if input_file is not None:
            with open(output_file, "w") as f:
                writer = csv.writer(f)
                for row in self.data:
                    writer.writerow(row)

    def get_data(self):
        """Returns a list of lists of the data, where each sub-list is a row of
        data."""
        if not self.data:
            print "Data is empty. You need to run clean_data first."
            return

        return self.data

    def get_data_from_csv(self, input_file="data/clean_train_input.csv"):
        """Returns a list of lists of the data, where each sub-list is a row of
        data, by loding past results from a CSV file (so you don't need to run
        the algorithm)."""
        if not os.path.exists(input_file):
            print "File", input_file, "does not exist. You need to run clean_data with this path as the output_file."
            return

        with open(input_file, 'rb') as f:
            reader = csv.reader(f)
            data = list(reader)

        return data


if __name__ == '__main__':
    cd = CleanData()
    cd.clean_data("data/train_input.csv", "data/clean_train_input.csv")

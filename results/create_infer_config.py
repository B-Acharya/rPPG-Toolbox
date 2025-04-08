import csv

def read_csv_file(file_path):
    """
    Reads a CSV file and returns its contents as a dictionary.

    :param file_path: Path to the CSV file.
    :return: Dictionary containing the CSV contents.
    """
    data = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header row
        headers = next(reader)

        for row in reader:
            if len(row) == 2:  # Ensure each row has a key and value
                key, value = row
                try:
                    # Convert value to a float if possible
                    data[key] = float(value)
                except ValueError:
                    data[key] = value  # Leave as string if conversion fails
    return data


if __name__=="__main__":

    file_path = read_csv_file("./pseudo-ppg.csv")
    print(file_path)
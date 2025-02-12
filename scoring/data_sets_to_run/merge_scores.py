import sys
import csv


if __name__ == "__main__":
    """
    Use to merge file1=radial distortion scores and file2=focal
    and FOV scores. Assumes file1 and file2 are in the same format,
    i.e. both are csv files that have the format
    [image name, radial distortion scored, focal length score, FOV score]
    at each row.
    """

    print "usage: python merge_auto_scores.py <rad dist file> <f and FOV dis file> <optional output name>"

    if len(sys.argv) < 4:
        output_name = 'auto_scores.csv'
    else:
        output_name = sys.argv[3]

    with open(sys.argv[1], 'rU') as f_rad:
        rad_reader = csv.reader(f_rad)

        with open(sys.argv[2], 'rb') as f_foc:
            foc_reader = csv.reader(f_foc)

            with open(output_name, 'wb') as f_write:
                writer = csv.writer(f_write)

                for (row1, row2) in zip(rad_reader, foc_reader):
                    merged = row1[:] + row2[1:]
                    writer.writerow(merged)

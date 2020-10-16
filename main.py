#!/usr/bin/python
import getopt
import sys
import docx
import unidecode
import os

from datetime import datetime
from pandas import DataFrame
from utilities_nlp import load_roles, process_line

students, co_facilitators, facilitators, non_role = load_roles()


def process_files(file_name: str, dir_name: str):
    files = []
    if dir_name is not None and dir_name != '':
        file_names = os.listdir(dir_name)

        for sub_file_name in file_names:
            files.append(f'{dir_name}/{sub_file_name}')
    elif file_name is not None:
        files.append(file_name)
    else:
        print('Error: It is necessary to pass input file (-i) or input directory (-d)')
        print('main.py [-i --ifile <inputfile>] [-d --dir <inputfolder>]')
        sys.exit(2)

    data_frame = DataFrame([], columns=['Role', 'Text'])

    for file in files:
        document = docx.Document(docx=file)

        data_vector = []

        for line in document.paragraphs:
            line_text = line.text.replace('¿', '')
            line_text = unidecode.unidecode(line_text)
            role, text = process_line(line_text, students, co_facilitators, facilitators, non_role)

            if role is not None and text is not None:
                if text != '':
                    data_vector.append([role, text])

        data_frame = data_frame.append(DataFrame(data_vector, columns=['Role', 'Text']))

    return data_frame


def process_params(argv):
    input_file = ''
    input_dir = ''

    try:
        opts, args = getopt.getopt(argv, "hi:d:", ["ifile=", "dir="])
        if len(opts) == 0:
            print('main.py [-i --ifile <inputfile>] [-d --dir <inputfolder>]')
            sys.exit(2)
    except getopt.GetoptError:
        print('main.py [-i --ifile <inputfile>] [-d --dir <inputfolder>]')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('main.py [-i --ifile <inputfile>] [-d --dir <inputfolder>]')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-d", "--dir"):
            input_dir = arg

    return input_file, input_dir


if __name__ == '__main__':
    input_file, input_dir = process_params(sys.argv[1:])
    df = process_files(input_file, input_dir)

    students_count = len(df[df['Role'] == 'Student'])
    facilitators_count = len(df[df['Role'] == 'Facilitator'])
    co_facilitators_count = len(df[df['Role'] == 'Co-Facilitator'])

    total = students_count + facilitators_count + co_facilitators_count

    print(f'Students Count: {students_count} ({students_count * 100 / total:.2f}%)')
    print(f'Facilitators Count: {facilitators_count} ({facilitators_count * 100 / total:.2f}%)')
    print(f'CoFacilitators Count: {co_facilitators_count} ({co_facilitators_count * 100 / total:.2f}%)')
    print(f'Total Rows: {total}')

    # Define output filename
    today = datetime.now()
    d1 = today.strftime("%Y%m%d%H%M%S")
    output_to = f'output/unbalanced_{d1}.csv'

    df.to_csv(output_to, index=False)

    output_filename = f'The CSV file has been created at {output_to}'
    print('\n' + '#' * len(output_filename))
    print(output_filename)


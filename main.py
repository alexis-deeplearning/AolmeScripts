#!/usr/bin/python
import getopt
import sys
import docx
import json
import re
import unidecode
import os

from datetime import datetime
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from pandas import DataFrame

students = []
facilitators = []
co_facilitators = []
non_role = []


def load_roles():
    with open('roles.json') as json_file:
        data = json.load(json_file)

        for p in data['Student']:
            students.append(p)

        for p in data['Facilitator']:
            facilitators.append(p)

        for p in data['CoFacilitator']:
            co_facilitators.append(p)

        for p in data['NonRole']:
            non_role.append(p)


def process_file(file_name: str, dir_name: str, output: str):
    files = []
    if dir_name is not None and dir_name != '':
        fnames = os.listdir(dir_name)

        for fname in fnames:
            files.append(f'{dir_name}/{fname}')
    elif file_name is not None:
        files.append(file_name)
    else:
        print('Error: It is necessary to pass input file (-i) or input directory (-d)')
        print('main.py [-i --ifile <inputfile>] [-d --dir <inputfolder>] [-o --ofile <outputfile>]')
        sys.exit(2)

    students_count = 0
    facilitators_count = 0
    co_facilitators_count = 0

    for file in files:
        document = docx.Document(docx=file)
        today = datetime.now()
        d1 = today.strftime("%Y%m%d%H%M%S")
        output_to = f'output/{output}_{d1}.csv'

        data_vector = []

        for line in document.paragraphs:
            line_text = line.text.replace('¿', '')
            line_text = unidecode.unidecode(line_text)
            role, text = process_line(line_text)

            if role is not None and text is not None:
                if text != '':
                    data_vector.append([role, text])

        df = DataFrame(data_vector, columns=['Role', 'Text'])
        # print(df)

        students_count += len(df[df['Role'] == 'Student'])
        facilitators_count += len(df[df['Role'] == 'Facilitator'])
        co_facilitators_count += len(df[df['Role'] == 'Co-Facilitator'].value_counts())

    total = students_count + facilitators_count + co_facilitators_count

    print(f'Students Count: {students_count} ({students_count * 100 / total:.2f}%)')
    print(f'Facilitators Count: {facilitators_count} ({facilitators_count * 100 / total:.2f}%)')
    print(f'CoFacilitators Count: {co_facilitators_count} ({co_facilitators_count * 100 / total:.2f}%)')

    df.to_csv(output_to, index=False)

    output_filename = f'The CSV file has been created at {output_to}'
    print('\n' + '#' * len(output_filename))
    print(output_filename)


def process_line(line: str):
    role = ''
    index = line.find(':')

    if index != -1:
        first_part = line[0:index]
        second_part = line[index + 1:].lower().strip()

        for pattern in non_role:
            m = re.search(pattern, first_part)

            if m is not None:
                return None, None

        keep_searching = True

        for pattern in facilitators:
            m = re.search(pattern, first_part)

            if m is not None:
                role = 'Facilitator'
                keep_searching = False

        if keep_searching:
            for pattern in co_facilitators:
                m = re.search(pattern, first_part)

                if m is not None:
                    role = 'Co-Facilitator'
                    keep_searching = False

        if keep_searching:
            for pattern in students:
                m = re.search(pattern, first_part)

                if m is not None:
                    role = 'Student'
                    keep_searching = False

        if keep_searching:
            role = 'Student'

        second_part = remove_spanish(second_part)

        if second_part is None:
            return None, None

        return role, second_part
    return None, None


def remove_spanish(text: str):
    processed_text = text.replace('(inaudible)', '').replace('...', '')
    if processed_text.find("(") != -1:
        processed_text = processed_text[processed_text.find("(")+1:processed_text.find(")")]
    processed_text = re.sub("[\[].*?[\]]", "", processed_text)

    index = processed_text.find('//')
    if index != -1:
        parts = processed_text.split('//')
        sub_parts = []

        for part in parts:
            try:
                part = part.strip()
                if part != '' and detect(part) == 'en':
                    sub_parts.append(part)
            except:
                return None

        processed_text = ' '.join(sub_parts)
        return processed_text

    try:
        if detect(processed_text) != 'es':
            return processed_text
    except LangDetectException:
        return None

    return None


def process_params(argv):
    input_file = ''
    output_file = ''
    input_dir = ''

    try:
        opts, args = getopt.getopt(argv, "hi:d:o:", ["ifile=", "dir=", "ofile="])
        if len(opts) == 0:
            print('main.py [-i --ifile <inputfile>] [-d --dir <inputfolder>] [-o --ofile <outputfile>]')
            sys.exit(2)
    except getopt.GetoptError:
        print('main.py [-i --ifile <inputfile>] [-d --dir <inputfolder>] [-o --ofile <outputfile>]')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('main.py [-i --ifile <inputfile>] [-d --dir <inputfolder>] [-o --ofile <outputfile>]')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-d", "--dir"):
            input_dir = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg

    return input_file, input_dir, output_file


if __name__ == '__main__':
    input_file, input_dir, output_file = process_params(sys.argv[1:])
    load_roles()
    process_file(input_file, input_dir, output_file)

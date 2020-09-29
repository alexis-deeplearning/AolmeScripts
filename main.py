#!/usr/bin/python
import getopt
import sys
import docx
import json
import re
import unidecode
import csv

from datetime import datetime
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

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


def process_file(file_name: str, output: str):
    document = docx.Document(docx=file_name)
    today = datetime.now()
    d1 = today.strftime("%Y%m%d%H%M%S")
    output_to = f'output/{output}_{d1}.csv'

    with open(output_to, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Text", "Role"])

        for line in document.paragraphs:
            line_text = line.text.replace('¿', '')
            line_text = unidecode.unidecode(line_text)
            role, text = process_line(line_text)

            if role is not None and text is not None:
                if text != '':
                    writer.writerow([role, text])
                    print(f'{role} : {text}')

    print('\n#############################################')
    print(f'The CSV file has been created at {output_to}')


def process_line(line: str):
    role = ''
    index = line.find(':')

    if index != -1:
        first_part = line[0:index]
        second_part = line[index + 1:].lower()

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
    processed_text = re.sub("([\(\[]).*?([\)\]])", "", processed_text)

    index = processed_text.find('(')
    if index != -1:
        processed_text = processed_text[index + 1:]
        processed_text = processed_text.replace(')', '')

        return processed_text

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
        if detect(processed_text) == 'en':
            return processed_text
    except LangDetectException:
        return None

    return None


def process_params(argv):
    input_file = ''
    output_file = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
        if len(opts) == 0:
            print('main.py -i <inputfile> -o <outputfile>')
            sys.exit(2)
    except getopt.GetoptError:
        print('main.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg

    return input_file, output_file


if __name__ == '__main__':
    input_file, output_file = process_params(sys.argv[1:])
    load_roles()
    process_file(input_file, output_file) #'data/TNS-G-C2L1P-Apr12-C-Issac_q2_01-06.docx')

from tqdm import tqdm
import docx
import json
import re

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


def load_file(file_name: str):
    document = docx.Document(docx=file_name)
    for line in document.paragraphs:

        role, text = process_line(line.text)

        if role is not None and text is not None:
            print(f'{role} : {text}')


def process_line(line: str):
    role = ''
    index = line.find(':')

    if index != -1:
        first_part = line[0:index]
        second_part = line[index + 1:]

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

        # print(f'{first_part} : {role}')

        second_part = second_part.lower()
        second_part = second_part.replace('(inaudible)', '')

        index = second_part.find('(')
        if index == -1:
            return role, second_part

        second_part = second_part[index + 1:]
        second_part = second_part.replace(')', '')

        return role, second_part
    return None, None


if __name__ == '__main__':
    load_roles()
    load_file('data/TNS-G-C2L1P-Apr12-C-Issac_q2_01-06.docx')

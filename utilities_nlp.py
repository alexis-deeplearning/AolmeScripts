import re
import json
import string
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def remove_punctuation(text: str):
    no_punctuation = [words for words in text if words not in string.punctuation]
    words_no_punctuation = ''.join(no_punctuation)
    return words_no_punctuation


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


def load_roles():
    students = []
    facilitators = []
    co_facilitators = []
    non_role = []

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
            non_role.append(p)

    return students, co_facilitators, facilitators, non_role


def process_line(line: str, students, co_facilitators, facilitators, non_role):
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

        return role, remove_punctuation(second_part)
    return None, None

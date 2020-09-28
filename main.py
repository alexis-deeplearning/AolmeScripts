from tqdm import tqdm
import docx
import json

students = []
facilitators = []
co_facilitators = []


def load_roles():
    with open('roles.json') as json_file:
        data = json.load(json_file)

        for p in data['Student']:
            students.append(p)

        for p in data['Facilitator']:
            facilitators.append(p)

        for p in data['CoFacilitator']:
            co_facilitators.append(p)


def load_file(file_name: str):
    document = docx.Document(docx=file_name)
    for i in document.paragraphs:
        print(i.text)


#    for i in tqdm(range(int(9e7))):
#        pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load_roles()
    load_file('data/TNS-G-C2L1P-Apr12-C-Issac_q2_01-06.docx')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
